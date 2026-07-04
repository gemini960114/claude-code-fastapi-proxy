import json
import os
import re
from typing import List, Dict, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class ProviderCustomHeader(BaseModel):
    header: str
    value: str

class ProviderConfig(BaseModel):
    provider: str
    api_key: Optional[str] = None
    api_keys: List[str] = []
    base_url: Optional[str] = None
    custom_headers: List[ProviderCustomHeader] = []
    models: List[str] = []

    def bearer_key_chain(self) -> List[str]:
        if self.api_keys:
            return self.api_keys
        if self.api_key:
            return [self.api_key]
        return []

class Settings(BaseSettings):
    # Fallback default values
    _default_host: str = "127.0.0.1"
    _default_port: int = 1337
    _default_prefix: str = "/v1"
    _default_api_key: str = ""
    _default_proxy_timeout: int = 600

    trusted_hosts: List[str] = ["*"]
    llamacpp_url: str = "http://127.0.0.1:3928"
    mlx_url: str = "http://127.0.0.1:8080"
    jan_data_folder: str = os.path.expanduser("~/jan")
    enable_server_tool_execution: bool = False

    # Cached config data
    _config_data: Optional[dict] = None

    def load_unified_config(self) -> dict:
        if self._config_data is not None:
            return self._config_data

        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "config.json")
        if not os.path.exists(path):
            path = os.path.join(base_dir, "config.json.example")

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._config_data = json.load(f)
            except Exception as e:
                print(f"Error loading config.json from {path}: {e}")
        
        if not self._config_data:
            self._config_data = {"server": {}, "providers": {}, "models": []}

        return self._config_data

    def get_server_setting(self, name: str, default: any) -> any:
        # 1. Check environment variables first (higher priority)
        env_val = os.environ.get(name.upper())
        if env_val is not None:
            if isinstance(default, bool):
                return env_val.lower() == "true"
            if isinstance(default, int):
                return int(env_val)
            return env_val

        # 2. Check config.json
        cfg = self.load_unified_config()
        return cfg.get("server", {}).get(name, default)

    # Properties to allow main.py to access settings directly with settings.port, etc.
    @property
    def host(self) -> str:
        return self.get_server_setting("host", self._default_host)

    @property
    def port(self) -> int:
        return self.get_server_setting("port", self._default_port)

    @property
    def prefix(self) -> str:
        return self.get_server_setting("prefix", self._default_prefix)

    @property
    def api_key(self) -> str:
        return self.get_server_setting("api_key", self._default_api_key)

    @property
    def proxy_timeout(self) -> int:
        return self.get_server_setting("proxy_timeout", self._default_proxy_timeout)

    def load_providers_config(self) -> Dict[str, ProviderConfig]:
        configs = {}
        cfg = self.load_unified_config()
        providers_data = cfg.get("providers", {})

        for p_name, p_data in providers_data.items():
            # Gather models linked to this provider
            p_models = []
            for m in cfg.get("models", []):
                if m.get("provider") == p_name:
                    p_models.append(m.get("backend_model") or m.get("id"))

            configs[p_name] = ProviderConfig(
                provider=p_name,
                api_key=p_data.get("api_key"),
                api_keys=p_data.get("api_keys", []),
                base_url=p_data.get("base_url"),
                custom_headers=p_data.get("custom_headers", []),
                models=p_models
            )

        # Allow environment override for testing
        env_providers = os.environ.get("PROVIDERS_CONFIG")
        if env_providers:
            try:
                data = json.loads(env_providers)
                if isinstance(data, dict):
                    for k, v in data.items():
                        configs[k] = ProviderConfig(**v)
            except Exception as e:
                print(f"Error parsing PROVIDERS_CONFIG environment variable: {e}")

        return configs

    def load_model_config(self) -> dict:
        cfg = self.load_unified_config()
        return {"models": cfg.get("models", [])}

    def map_model(self, model: str) -> str:
        import re
        normalized_model = re.sub(r"-\d{8}$", "", model)
        cfg = self.load_unified_config()
        for entry in cfg.get("models", []):
            names = [entry.get("id"), *entry.get("aliases", [])]
            if model in names or normalized_model in names:
                return entry.get("backend_model") or entry.get("id")
        return model

settings = Settings()
