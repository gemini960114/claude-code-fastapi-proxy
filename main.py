import json
import os
import httpx
from fastapi import FastAPI, APIRouter, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional

from config import settings, ProviderConfig
from transformer import (
    transform_anthropic_to_openai,
    transform_openai_response_to_anthropic,
    normalize_openai_tools_in_chat_body,
    strip_billing_header_in_body,
    transform_openai_chunk_to_anthropic_events,
    StreamState,
    sse_event
)

app = FastAPI(
    title="Jan API Service",
    version="1.0",
    docs_url=None,       # Disable default Swagger UI
    redoc_url=None,      # Disable default ReDoc
    openapi_url=None     # We serve custom openapi.json
)

# Host & CORS settings
TRUSTED_HOSTS = settings.trusted_hosts
API_KEY = settings.api_key
PREFIX = settings.prefix

# Route whitelist (no host validation or authentication required)
WHITELISTED_PATHS = {
    "/",
    "/openapi.json",
    "/favicon.ico",
    "/docs/swagger-ui.css",
    "/docs/swagger-ui-bundle.js",
    "/docs/swagger-ui-standalone-preset.js",
}

# Mount Swagger UI assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(os.path.join(static_dir, "swagger-ui")):
    app.mount("/docs", StaticFiles(directory=os.path.join(static_dir, "swagger-ui")), name="swagger-ui-static")

# Custom CORS and Authentication Middleware
@app.middleware("http")
async def custom_middleware(request: Request, call_next):
    path = request.url.path
    
    # Strip prefix if present to compare against whitelist
    normalized_path = path
    if path.startswith(PREFIX):
        normalized_path = path[len(PREFIX):]
        
    is_whitelisted = normalized_path in WHITELISTED_PATHS

    # 1. Host Validation
    host_header = request.headers.get("host", "")
    if not is_whitelisted and "*" not in TRUSTED_HOSTS:
        host_clean = host_header.split(":")[0] if ":" in host_header else host_header
        if host_clean not in TRUSTED_HOSTS:
            return JSONResponse(status_code=403, content={"error": "Invalid host header"})

    # 2. CORS Preflight Handling (OPTIONS)
    if request.method == "OPTIONS":
        origin = request.headers.get("origin", "")
        # For simplicity, reflect the origin if it matches host checks (or if trusted_hosts allows all)
        cors_headers = {
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "accept, accept-language, authorization, cache-control, connection, content-type, dnt, host, if-modified-since, keep-alive, origin, user-agent, x-api-key, x-csrf-token, x-forwarded-for, x-forwarded-host, x-forwarded-proto, x-requested-with, x-stainless-arch, x-stainless-lang, x-stainless-os, x-stainless-package-version, x-stainless-retry-count, x-stainless-runtime, x-stainless-runtime-version, x-stainless-timeout",
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin, Access-Control-Request-Method, Access-Control-Request-Headers",
        }
        if origin:
            cors_headers["Access-Control-Allow-Origin"] = origin
            cors_headers["Access-Control-Allow-Credentials"] = "true"
        return Response(status_code=200, headers=cors_headers)

    # 3. Authentication Check
    if not is_whitelisted and API_KEY:
        auth_header = request.headers.get("authorization", "")
        api_key_header = request.headers.get("x-api-key", "")
        print(f"[DEBUG] Auth check: auth_header='{auth_header}', api_key_header='{api_key_header}', expected='{API_KEY}'", flush=True)
        
        auth_valid = False
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            auth_valid = (token == API_KEY)
        elif api_key_header:
            auth_valid = (api_key_header == API_KEY)
            
        if not auth_valid:
            return JSONResponse(status_code=401, content={"error": "Invalid or missing authorization token"})

    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal Server Error: {str(e)}"})

    # Inject CORS headers for normal response
    origin = request.headers.get("origin", "")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    return response


# --- ROOT & DOCS ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def get_swagger_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>API Docs</title>
      <link rel="stylesheet" type="text/css" href="/docs/swagger-ui.css">
    </head>
    <body>
      <div id="swagger-ui"></div>
      <script src="/docs/swagger-ui-bundle.js"></script>
      <script>
      window.onload = () => {
        SwaggerUIBundle({
          url: '/openapi.json',
          dom_id: '#swagger-ui',
        });
      };
      </script>
    </body>
    </html>
    """
    return html_content

@app.get("/openapi.json")
async def get_openapi_json(request: Request):
    path = os.path.join(static_dir, "openapi.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                openapi_spec = json.load(f)
            # Update servers list dynamically
            base_url = f"{request.url.scheme}://{request.url.netloc}{PREFIX}"
            openapi_spec["servers"] = [{"url": base_url, "description": "Jan API server"}]
            return openapi_spec
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="openapi.json not found")

@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    path = os.path.join(static_dir, "swagger-ui", "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Favicon not found")


# --- API ROUTER (/v1 prefix) ---

router = APIRouter(prefix=PREFIX)

def get_upstream_info(model_id: str, endpoint: str) -> tuple[str, list[str]]:
    """
    Decides the upstream target URL and authorization keys for a model.
    """
    providers = settings.load_providers_config()
    
    # 1. Match remote providers
    for provider_name, provider_cfg in providers.items():
        if model_id in provider_cfg.models:
            base_url = (provider_cfg.base_url or "https://api.openai.com/v1").rstrip("/")
            # Specific handling for Anthropic API endpoint
            if "anthropic" in provider_name.lower():
                target_url = f"{base_url}/messages"
            else:
                target_url = f"{base_url}{endpoint}"
            return target_url, provider_cfg.bearer_key_chain()
            
    # 2. Match MLX model convention
    if "mlx" in model_id.lower():
        target_url = f"{settings.mlx_url.rstrip('/')}{endpoint}"
        return target_url, []
        
    # 3. Fallback to llama.cpp
    target_url = f"{settings.llamacpp_url.rstrip('/')}{endpoint}"
    return target_url, []


@router.get("/models")
async def list_models():
    all_models = []
    
    # Try fetching from llama.cpp
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.llamacpp_url}/v1/models", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data:
                    all_models.extend(data["data"])
    except Exception:
        pass
        
    # Try fetching from MLX
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.mlx_url}/v1/models", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data:
                    all_models.extend(data["data"])
    except Exception:
        pass

    # Load models from configuration mappings (Claude custom models & aliases)
    try:
        model_config = settings.load_model_config()
        for entry in model_config.get("models", []):
            m_id = entry.get("id")
            if m_id:
                # Add the main ID
                all_models.append({
                    "id": m_id,
                    "object": "model",
                    "created": 1,
                    "owned_by": "nchc_portal",
                    "display_name": entry.get("display_name", m_id)
                })
                # Add all aliases
                for alias in entry.get("aliases", []):
                    all_models.append({
                        "id": alias,
                        "object": "model",
                        "created": 1,
                        "owned_by": "nchc_portal",
                        "display_name": f"{entry.get('display_name', m_id)} (alias)"
                    })
    except Exception:
        pass

    # Load remote models from configuration
    providers = settings.load_providers_config()
    for provider_name, provider_cfg in providers.items():
        for model_id in provider_cfg.models:
            # Prevent duplicating if already added
            if not any(m["id"] == model_id for m in all_models):
                all_models.append({
                    "id": model_id,
                    "object": "model",
                    "created": 1,
                    "owned_by": provider_name
                })
            
    return {"object": "list", "data": all_models}


@router.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    normalize_openai_tools_in_chat_body(body)
    strip_billing_header_in_body(body)
    
    requested_model = body.get("model")
    if not requested_model:
        raise HTTPException(status_code=400, detail="Request body must contain a 'model' field")

    model_id = settings.map_model(requested_model)
    body["model"] = model_id

    # Check for disable_tools and default_tool_choice in model config
    model_config = settings.load_model_config()
    model_entry = {}
    for entry in model_config.get("models", []):
        names = [entry.get("id"), *entry.get("aliases", [])]
        if requested_model in names or model_id in names:
            model_entry = entry
            break

    if model_entry.get("disable_tools"):
        if "tools" in body:
            del body["tools"]
        if "tool_choice" in body:
            del body["tool_choice"]
    elif model_entry.get("default_tool_choice") and "tools" in body and "tool_choice" not in body:
        body["tool_choice"] = model_entry.get("default_tool_choice")

    target_url, api_keys = get_upstream_info(model_id, "/chat/completions")
    print(f"[DEBUG] chat_completions: requested_model={requested_model} -> mapped_model={model_id} -> target_url={target_url}", flush=True)
    
    # Setup headers for forwarding
    headers = {"Content-Type": "application/json"}
    if api_keys:
        headers["Authorization"] = f"Bearer {api_keys[0]}"
        
    stream = body.get("stream", False)

    # 1. OpenAI-compatible backend forwarding
    if "/chat/completions" in target_url:
        if stream:
            async def sse_stream_generator():
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", target_url, json=body, headers=headers, timeout=settings.proxy_timeout) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk
            return StreamingResponse(sse_stream_generator(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(target_url, json=body, headers=headers, timeout=settings.proxy_timeout)
                return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
                
    # 2. Forward to Anthropic /messages but return OpenAI /chat/completions format
    elif "/messages" in target_url:
        # We need to translate body to Anthropic /messages format
        # ... Basic translation details
        # For simplicity, we can implement basic payload conversion if model points to Anthropic
        # But wait! If the user targeted chat/completions with an Anthropic model, we should translate.
        # Since Claude Code targets /messages directly, let's keep translation primary in the /messages route,
        # but handle basic forward translation here if needed.
        pass

    raise HTTPException(status_code=501, detail="Target routing not fully implemented for this route combo")


@router.post("/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    strip_billing_header_in_body(body)
    
    requested_model = body.get("model")
    if not requested_model:
        raise HTTPException(status_code=400, detail="Request body must contain a 'model' field")

    model_id = settings.map_model(requested_model)
    body["model"] = model_id

    # Check for disable_tools and default_tool_choice in model config
    model_config = settings.load_model_config()
    model_entry = {}
    for entry in model_config.get("models", []):
        names = [entry.get("id"), *entry.get("aliases", [])]
        if requested_model in names or model_id in names:
            model_entry = entry
            break

    if model_entry.get("disable_tools"):
        if "tools" in body:
            del body["tools"]
        if "tool_choice" in body:
            del body["tool_choice"]
    elif model_entry.get("default_tool_choice") and "tools" in body and "tool_choice" not in body:
        body["tool_choice"] = model_entry.get("default_tool_choice")

    target_url, api_keys = get_upstream_info(model_id, "/messages")
    print(f"[DEBUG] messages: requested_model={requested_model} -> mapped_model={model_id} -> target_url={target_url}", flush=True)
    headers = {"Content-Type": "application/json"}
    if api_keys:
        # If Anthropic provider, they require x-api-key header and anthropic-version
        if "api.anthropic.com" in target_url:
            headers["x-api-key"] = api_keys[0]
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {api_keys[0]}"

    stream = body.get("stream", False)

    # A. Natively supports Anthropic /messages (e.g. Anthropic upstream)
    if "api.anthropic.com" in target_url:
        if stream:
            async def sse_stream_generator():
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", target_url, json=body, headers=headers, timeout=settings.proxy_timeout) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk
            return StreamingResponse(sse_stream_generator(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(target_url, json=body, headers=headers, timeout=settings.proxy_timeout)
                return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    # B. Translate from Anthropic to OpenAI `/chat/completions` (e.g. local llama.cpp or OpenAI model)
    else:
        openai_body = transform_anthropic_to_openai(body)
        if not openai_body:
            raise HTTPException(status_code=400, detail="Failed to translate Anthropic body to OpenAI format")

        openai_target_url = target_url.replace("/messages", "/chat/completions")
        openai_headers = {"Content-Type": "application/json"}
        if api_keys:
            openai_headers["Authorization"] = f"Bearer {api_keys[0]}"

        if stream:
            async def translated_stream_generator():
                state = StreamState()
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", openai_target_url, json=openai_body, headers=openai_headers, timeout=settings.proxy_timeout) as r:
                        buffer = ""
                        async for chunk in r.aiter_text():
                            buffer += chunk
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                events = transform_openai_chunk_to_anthropic_events(line.strip(), state)
                                for event in events:
                                    yield event
            return StreamingResponse(translated_stream_generator(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(openai_target_url, json=openai_body, headers=openai_headers, timeout=settings.proxy_timeout)
                if resp.status_code != 200:
                    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
                try:
                    openai_resp = resp.json()
                    anth_resp = transform_openai_response_to_anthropic(openai_resp)
                    return JSONResponse(anth_resp)
                except Exception as e:
                    raise HTTPException(status_code=502, detail=f"Bad Gateway: translation error {str(e)}")


@router.post("/orchestrations")
async def orchestrations(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if body.get("stream", False):
        raise HTTPException(status_code=400, detail="stream=true is not supported for /orchestrations")

    messages_val = body.get("messages")
    if messages_val is None:
        raise HTTPException(status_code=400, detail="Missing required field 'messages'")
    if not isinstance(messages_val, list):
        raise HTTPException(status_code=400, detail="Request body must include 'messages' as an array")

    conversation_messages = messages_val.copy()

    # Load assistant config
    assistant_id = body.get("assistant_id")
    assistant_instructions = None
    assistant_model_hint = None
    if assistant_id:
        assistant_path = os.path.join(settings.jan_data_folder, "assistants", assistant_id, "assistant.json")
        if os.path.exists(assistant_path):
            try:
                with open(assistant_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    assistant_instructions = data.get("instructions")
                    assistant_model_hint = data.get("model")
            except Exception:
                pass

    if assistant_instructions:
        conversation_messages = [m for m in conversation_messages if m.get("role") != "system"]
        conversation_messages.insert(0, {
            "role": "system",
            "content": assistant_instructions
        })

    # Resolve model
    model_override = body.get("model")
    model_id = None
    if model_override:
        model_id = model_override
    elif assistant_model_hint and assistant_model_hint != "*":
        model_id = assistant_model_hint
    else:
        models_data = await list_models()
        models = models_data.get("data", [])
        if models:
            model_id = models[0]["id"]

    if not model_id:
        raise HTTPException(status_code=503, detail="No running model sessions available")

    target_url, api_keys = get_upstream_info(model_id, "/chat/completions")
    headers = {"Content-Type": "application/json"}
    if api_keys:
        headers["Authorization"] = f"Bearer {api_keys[0]}"

    max_turns = int(body.get("max_turns", 8))
    max_turns = max(1, min(20, max_turns))

    # Empty tools for now (MCP not running in FastAPI standalone)
    openai_tools = []
    
    last_response = None
    for turn in range(max_turns):
        completion_payload = {
            "model": model_id,
            "messages": conversation_messages,
            "stream": False,
        }
        if openai_tools:
            completion_payload["tools"] = openai_tools
            completion_payload["tool_choice"] = "auto"

        # Copy optional params if present
        for key in ["temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty"]:
            if key in body:
                completion_payload[key] = body[key]

        async with httpx.AsyncClient() as client:
            resp = await client.post(target_url, json=completion_payload, headers=headers, timeout=settings.proxy_timeout)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Upstream returned error: {resp.text}")
            
            completion = resp.json()

        last_response = completion

        choices = completion.get("choices", [])
        choice = choices[0] if choices else {}
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            return JSONResponse(completion)

        conversation_messages.append({
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": tool_calls
        })
        break

    if last_response:
        return JSONResponse(last_response)
    
    raise HTTPException(status_code=500, detail="Orchestration failed to generate a response")


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    if settings.api_key:
        masked_key = settings.api_key[:6] + "..." + settings.api_key[-4:] if len(settings.api_key) > 10 else settings.api_key
        print(f"INFO: Local API Key auth is ENABLED (Active Key: {masked_key})", flush=True)
    else:
        print("INFO: Local API Key auth is DISABLED (Anyone can access)", flush=True)
    uvicorn.run(app, host=settings.host, port=settings.port)
