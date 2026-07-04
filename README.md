# Jan API Gateway / Proxy (FastAPI)

這是一個使用 **FastAPI** 重寫的 **Jan API 伺服器** 代理閘道。它可以將本地端 Anthropic Claude API 規格（如 `/v1/messages`）無縫轉換為 OpenAI 規格（如 `/v1/chat/completions`）並轉發給 **國網中心 (NCHC Inner Medusa)** 或是本地模型（如 `llama.cpp`），並且完整支援：

- **動態別名對應**：將 `haiku`/`sonnet`/`opus` 等別名自動映射到國網中心真實的模型（如 MiniMax-M2.7, MiniMax-M3 等）。
- **進階 Claude Code 相容性**：內建針對 Claude Code 的 PDF 文件解析轉換與 Tool-Call 訊息順序校正。
- **Swagger UI 測試文件**：啟動後可在瀏覽器中直接進行可視化 API 測試。
- **免金鑰認證切換**：可自由開啟或關閉本機服務的 API Key 保護。

---

## 📂 專案目錄結構

```text
fastapi/
├── main.py                # FastAPI 核心服務與路由
├── config.py              # 設定檔載入模組 (環境變數與絕對路徑解析)
├── transformer.py         # 雙向格式轉換器 (OpenAI <-> Anthropic)
├── providers.json         # 遠端 API 上游供應商 (如國網中心) 與 API Key 配置
├── models_portal.json     # 模型別名與工具停用 (disable_tools) 配置
├── requirements.txt       # 依賴套件清單
└── .env                   # 系統環境變數配置
```

---

## 🛠️ 安裝與設定步驟

### 1. 安裝環境與套件
在 Windows PowerShell 中，切換至 `fastapi` 目錄，建立虛擬環境並安裝依賴：

```powershell
# 1. 切換至 fastapi 目錄
cd d:\antigravity\jan\fastapi

# 2. 建立 Python 虛擬環境 (使用 uv)
uv venv

# 3. 啟用虛擬環境
.\.venv\Scripts\activate

# 4. 安裝依賴套件 (使用 uv)
uv pip install -r requirements.txt
```

---

### 2. 設定統一配置檔 (`config.json`)

請直接將 `config.json.example` 複製一份並命名為 `config.json`，然後編輯該檔案填入您的 API 金鑰：

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 1337,
    "prefix": "/v1",
    "api_key": "sk-1234567890987654321",
    "proxy_timeout": 600
  },
  "providers": {
    "nchc_portal": {
      "base_url": "https://inner-medusa.genai.nchc.org.tw/v1",
      "api_keys": [
        "YOUR_NCHC_PORTAL_API_KEY_HERE"  // <-- 在此填寫國網真實 Key
      ]
    }
  },
  "models": [
    {
      "id": "MiniMax-M2.7",
      "provider": "nchc_portal",
      "backend_model": "MiniMax-M2.7",
      "display_name": "MiniMax-M2.7",
      "aliases": [
        "haiku",
        "claude-haiku-4-5",
        "m7",
        "claude-3-5-haiku-20241022",
        "claude-3-5-haiku",
        "claude-3-haiku-20240307",
        "claude-3-haiku"
      ]
    },
    {
      "id": "MiniMax-M3",
      "provider": "nchc_portal",
      "backend_model": "MiniMax-M3",
      "display_name": "MiniMax-M3",
      "aliases": [
        "sonnet",
        "claude-sonnet-4-6",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet",
        "claude-3-sonnet-20240229",
        "claude-3-sonnet"
      ]
    },
    {
      "id": "GLM-5.2",
      "provider": "nchc_portal",
      "backend_model": "GLM-5.2",
      "display_name": "GLM-5.2",
      "disable_tools": true,
      "default_tool_choice": "auto",
      "aliases": [
        "opus",
        "claude-opus-4-7",
        "claude-3-opus-20240229",
        "claude-3-opus"
      ]
    },
    {
      "id": "NVIDIA-Nemotron-3-Super-120B-A12B",
      "provider": "nchc_portal",
      "backend_model": "NVIDIA-Nemotron-3-Super-120B-A12B",
      "display_name": "NVIDIA-Nemotron-3-Super-120B-A12B",
      "aliases": [
        "nemotron-super",
        "super"
      ]
    }
  ]
}
```

> [!NOTE]
> 如果您想免除本地 API Key 認證，可以將 `"api_key"` 欄位直接設為空字串 `""`。您也可以隨時藉由設定系統環境變數（如 `PORT`、`HOST`、`API_KEY`）來覆蓋 JSON 檔案中的相應設定。

### ⚙️ 進階模型設定 (工具呼叫控制)
在 `"models"` 的列表中，您可以針對特定不支持 Function Calling (工具呼叫) 的後端模型（例如 `GLM-5.2`）進行進階控制：
* **`"disable_tools": true`** *(選填)*：設定為 `true` 後，本地 Proxy 在向後端發送請求前，會自動將 `tools` 與 `tool_choice` 參數剔除，防止不支持該功能的最底層模型崩潰。
* **`"default_tool_choice": "auto"`** *(選填)*：設定預設的工具呼叫模式。當客戶端傳入 `tools` 卻未指定 `tool_choice` 時，伺服器會自動補上此欄位。

---

## 🚀 啟用服務

請開啟您的終端機 (PowerShell)，並切換至 `fastapi` 工作目錄，選擇以下任一方式啟動：

### 方式一：直接執行虛擬環境中的 Python (最簡潔)
```powershell
.venv\Scripts\python main.py
```

### 方式二：先啟用虛擬環境再執行
```powershell
.\.venv\Scripts\activate
python main.py
```

* **API 文件位址 (Swagger UI)**: 用瀏覽器打開 [http://127.0.0.1:1337/](http://127.0.0.1:1337/) 即可進行可視化測試。
* **模型列表 API**: [http://127.0.0.1:1337/v1/models](http://127.0.0.1:1337/v1/models)

---

## 💬 整合 Claude Code 客戶端使用

欲使 **Claude Code** 使用此代理連線至國網中心，您可以直接複製本專案中的 `claude_settings.json.example` 內容，並覆蓋貼到您使用者目錄下的 `~/.claude.json` 檔案中（在 Windows 上的實際路徑為 `C:\Users\<您的使用者名稱>\.claude.json`）：

```json
{
  "env": {
    "ANTHROPIC_API_KEY": "anything",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:1337",
    "NO_PROXY": "localhost,127.0.0.1",
    "no_proxy": "localhost,127.0.0.1",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.7",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME": "MiniMax-M2.7",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M3",
    "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME": "MiniMax-M3",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "GLM-5.2",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME": "GLM-5.2",
    "ANTHROPIC_MODEL": "MiniMax-M2.7",
    "CLAUDE_CODE_DISABLE_THINKING": "1",
    "ANTHROPIC_DISABLE_THINKING": "1",
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  },
  "model": "haiku"
}
```

這會讓 Claude Code 預設選用 `haiku`（映射至國網 `MiniMax-M2.7`）並通過 Proxy 發送。在 Claude Code 聊天中可隨時輸入 `/model sonnet`、`/model opus` 或 `/model ultra` (以及任何在 `config.json` 中定義的模型別名) 切換至其他模型。

---

## 🌐 整合 Claude Desktop Gateway (如 LibreChat) 使用

如果您使用如 **LibreChat** 等圖形化客戶端，並希望透過自訂 Anthropic Gateway 對接此代理服務，請依照以下設定值配置：

### 1. 閘道憑證設定 (Gateway Credentials)
- **Gateway base URL**: `http://127.0.0.1:1337`
- **Gateway API key**: `sk-1234567890987654321` *(若 `config.json` 中 `api_key` 為空則可留空)*
- **Gateway auth scheme**: `bearer`
- **Credential kind**: `Static API key`
- **Custom inference headers**: *(留空，不要新增任何自訂標頭)*

### 2. 模型清單對應 (Models Override)
請在客戶端自訂模型清單（Model list）中進行以下設定：

| Model ID (請求的模型 ID) | Display Name (顯示名稱) |
| :--- | :--- |
| `claude-haiku-4-5` | `MiniMax-M2.7` |
| `claude-sonnet-4-6` | `MiniMax-M3` |
| `claude-opus-4-7` | `GLM-5.2` |

---

## 🧪 常用測試指令 (PowerShell)

> [!TIP]
> 如果您已在 `config.json` 中將 `"api_key"` 設為 `""`（免金鑰模式），以下測試指令中的金鑰部分（如 `Bearer ...` 或 `X-Api-Key`）可以填入任何字串（例如 `anything` 或是 `jan`），伺服器都會直接放行。

### 測試模型清單
```powershell
curl.exe -X GET http://127.0.0.1:1337/v1/models `
  -H "Authorization: Bearer sk-1234567890987654321"
```

### 測試對話生成 (OpenAI 規格)
```powershell
curl.exe -X POST http://127.0.0.1:1337/v1/chat/completions `
  -H "Authorization: Bearer sk-1234567890987654321" `
  -H "Content-Type: application/json" `
  -d '{\"model\": \"haiku\", \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}]}'
```

### 測試串流對話 (Anthropic 規格)
```powershell
curl.exe -X POST http://127.0.0.1:1337/v1/messages `
  -H "X-Api-Key: sk-1234567890987654321" `
  -H "Content-Type: application/json" `
  -d '{\"model\": \"haiku\", \"messages\": [{\"role\": \"user\", \"content\": \"請說1到3\"}], \"stream\": true}'
```
