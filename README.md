# DualSource Data Agent

Portfolio project: a streaming **data agent** that answers questions from **your internal warehouse** and **public web sources**.

- **Internal:** local DuckDB loaded from small seed CSVs (fictional Aether Mobility hubs)
- **External:** Tavily web search
- **LLM:** Azure OpenAI / Microsoft Foundry (your own keys)
- **UI:** React chat with charts, citation chips, session history, Excel/PPT export

> Scrubbed portfolio rebuild. No Databricks connection and no real client production data.

## Prerequisites

- Python 3.11+ (3.12/3.13/3.14 also fine)
- Node.js 18+
- Azure OpenAI / Foundry API key
- Tavily API key

## Local setup (Windows)

```powershell
# from repo root
cd dualsource-data-agent

# 1) Python deps (venv recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app/backend/requirements.txt -r app/backend/requirements-langchain.txt

# 2) Build local DuckDB from committed seed CSVs
python data/load_duckdb.py

# 3) Secrets
copy .env.example .env
# Edit .env and fill:
#   AZURE_OPENAI_API_KEY=...
#   AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/openai/v1
#   TAVILY_API_KEY=...

# 4) Backend (terminal 1) — keep this running
cd appackend
python run.py
# health check: http://127.0.0.1:8000/api/health

# 5) Frontend (terminal 2) — keep this running
cd approntend
npm install
npm run dev
# open: http://localhost:5173
```

### Important URL note

Use **`http://localhost:5173`** (not only `http://127.0.0.1:5173` if the Vite process bound to IPv6).

Backend API: `http://127.0.0.1:8000`  
Frontend (Vite proxies `/api` → backend): `http://localhost:5173`

## Local setup (macOS / Linux)

```bash
cd dualsource-data-agent
python -m venv .venv
source .venv/bin/activate
pip install -r app/backend/requirements.txt -r app/backend/requirements-langchain.txt
python data/load_duckdb.py
cp .env.example .env   # then edit keys

# terminal 1
cd app/backend && python run.py

# terminal 2
cd app/frontend && npm install && npm run dev
```

## Seed data: committed CSVs, local DuckDB

| Path | In git? | Notes |
|---|---|---|
| `data/seed/*.csv` | **Yes** | Small demo fixtures (~48KB total) |
| `data/warehouse/*.duckdb` | **No** | Rebuilt locally via `python data/load_duckdb.py` |
| `.env` | **No** | Your Foundry + Tavily secrets |

### Tables

| Table | Grain | Key columns |
|---|---|---|
| `hub_kpi_monthly` | hub × month | passengers, cargo_tonnes, aircraft_movements |
| `route_kpi_monthly` | origin hub × dest × month | passengers, cargo_tonnes |
| `hub_retail_monthly` | hub × month × category | revenue_usd |
| `hub_otp_monthly` | hub × month | on_time_pct, avg_delay_minutes |

Hubs: **NVA**, **PRH**, **ATG**, **SKY** (fictional). Months: **2024-01 … 2025-12**.

## Demo questions

**Internal**
- How many passengers did Nova International (NVA) handle in 2025?
- Show NVA cargo trend for the last 6 months of 2025 as a chart
- What is Pacific Rim Hub retail revenue by category in 2025?

**Internal + external**
- Compare NVA passenger volumes with Singapore Changi using sources
- How does NVA cargo compare with published Shanghai Pudong figures?

**External-heavy**
- What are the latest public passenger figures for Singapore Changi and Incheon?

## Architecture

```
React UI  ->  FastAPI  ->  Local ReAct (LangChain harness)
                             |- DuckDB SQL tools
                             |- Tavily web_search
                             |- Foundry FAST/STRONG models
                             |- chart planner + answer synthesis
```

No model-serving / Databricks deploy path is included in this portfolio build.

## License / disclaimer

Demo fiction + your own API keys. Not affiliated with any airport authority.
