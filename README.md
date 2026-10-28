# DualSource Agent

Portfolio project: a streaming analytics agent that answers questions from **your internal warehouse** and **public web sources**.

- **Internal:** local DuckDB loaded from small seed CSVs (Aether Mobility fictional hubs)
- **External:** Tavily web search
- **LLM:** Azure OpenAI / Microsoft Foundry (your keys)
- **UI:** React chat with charts, citation chips, history, Excel/PPT export

> This repository is a scrubbed portfolio rebuild. It does **not** connect to Databricks and contains no client production data.

## Quick start (localhost)

```bash
# 1) Python deps
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r app/backend/requirements.txt -r app/backend/requirements-langchain.txt

# 2) Build DuckDB from seed CSVs (committed)
python data/load_duckdb.py

# 3) Configure secrets
cp .env.example .env
# fill AZURE_OPENAI_* and TAVILY_API_KEY

# 4) Backend
cd app/backend
python run.py
# -> http://127.0.0.1:8000

# 5) Frontend (new terminal)
cd app/frontend
npm install
npm run dev
# -> http://127.0.0.1:5173
```

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

## Seed schema (DuckDB)

| Table | Grain | Key columns |
|---|---|---|
| `hub_kpi_monthly` | hub × month | passengers, cargo_tonnes, aircraft_movements |
| `route_kpi_monthly` | origin hub × dest × month | passengers, cargo_tonnes |
| `hub_retail_monthly` | hub × month × category | revenue_usd |
| `hub_otp_monthly` | hub × month | on_time_pct, avg_delay_minutes |

Hubs: **NVA**, **PRH**, **ATG**, **SKY** (fictional). Months: 2024-01 … 2025-12.

Seed CSVs are intentionally small and committed under `data/seed/`. Rebuild the DB anytime with `python data/load_duckdb.py`.

## Architecture

```
React UI  ->  FastAPI  ->  Local ReAct (LangChain/LangGraph harness)
                             |- DuckDB SQL tools
                             |- Tavily web_search
                             |- Foundry FAST/STRONG models
                             |- chart planner + answer synthesis
```

No model serving / Databricks deploy path is included in this portfolio build.

## License / disclaimer

Demo fiction + your own API keys. Not affiliated with any airport authority.
