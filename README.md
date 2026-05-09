# 📈 Marketing Intelligence Suite

A unified Streamlit application that combines three marketing-analytics modules into a single entry point:

- **📊 Analysis** — interactive campaign EDA dashboard with KPIs, drill-downs, and report export.
- **💬 Chatbot** — natural-language Q&A over a campaign CSV, powered by Gemini.
- **🎯 Recommendation** — pre-campaign KPI prediction, in-flight optimization decisions, post-campaign success classification.

---

## 📂 Project Structure

```
data-science/
├── app.py                       # Unified Streamlit entry point (sidebar nav)
├── requirements.txt             # Consolidated dependencies
├── .env.example                 # Template for API keys
├── .gitignore
├── README.md
├── REPORT.md                    # Technical report
│
├── data/
│   └── Cleaned_Social_Media_Advertising.csv
│
├── analysis/                    # EDA dashboard module
│   ├── __init__.py              # exposes render()
│   ├── app.py                   # Streamlit UI (render())
│   ├── clean_script.py          # Data-cleaning utilities
│   └── README.md                # Module-specific notes
│
├── chatbot/                     # NL → pandas chatbot
│   ├── __init__.py              # exposes render()
│   ├── app.py                   # Streamlit UI (render())
│   └── engine.py                # MarketingAnalyst (Gemini)
│
└── recommendation/              # ML recommendation module
    ├── __init__.py              # exposes render()
    ├── app.py                   # Streamlit UI with 3 sub-tabs
    ├── common/                  # Shared data prep, scoring, normalization
    ├── pre_campaign/            # XGBoost KPI predictor + saved model
    │   └── models/
    ├── in_flight/               # KEEP/OPTIMIZE/STOP decision engine
    │   └── ml/                  # ROI + conversion forecasters
    ├── post_campaign/           # PCA-weighted success classifier
    └── llm/                     # Gemini/Groq client + prompt templates
```

---

## 🚀 Quick Start

### 1. Clone and enter the project

```bash
git clone <your-repo-url> data-science
cd data-science
```

### 2. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
```

Edit `.env` and add at least one of:

```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

> ⚠️ **Security note**: never commit `.env`. The chatbot needs `GEMINI_API_KEY`; the recommendation LLM explainer prefers Gemini and falls back to Groq.

### 4. Run the unified app

```bash
streamlit run app.py
```

The sidebar lets you switch between Analysis, Chatbot, and Recommendation.

---

## 🧭 Module Guide

### 📊 Analysis

- Drag-and-drop a campaign CSV or use the bundled sample at `data/Cleaned_Social_Media_Advertising.csv`.
- Filter by company, platform, interest, geography, age, gender, objective, language, status, ROI, and budget.
- View KPI cards, trend lines, platform/geo/demographic breakdowns, and generate downloadable reports (TXT / HTML / CSV).

### 💬 Chatbot

1. Upload a CSV (auto-cleaned: column normalization, currency stripping, date parsing).
2. Ask in English or Arabic — e.g. *"which platform has the highest ROI by interest?"*.
3. The model writes pandas code, executes it server-side, and returns a dataframe or text.

### 🎯 Recommendation

| Sub-tab | Input | Output |
|---|---|---|
| **Pre-Campaign** | Platform, audience, budget, duration | Predicted impressions, clicks, conversions, revenue, ROI, CTR, CPA |
| **In-Flight** | Snapshot CSV of running campaigns | KEEP / OPTIMIZE / STOP per campaign + LLM summary |
| **Post-Campaign** | Finished-campaign CSV | Success classification + failure-driver analysis |

---

## 🛠️ Development Notes

- All three modules expose a `render()` function; the top-level `app.py` dispatches based on sidebar selection.
- `st.set_page_config` is called **once** in `app.py` — module render functions must not call it.
- Pre-campaign ML artifacts (`model.pkl`, `scaler.pkl`, `label_encoders.pkl`) live in `recommendation/pre_campaign/models/`.
- In-flight ML artifacts (`roi_model.pkl`, `conversion_model.pkl`, `scaler.pkl`) live in `recommendation/in_flight/ml/`.
- The recommendation LLM client (`recommendation/llm/client.py`) tries Gemini first, then Groq, then returns a static fallback.

---

## 📄 Documentation

- **[REPORT.md](REPORT.md)** — full technical report: architecture, models, methodology, results.
- **[analysis/README.md](analysis/README.md)** — analysis dashboard details.

---

## 👥 Contributors

- Data & analysis dashboard
- Chatbot engine
- Recommendation models (pre / in-flight / post)
