# Technical Report — Marketing Intelligence Suite

> Architecture, models, and methodology behind the unified Streamlit application.

---

## 1. Overview

The Marketing Intelligence Suite consolidates three previously separate marketing-analytics tools into a single Streamlit application:

1. **Analysis dashboard** — descriptive EDA over historical social-media advertising data.
2. **Chatbot** — natural-language interface that translates user questions into pandas code and executes it against an uploaded dataframe.
3. **Recommendation models** — three predictive/prescriptive engines covering the campaign lifecycle: *pre-launch*, *in-flight*, *post-mortem*.

A shared dataset (`data/Cleaned_Social_Media_Advertising.csv`) underpins the EDA and the example workflows.

---

## 2. Architecture

### 2.1 Module layout

The codebase is organized as three sibling Python packages, each exposing a single `render()` entry point that builds its own Streamlit UI. The top-level `app.py` is a thin dispatcher with a sidebar radio that imports and invokes the chosen module.

```
app.py            ── sidebar nav, page config, dispatch
  └── analysis/render()       ── ~2000 LOC dashboard
  └── chatbot/render()        ── chat-style UI on top of MarketingAnalyst
  └── recommendation/render() ── three st.tabs (pre / in-flight / post)
```

### 2.2 Why one entry point

- `st.set_page_config` can be called only once per session — centralizing it avoids ordering bugs.
- Shared sidebar real estate stays consistent across modules.
- Keeps three pip-installable packages independent: each can be unit-tested or imported elsewhere without dragging Streamlit page state with it.

### 2.3 Configuration

Secrets (`GEMINI_API_KEY`, `GROQ_API_KEY`) load via `python-dotenv` from a project-root `.env`. The repo ships `.env.example`; `.env` is gitignored.

---

## 3. Data

### 3.1 Source

Cleaned social-media advertising records (~ 60 MB, in `data/`). Columns include campaign metadata (`Campaign_Name`, `Platform_Name`, `Start_Date`, `End_Date`), targeting (`Age_Group`, `Gender`, `Location`, `interest`, `Language`, `Objective`), spend (`Total_Budget`, `Budget_Spent`, `CPC`, `CPA`), and outcomes (`Impressions`, `Clicks`, `Conversions`, `Revenue`, `ROI`, `CTR`, `Conversion_Rate`).

### 3.2 Cleaning pipeline

`recommendation/common/data_prep.py::prepare_data` is the canonical cleaner:

- Strips whitespace from column names.
- Coerces `Start_Date` / `End_Date` to datetime; derives `duration_days` if missing.
- Coerces all numeric KPI/spend columns via `pd.to_numeric(..., errors="coerce")`.
- Drops exact duplicates (keyed on `campaign_ID`, `Start_Date`, `End_Date` when present).
- Filters rows with negative or NaN ROI / Conversion_Rate.
- Drops rows missing Impressions or Clicks.

The chatbot has its own lighter normalization in `MarketingAnalyst._preprocess_data` (lowercase column names, currency-symbol stripping, date parsing) so users can drop in arbitrary CSVs.

---

## 4. Analysis Dashboard

### 4.1 Behavior

- Procedural Streamlit app wrapped in `analysis.app.render()`.
- Sidebar holds the data source toggle (default file vs upload), date range, and ten cascading multi-select filters (company, campaign, platform, interest, location, age, gender, objective, language, status) plus ROI and budget sliders.
- Main body renders KPI cards, dual-axis Plotly charts (Revenue vs ROI), treemaps (Language), sunbursts (Geo × Platform), demographic heatmaps, and a comprehensive auto-generated report with TXT / HTML / CSV download.

### 4.2 Derived features

`Profit = Revenue − Budget_Spent`, `Budget_Utilization = Budget_Spent / Total_Budget × 100`, plus calendar features (`Month`, `Quarter`, `Year`, `Campaign_Duration`).

---

## 5. Chatbot — `chatbot/`

### 5.1 Pattern

Code-generation agent. Given a user question, the LLM is asked to write **pandas code** that assigns a result to the variable `result`. The code is extracted from a fenced Python block and executed via `exec` against the in-memory dataframe.

### 5.2 Prompt design (engine.py)

System instruction enforces:

- `result` must be a `pd.DataFrame` or `str` (never a scalar) — keeps the UI uniform.
- `df = df[df['campaign_name'].notna()]` is always applied first.
- Reply language matches the user (English / Arabic).
- A tight metric mapping (Spend → `amount_spent_…`, ROI formula, etc.) so the model targets the right columns.

A sliding window of the last 6 messages is included as context, with assistant messages truncated to 800 chars to control prompt size.

### 5.3 Model selection

`_get_best_available_model` enumerates Gemini models supporting `generateContent` and prefers `flash`, then `pro` (non-vision), with `gemini-pro` as a hard fallback.

### 5.4 Memory model

In Streamlit mode, conversation state lives in `st.session_state.chat_history` (capped at 20 turns). The original FastAPI version (now removed) used a `SESSION_STORE` keyed by UUID; Streamlit's per-tab session state replaces that.

### 5.5 Trade-offs / known risks

- Executing model-generated code is powerful but trusts the LLM. Run only on data you control; do not deploy the chatbot to untrusted users without sandboxing.
- Errors in generated code are caught and returned as text rather than crashing the UI.

---

## 6. Recommendation Models — `recommendation/`

### 6.1 Pre-campaign predictor

**Goal:** estimate Impressions, Clicks, Conversions, Revenue, and Spend for a *not-yet-launched* campaign given platform, interest, audience, budget, and duration.

- **Model:** persisted XGBoost multi-output regressor (`recommendation/pre_campaign/models/model.pkl`).
- **Features (8):** `Platform_Name`, `interest`, `Total_Budget`, `Age_Group`, `duration_days`, `Gender`, `Objective`, `Daily_Budget`.
- **Categoricals** are label-encoded with the trained `label_encoders.pkl`; unknown values fall back to the first known class.
- **Numerics** (`Total_Budget`, `Daily_Budget`) scaled with the saved `scaler.pkl`.
- **Targets** are stored in log space — `np.expm1` reverses on prediction; outputs are floored at 1.0 to avoid divide-by-zero in derived KPIs.
- **Derived KPIs:** Conversion_Rate, ROI, CTR, CPC, CPA computed from the five raw predictions.
- **Platform fallback table** maps unseen platforms (e.g. `tiktok ads → instagram`, `linkedin ads → facebook`) so the model degrades gracefully.

### 6.2 In-flight optimizer

**Goal:** for each currently-running campaign, decide **KEEP / OPTIMIZE / STOP**.

Pipeline (`recommendation/in_flight/engine.py`):

1. **Feature enrichment** (`features.py`): compute `time_elapsed_ratio`, `budget_spent_ratio`, current `ctr`, `conversion_rate`, `cpc`, `roi`.
2. **Min-max normalization** of `roi`, `ctr`, `conversion_rate`.
3. **ML forecast** (`ml/predict.py`) of `expected_final_roi` and `expected_final_conversions` using two saved RandomForestRegressors trained on synthetic snapshots of historical campaigns at 25% / 50% / 75% elapsed.
4. **Three component scores:**
   - `performance_score` = mean of normalized ROI / CTR / Conversion (recent quality).
   - `pace_score` = `1 − |budget_spent_ratio − time_elapsed_ratio|` (over- or under-spend penalty).
   - `potential_score` = `1 − time_elapsed_ratio` (room left to course-correct).
5. **PCA weighting** (`pca_weights.py`): the first principal component of the (campaigns × 3 scores) matrix gives an unsupervised weight per dimension. Loadings are abs-normalized to sum to 1 — adapts to whatever variance dominates the current portfolio.
6. **Decision threshold** (`decision.py`): final_score ≥ 0.7 → KEEP; ≥ 0.4 → OPTIMIZE; else STOP.
7. **Optional LLM summary** in Egyptian Arabic via `explain_inflight_summary_from_stats`.

### 6.3 Post-campaign auditor

**Goal:** classify finished campaigns as *successful / average / failed* and explain *why* failures failed.

Pipeline (`recommendation/post_campaign/engine.py`):

1. `prepare_data` → drop rows missing the four success features (ROI, Conversion_Rate, CTR, CPA).
2. Min-max normalize, then **invert CPA** (lower is better → higher score).
3. **PCA weights** over the normalized success features.
4. **Success_Score** = weighted sum.
5. **Classification** via empirical quantiles: `≥ Q75 → successful`, `≥ Q40 → average`, else `failed`.
6. **Driver analysis** (`driver_analysis.py`): for each numerical driver (`Total_Budget`, `CPC`, `duration_days`, `engagement_score`) compare mean of failed vs successful; for each categorical driver (`Platform_Name`, `Objective`, `Age_Group`, `Gender`, `Location`, `Language`) compare modal value.
7. The driver dictionary is rendered into a long-form prompt and sent to the LLM for a strategic Arabic summary.

### 6.4 LLM service — `recommendation/llm/`

- **Failover order:** Gemini (`gemini-2.5-flash` → `gemini-pro`) → Groq (`llama-3.3-70b-versatile`) → static fallback string.
- **Prompt safety:** `normalize_prompt` truncates to ≤ 1400 chars (≈ 350 tokens).
- **Prompt templates** are domain-specific (failed-summary, failure-driver, in-flight-summary, pre-campaign) and instruct the model to write in Egyptian-business Arabic without quoting raw numbers.

---

## 7. Methodology Notes

### 7.1 Why PCA for weighting?

For both in-flight (3 component scores) and post-campaign (4 success features) we want an *unsupervised* weighting that reflects which dimension drives the variance in the current dataset. Hand-tuned weights would lock in a single user's intuition; PCA adapts. The first principal component captures the dominant axis of variation; absolute loadings, normalized to sum to 1, become the weights.

### 7.2 Why log-space targets in the pre-campaign model?

Impressions, clicks, conversions, revenue, and spend are heavily right-skewed (a few mega-campaigns dwarf the rest). Training on `log1p` reduces the influence of outliers and keeps MAE meaningful across orders of magnitude.

### 7.3 Why synthetic snapshots for in-flight training?

We don't have time-series snapshots of historical campaigns mid-flight; only the final outcome. `build_inflight_training_features` simulates 25/50/75% snapshots by linearly scaling cumulative metrics — a strong assumption (linearity of pacing) but tractable given the data.

### 7.4 Decision thresholds are heuristic

The 0.7 / 0.4 cutoffs in `make_decision` and the Q75 / Q40 cutoffs in the post-campaign classifier are calibrated against the bundled dataset. They should be reviewed when applied to a different ad ecosystem.

---

## 8. Limitations & Future Work

- **Chatbot code execution** is not sandboxed. A malicious CSV + crafted question could read environment variables. Mitigations: docker isolation, restricted globals (`pd`/`np` only), allow-listed AST nodes.
- **Pre-campaign model** uses a fixed 8-feature schema; richer creative or seasonality features would likely improve accuracy.
- **In-flight forecaster** treats pacing as linear. A richer time-series model (Prophet, lightGBM with lag features) would handle weekly/quarterly seasonality.
- **No A/B testing harness** is included. The recommendations should be paired with an experimentation layer before driving budget shifts.
- **No automated retraining**. The `.pkl` artifacts are point-in-time; a scheduled retraining job (e.g. weekly) would keep the models calibrated.
- **Localization**: LLM outputs are Arabic-first by prompt design. Toggling language per request is a small extension.

---

## 9. Reproducibility

- **Pre-campaign model:** training notebook lives at `recommendation/pre_campaign/models/preprocessing.ipynb`.
- **In-flight ROI / conversion models:** training entry point at `recommendation/in_flight/ml/train.py` (`python -m recommendation.in_flight.ml.train`).
- **Post-campaign:** no training step — fully unsupervised at runtime.

All artifacts are committed `.pkl` files and the dataset is in `data/`. Versions are pinned in `requirements.txt`.

---

## 10. Security Posture

- Secrets are loaded from `.env` (gitignored).
- `.env.example` ships placeholders only.
- The chatbot's `exec` path is the largest attack surface — see §8.
- LLM prompts are length-capped to keep cost bounded and limit prompt-injection blast radius.
