# Persona0

Simulates Japanese General Social Survey (JGSS) responses using LLMs and measures how closely they match the real survey distribution using Jensen-Shannon Divergence (JSD).

Adapted from Rupprecht et al. (2025), *"German General Social Survey Personas"* ([arXiv:2511.21722](https://arxiv.org/abs/2511.21722)) for the JGSS-2017/2018 dataset.

---

## How It Works

1. **Variable selection** — A leave-one-out Random Forest ranks all non-demographic, non-outcome JGSS variables by their collective importance. This produces a single ranked list, sliced at `TOP_K_VALUES = [2, 4, 8, 16, 32, 64, 128]` to create persona sets of varying richness.

2. **Persona construction** — Each of the 2,660 respondents becomes a JSON persona: 10 fixed core demographic variables + the top-*k* RF variables selected by `ACTIVE_PERSONA_K`.

3. **Simulation** — Each persona is prompted with each outcome question in Japanese. The LLM returns a single integer on the question's scale.

4. **Evaluation** — Simulated answer distributions are compared against the real JGSS distribution using JSD (lower = better match). Results are stored per-model in `results_by_model` and summarized in `metrics_df`.

---

## Dataset

**JGSS-2017/2018 Integrated Data (v1.0)**
- N = 2,660 respondents (744 from 2017 + 1,916 from 2018)
- 559 variables total

Variable roles (defined in `config.py`):

| Role | Count | Description |
|---|---|---|
| Excluded | 30 | Metadata, identifiers, empty variables, redundant re-codings |
| Core demographics | 10 | Fixed in every persona (sex, age, education, income, occupation, etc.) |
| Outcomes | 19 | Held-out prediction targets across 6 thematic categories |
| RF pool | remainder | Compete for top-*k* persona slots via the importance ranking |

**Outcome categories:** Trust · Ethnocentrism · Gender & family norms · Social inequality · Wellbeing · Community & civic

Missing value codes `[9, 99, 999]` are replaced with `NaN` on load.

---

## Models

| Key | Model | Provider |
|---|---|---|
| `gemini` | gemini-2.5-flash | Google |
| `chatgpt` | gpt-4o-mini | OpenAI |
| `claude` | claude-haiku-4-5 | Anthropic |
| `llama` | llama-3.1-swallow-8b-instruct-v0.2-i1 | Local (Ollama) |

---

## Files

```
Persona0/
├── persona20172018.ipynb   # Main notebook (active)
├── config.py               # Variable classification + helpers (EXCLUDES, CORE_DEMOGRAPHICS, OUTCOMES)
├── data/
│   ├── 20172018_data.sav       # JGSS source data (SPSS format)
│   ├── rf_ranking_filtered.csv # Saved RF importance ranking
│   ├── personas_by_k.json      # Saved persona sets for all k values
│   ├── results_<model>.csv     # Saved simulated answers per LLM
│   └── metrics_df.csv          # Saved JSD + valid-rate metrics
└── p0book.ipynb            # Original notebook (archived)
```

---

## Setup

**1. Install dependencies** (run once in the notebook kernel):

```python
%pip install python-dotenv google-genai openai anthropic pandas numpy scikit-learn scipy matplotlib pyreadstat
```

**2. Create a `.env` file** in this folder:

```env
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Llama local endpoint (Ollama defaults shown)
LLAMA_BASE_URL=http://localhost:1234/v1
LLAMA_API_KEY=ollama
LLAMA_MODEL_ID=llama-3.1-swallow-8b-instruct-v0.2-i1
```

---

## Notebook Structure (`persona20172018.ipynb`)

| Part | Section | Description |
|---|---|---|
| 1 | Setup | `%pip install` + all imports |
| 2 | Configuration | All tunable parameters (see below) |
| 3 | Connectivity Check | Smoke test for each API client |
| 4 | Data | Load SAV, replace missing codes, run `config.sanity_check` |
| 5 | RF Feature Selection | Leave-one-out RF ranking, build `persona_slices` |
| 6 | Persona Creation | Build `personas_by_k` for all top-*k* values |
| 7 | Checkpoint — Save / Load | Persist or restore RF ranking, personas, results |
| 8 | Simulation | Run each LLM (one cell per model) |
| 9 | Results Visualization | JSD bar charts, valid-rate, distribution comparison |
| — | Archive | Reference code; not part of active pipeline |

**Run order for a fresh session:** Parts 1 → 2 → 4 → 5 → 6 → 7 (Load) → 8 → 7 (Save) → 9

**Run order after a checkpoint:** Parts 1 → 2 → 7 (Load) → 9

---

## Key Configuration (Part 2)

All parameters are centralized in the Configuration cell. Re-running it applies changes without a kernel restart.

| Parameter | Default | Description |
|---|---|---|
| `K_ACCUM` | `10` | Top-*k* features accumulated per RF model (per Rupprecht et al.) |
| `MIN_VALID_RATE` | `0.75` | Variables below this valid-response rate are dropped from the RF pool |
| `TOP_K_VALUES` | `[2,4,8,16,32,64,128]` | Persona sizes to build and compare |
| `ACTIVE_PERSONA_K` | `2` | Which top-*k* slice feeds the current simulation run |
| `MAX_PERSONAS` | `50` | Safety cap on API calls per run — set to `None` for the full 2,660 |
| `ACTIVE_QUESTION_KEYS` | `['op4trust', 'stalllf', 'q5gveqaa']` | Outcome questions to simulate in this run |

---

## Output

- **`results_by_model`** — dict of DataFrames, one per LLM. Columns = active question variable names, rows = persona indices.
- **`metrics_df`** — one row per (model × question). Columns: `model`, `question_var`, `persona_k`, `valid_answers`, `total_personas`, `jsd`.
- **Token usage** — tracked per model during simulation; cost estimate printed at end of each run.
