# Persona0

Project for persona-based survey simulation and model comparison.

## What It Does
- Builds persona JSON profiles from `jgss_dummy_data.csv`.
- Prompts multiple LLMs with the same personas/questions.
- Compares simulated vs. real distributions using JSD.
- Stores outputs in separate tables:
	- `results_df`: simulated answers by model/question
	- `metrics_df`: per-model per-question metrics (JSD, valid rate)

## Models
- Gemini
- ChatGPT
- Claude
- Local Llama (OpenAI-compatible endpoint)

## Main File
- `p0book.ipynb` (run top to bottom)

## Quick Setup
Install in the notebook kernel:

```python
%pip install python-dotenv google-genai openai anthropic pandas numpy scikit-learn scipy matplotlib
```

Add a `.env` file in this folder:

```env
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
LLAMA_BASE_URL=http://localhost:11434/v1
LLAMA_API_KEY=ollama
LLAMA_MODEL_ID=llama3.1:8b-instruct-q4_K_M
```

## Run Order
1. Install/import/init cells
2. Data processing + persona creation
3. Model run cells (Gemini/ChatGPT/Claude/Llama)
4. Results visualization cell
