# Persona0: AI Persona Simulation from Survey Data

## Overview
This project builds synthetic respondent personas from survey data, then uses the Gemini API to simulate answers and compare simulated vs. real response distributions.

Main goals:
- Rank non-outcome survey variables by predictive importance.
- Build compact JSON persona profiles per respondent.
- Prompt Gemini with those personas for one or more survey questions.
- Evaluate simulation quality with Jensen-Shannon Distance (JSD).

## Project Structure
- `p0book.ipynb`: Main end-to-end notebook pipeline.
- `.env`: Local API key storage (not committed).
- `jgss_dummy_data.csv`: Input survey data file (expected in this folder).

## Pipeline Summary
The notebook does the following:
1. Loads libraries and initializes Gemini client from environment variables.
2. Loads survey data and defines variable groups.
3. Imputes missing values with most-frequent strategy.
4. Runs iterative Random Forest models to identify globally informative variables.
5. Creates persona JSON objects using demographics + top-ranked features.
6. Runs Gemini prompting for multiple target questions.
7. Computes per-question JSD between real and simulated distributions.

## Requirements
- Python 3.10+
- Jupyter Notebook / VS Code Notebook support
- A Google Gemini API key

Python packages used:
- `python-dotenv`
- `google-genai`
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`

## Installation
From this folder (`Persona0`):

```bash
pip install python-dotenv google-genai pandas numpy scikit-learn scipy
```

If you are using conda, you can also install in your environment first and then run the notebook in that same kernel.

## Configure API Key with .env
Create a `.env` file in this folder with:

```env
GEMINI_API_KEY=your_real_api_key_here
```

Important:
- Do not wrap the key in quotes.
- Make sure there are no extra spaces around `=`.
- Save the file, then restart/re-run notebook initialization cells.

## How to Run
Open `p0book.ipynb` and run cells in order from top to bottom.

Recommended checkpoints:
1. Import and init section runs without errors.
2. API test cell confirms client can generate a response.
3. Data processing + ranking produces top-k variables.
4. Persona creation prints example persona JSON.
5. Multi-question prompting writes simulated columns to dataframe.
6. JSD results print for each question.

## Interpreting JSD
- JSD ranges from `0` to `1`.
- Lower is better (simulated distribution is closer to real distribution).
- If all simulated responses are missing (for example, API quota errors), JSD is skipped.

## Common Issues & Fixes
### 1) `API key not valid`
- Verify `.env` key value is correct.
- Re-run the environment/client initialization cells.
- Ensure notebook kernel is using the expected Python environment.

### 2) `429 RESOURCE_EXHAUSTED`
- You have hit Gemini quota/rate limits.
- Wait and retry later, reduce persona count, or use a paid plan with higher limits.

### 3) JSD returns `nan`
- Usually means no valid simulated responses for that question.
- Check API errors in output and rerun after quota recovers.

## Notes
- The notebook keeps an older single-question block marked as outdated; use the multi-question block for current workflow.
- Keep `.env` private and never commit API keys to source control.
