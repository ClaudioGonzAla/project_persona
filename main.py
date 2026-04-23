# INSTALL
# %pip install python-dotenv google-genai

# Core utilities
from dotenv import load_dotenv
import os
import time
import re
import json

# Data + modeling
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import jensenshannon
from collections import defaultdict

# Gemini API SDK
import google.genai as genai
from google.genai import types

print("Imports ready:", pd.__version__)

# Load .env from project root and build a reusable API client
load_dotenv()
client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

print("GenAI client initialized and API key loaded.")

# 1) Load survey data
df = pd.read_csv('jgss_dummy_data.csv')

# 2) Define variable groups used in feature selection and evaluation
core_demographics = [
    'Sex', 'Age', 'Region', 'Area', 'Education', 
    'Income', 'Occupation', 'Marital_Status'
]

# Held-out outcomes for persona evaluation (not used in feature ranking)
outcome_variables = [
    'ep01_gen_economic_sit', 'ep03_own_economic_sit', 'ep06_future_economic_sit',
    'mp12_foreigners_help_shortage', 'mn01_born_in_country_importance', 
    'mp18_refugees_opportunities_risks', 'li03_leisure_importance', 
    'lp03_ordinary_people_worse', 'lp04_unjustifiable_children', 'ca12_smoking_hashish', 
    'vm16_own_cells_ivf', 'vm17_donated_cells_ivf', 'pt15_trust_political_parties', 
    'pe05_politicians_represent_interests', 'ps01_satisfaction_government', 
    'rb01_personal_god', 'mm01_restrict_islam', 'mm06_religious_fanatics', 
    'st01_trust_people', 'sm01_union_member_current', 'sm02_union_member_past', 
    'im03_importance_education', 'im08_importance_diligence', 'iw04_state_welfare', 
    'vi06_help_disadvantaged', 'vi07_assert_needs', 'vi10_politically_active'
]

# Candidate predictors: everything except IDs, demographics, and held-out outcomes
features_to_rank = [
    col for col in df.columns
    if col not in core_demographics and col not in outcome_variables and col != 'Respondent_ID'
]

# 3) Impute missing values so each RF model can train without split-ballot null errors
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 4) For each candidate variable, train an RF classifier and accumulate top predictors
k = 5  # Top predictors kept per target model
master_importance_scores = defaultdict(float)

for target_var in features_to_rank:
    # Predict one variable from the rest (leave-one-target-out)
    X = df_imputed[features_to_rank].drop(columns=[target_var])
    y = df_imputed[target_var].astype(str)  # Force classification mode

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_k = importances.nlargest(k)

    # Aggregate importance across all target models to get globally useful variables
    for feature, score in top_k.items():
        master_importance_scores[feature] += score

# 5. Sort the master dictionary to find your global TOP-k variables
sorted_ranking = sorted(master_importance_scores.items(), key=lambda x: x[1], reverse=True)

# 6. Display the Results
print("\n--- MASTER VARIABLE IMPORTANCE RANKING ---")
for i, (var, score) in enumerate(sorted_ranking[:k]): # Displaying TOP-k
    print(f"Rank {i+1}: {var} (Aggregated Score: {score:.4f})")

# Build compact JSON persona profiles from demographics + top-k ranked attributes
personas = []

print(sorted_ranking[:k])
top_k_variables = [var for var, _ in sorted_ranking[:k]]
print(top_k_variables)

for index, row in df.iterrows():
    persona_dict = {}

    # Keep only non-missing demographic fields
    for demo in core_demographics:
        if pd.notna(row[demo]):
            persona_dict[demo] = row[demo]

    # Add non-missing high-signal attributes discovered by RF ranking
    for top_var in top_k_variables:
        if pd.notna(row[top_var]):
            persona_dict[top_var] = row[top_var]

    # Store as JSON string so it can be passed directly to prompt text
    persona_json = json.dumps(persona_dict, ensure_ascii=False)
    personas.append(persona_json)

print(f"Successfully generated {len(personas)} JSON personas.")
print("\n--- EXAMPLE PERSONA (Ready for API) ---")
print(personas[0])

# Question config: prompt text + valid response scale for each target variable
question_bank = {
    'ep01_gen_economic_sit': {
        'question': "現在の一般的な経済状況についてどう思いますか？ 1 (非常に良い) から 5 (非常に悪い) でお答えください。数字のみを出力してください。",
        'scale_min': 1,
        'scale_max': 5,
    },
    'pt15_trust_political_parties': {
        'question': "政党をどの程度信頼していますか？ 1 (全く信頼していない) から 5 (非常に信頼している) でお答えください。数字のみを出力してください。",
        'scale_min': 1,
        'scale_max': 5,
    },
    'st01_trust_people': {
        'question': "一般的に、人は信頼できると思いますか？ 1 (信頼できる) から 4 (信頼できない) でお答えください。数字のみを出力してください。",
        'scale_min': 1,
        'scale_max': 4,
    }
}

print(f"Starting generalized simulation for {len(question_bank)} questions and {len(personas)} personas...")

# Collect generated columns first, then assign once (avoids DataFrame fragmentation warnings)
simulated_column_data = {}

# AI client parameters
model_id = "gemini-2.5-flash"
sys_instruct = "あなたは日本のアンケート回答者です。提供されたペルソナプロファイルに基づいて、質問に答えてください。出力は1桁の数字のみにしてください。"


for var_name, spec in question_bank.items():
    question_text = spec['question']
    all_values = range(spec['scale_min'], spec['scale_max'] + 1)
    simulated_answers = []

    print(f"\n--- Testing Variable: {var_name} ---")

    for i, persona_json in enumerate(personas):
        user_prompt = f"Persona Profile:\n{persona_json}\n\nQuestion:\n{question_text}"
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruct,
                    temperature=0.0,
                ),
            )

            raw_text = response.text or ""
            # Extract the first integer produced by the model
            match = re.search(r"\d+", raw_text)
            answer = int(match.group()) if match else None
            simulated_answers.append(answer)

            print(f"Respondent {i+1} replied: {answer}")
            time.sleep(1)

        except Exception as e:
            # Keep row alignment even on quota/rate-limit/API errors
            print(f"API Error on respondent {i+1}: {e}")
            simulated_answers.append(None)

    simulated_column_name = f'simulated_{var_name}'
    simulated_column_data[simulated_column_name] = simulated_answers

    real_dist = df[var_name].value_counts(normalize=True).sort_index().reindex(all_values, fill_value=0)
    sim_series = pd.Series(simulated_answers)
    sim_dist = sim_series.value_counts(normalize=True).sort_index().reindex(all_values, fill_value=0)

    print(f"Real distribution for {var_name}:", real_dist.values)
    print(f"Simulated distribution for {var_name}:", sim_dist.values)

    if sim_dist.sum() == 0:
        print(f"JSD Score for {var_name}: skipped (no valid simulated responses)")
    else:
        jsd_score = jensenshannon(real_dist, sim_dist)
        print(f"JSD Score for {var_name}: {jsd_score:.4f}")

df = df.assign(**simulated_column_data)
