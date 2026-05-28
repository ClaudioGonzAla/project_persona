#!/usr/bin/env python3
"""
run_simulation.py — Execute one LLM persona simulation and save results.

Replicates the simulation logic from persona20172018.ipynb (Parts 1-6) in
a standalone CLI script suitable for remote-server execution.

Prerequisites (run Appendices B + C in the notebook once to generate these):
    data/20172018_data_jap.sav         — Japanese-labeled JGSS source
    data/isco08_ja.json                — ISCO-08 code → Japanese label map
                                         (used by Appendix C; personas_by_k.json
                                          must already be built with Japanese labels)
    data/personas_by_k.json            — full set of personas for each k
    data/rf_ranking_filtered.csv       — RF importance ranking (informational only)
    data/question_bank.json            — question text + scale info (informational only)

Output (auto-saved on completion):
    results/results_{model}_k{k}.csv   — per-respondent answers
    results/metrics_all.csv            — appends 19 rows (model × k × question)
    results/run_log.csv                — appends 1 row per batch

Usage:
    python run_simulation.py --model llama_tuned_8b --k 8
    python run_simulation.py --model chatgpt --k 16 --n 100
    python run_simulation.py --model llama_base_8b --k 2 8 16
    python run_simulation.py --model llama_base_70b --k 2 8 16 --seed 42

Multi-model loop (shell):
    for model in llama_tuned_8b llama_base_8b chatgpt; do
        python run_simulation.py --model $model --k 2 8 16
    done

Environment (.env):
    OPENAI_API_KEY        — for chatgpt
    ANTHROPIC_API_KEY     — for claude
    GEMINI_API_KEY        — for gemini

    # Local Llama 8B (LM Studio etc.)
    LLAMA_BASE_URL        — default 'http://localhost:1234/v1'
    LLAMA_API_KEY         — default 'lm-studio'
    LLAMA_TUNED_8B_ID     — model id of the Swallow-8B build loaded on the 8B server
    LLAMA_BASE_8B_ID      — model id of the base-8B build loaded on the 8B server

    # Llama 70B (separate machine — falls back to LLAMA_BASE_URL/LLAMA_API_KEY
    # above if these are not set, so you can also run 70B on the same endpoint)
    LLAMA_70B_BASE_URL    — endpoint of the beefier lab server hosting 70B
    LLAMA_70B_API_KEY     — its API key (if any)
    LLAMA_TUNED_70B_ID    — model id of the Swallow-70B build
    LLAMA_BASE_70B_ID     — model id of the Llama-3.3-70B-Instruct build
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from dotenv import load_dotenv
from scipy.spatial.distance import jensenshannon


# ─── Constants (must stay in sync with notebook) ──────────────────────────────

SYS_INSTRUCT = (
    "あなたは日本の社会調査（JGSS）の回答者です。"
    "以下のペルソナの視点に立って、質問に対する回答を選択肢から選んでください。"
    "回答は選択肢の番号（数字）のみを出力してください。説明や理由は不要です。"
)

QUESTION_TEXT = {
    # Trust
    'op4trust': "一般的に言って、人は信頼できると思いますか？",
    'tr3cgmnz': "国会議員をどの程度信頼していますか？",
    'tr3bcraz': "省庁・政府機関をどの程度信頼していますか？",
    # Ethnocentrism
    'qfnrincr': "日本の外国人を増やすべきだと思いますか？",
    'q4samesm': "同性婚についてどうお考えですか？",
    'op7gdevo': "人間の本質について、1（非常に自己中心的）から7（非常に善良）のどこに当てはまりますか？",
    # Gender & family norms
    'q7wwhhx':  "「男性は外で働き、女性は家庭を守るべきだ」という考えに同意しますか？",
    'q7jbmmcc': "「母親が働くと子どもに悪影響がある」という考えに同意しますか？",
    'q7mgcc':   "「結婚したカップルは子どもを持つべきだ」という考えに同意しますか？",
    # Social inequality
    'q5gveqaa': "政府は所得格差を縮小する責任があると思いますか？",
    'opincdif': "所得格差は大きくなりすぎていると思いますか？",
    'opnucpol': "原子力政策についてどうお考えですか？",
    # Wellbeing
    'stalllf':  "現在の生活全体にどの程度満足していますか？",
    'nofutr':   "将来に希望が持てないと感じますか？",
    'sfmhdprs': "過去4週間で、気分が落ち込んだり憂うつになったりすることがありましたか？",
    'op5happz': "あなたは幸福だと思いますか？",
    # Community & civic
    'opnbmtcn': "近所の人々は互いに気にかけ合っていると思いますか？",
    'wllive':   "現在住んでいる地域に住み続けたいと思いますか？",
    'mempltgp': "政治団体や市民運動に参加していますか？",
}

# JGSS missing-code conventions
MISSING_CODES = {9, 99, 999}

# Per-model defaults; CLI flags can override temperature / max_tokens
MODEL_DEFAULTS = {
    'gemini':         dict(model_id='gemini-2.5-flash',  sleep=1.0, temp=0.7, max_tokens=100, input_cost=0.075, output_cost=0.30),
    'chatgpt':        dict(model_id='gpt-4o-mini',       sleep=1.0, temp=0.7, max_tokens=100, input_cost=0.150, output_cost=0.60),
    'claude':         dict(model_id='claude-haiku-4-5',  sleep=1.0, temp=0.7, max_tokens=100, input_cost=0.800, output_cost=4.00),
    'llama_tuned_8b': dict(model_id=None,                sleep=0.0, temp=0.7, max_tokens=100, input_cost=0.000, output_cost=0.00),
    'llama_base_8b':  dict(model_id=None,                sleep=0.0, temp=0.7, max_tokens=100, input_cost=0.000, output_cost=0.00),
    'llama_tuned_70b': dict(model_id=None,                sleep=0.0, temp=0.7, max_tokens=100, input_cost=0.000, output_cost=0.00),
    'llama_base_70b':  dict(model_id=None,                sleep=0.0, temp=0.7, max_tokens=100, input_cost=0.000, output_cost=0.00),
}


# ─── Question-bank construction (mirrors notebook Part 5) ─────────────────────

def _strip_duplicate_code_prefix(code: int, label: str) -> str:
    """JGSS labels like '1：…' (where 1 matches the code) — strip the prefix."""
    return re.sub(rf'^{int(code)}[:：]\s*', '', str(label).strip())


def build_question_bank(value_labels: dict, question_keys: list[str]) -> dict:
    """Build question_bank from .sav value_labels + paraphrased question text.
    Guarantees the options shown to the LLM are byte-identical to the JGSS scale."""
    out = {}
    for qvar in question_keys:
        if qvar not in QUESTION_TEXT:
            raise ValueError(f"Unknown question variable: {qvar}")
        raw = value_labels.get(qvar, {})
        valid = sorted([(int(k), _strip_duplicate_code_prefix(k, v))
                        for k, v in raw.items() if int(k) not in MISSING_CODES])
        if not valid:
            raise ValueError(f"No valid options found for {qvar}")
        opts = [f"{c}: {lab}" if lab else str(c) for c, lab in valid]
        out[qvar] = {
            'question':  QUESTION_TEXT[qvar],
            'options':   opts,
            'scale_min': valid[0][0],
            'scale_max': valid[-1][0],
        }
    return out


# ─── Prompt + parse ───────────────────────────────────────────────────────────

def build_user_prompt(persona_json: str, spec: dict) -> str:
    return (
        f"以下の特徴を持つ人物の視点に立ってください：{persona_json}\n\n"
        f"次の質問に対するこの人物の回答はどの選択肢ですか：{spec['question']}\n\n"
        f"回答選択肢：\n" + "\n".join(spec['options'])
    )


def extract_answer(raw_text: str | None, scale_min: int, scale_max: int) -> int | None:
    """Return the LAST in-range digit (preambles may cite ages/codes first)."""
    if not raw_text:
        return None
    in_range = [int(m) for m in re.findall(r"\d+", raw_text)
                if scale_min <= int(m) <= scale_max]
    return in_range[-1] if in_range else None


# ─── Sampling (reproducible across models/k via SAMPLE_SEED) ──────────────────

def sample_indices(n_requested: int, n_total: int, seed: int) -> list[int]:
    if n_requested >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, n_requested, replace=False).tolist())


# ─── Model clients (lazy SDK imports — only the chosen model's SDK is needed) ─

def make_generate_fn(model: str, cfg: dict):
    """Return (generate_fn, resolved_model_id). generate_fn signature:
        (user_prompt: str) -> (raw_text: str, input_tokens: int, output_tokens: int)"""

    if model in ('llama_tuned_8b', 'llama_base_8b'):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("ERROR: pip install openai")
        env_key = 'LLAMA_TUNED_8B_ID' if model == 'llama_tuned_8b' else 'LLAMA_BASE_8B_ID'
        model_id = os.environ.get(env_key)
        if not model_id:
            sys.exit(f"ERROR: env var {env_key} not set")
        base_url = os.environ.get('LLAMA_BASE_URL', 'http://localhost:1234/v1')
        api_key = os.environ.get('LLAMA_API_KEY', 'lm-studio')
        client = OpenAI(base_url=base_url, api_key=api_key)

        def gen(prompt: str):
            r = client.chat.completions.create(
                model=model_id,
                messages=[{'role': 'system', 'content': SYS_INSTRUCT},
                          {'role': 'user',   'content': prompt}],
                temperature=cfg['temp'],
                max_tokens=cfg['max_tokens'],
            )
            return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
        return gen, model_id

    if model in ('llama_tuned_70b', 'llama_base_70b'):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("ERROR: pip install openai")
        env_key = 'LLAMA_TUNED_70B_ID' if model == 'llama_tuned_70b' else 'LLAMA_BASE_70B_ID'
        model_id = os.environ.get(env_key)
        if not model_id:
            sys.exit(f"ERROR: env var {env_key} not set")
        # Prefer 70B-specific endpoint; fall back to the shared LLAMA_BASE_URL
        # so a single-endpoint setup still works without extra .env entries.
        base_url = os.environ.get('LLAMA_70B_BASE_URL') or os.environ.get('LLAMA_BASE_URL', 'http://localhost:1234/v1')
        api_key  = os.environ.get('LLAMA_70B_API_KEY')  or os.environ.get('LLAMA_API_KEY', 'lm-studio')
        client = OpenAI(base_url=base_url, api_key=api_key)

        def gen(prompt: str):
            r = client.chat.completions.create(
                model=model_id,
                messages=[{'role': 'system', 'content': SYS_INSTRUCT},
                          {'role': 'user',   'content': prompt}],
                temperature=cfg['temp'],
                max_tokens=cfg['max_tokens'],
            )
            return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
        return gen, model_id

    if model == 'chatgpt':
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("ERROR: pip install openai")
        if not os.environ.get('OPENAI_API_KEY'):
            sys.exit("ERROR: OPENAI_API_KEY not set")
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        model_id = cfg['model_id']

        def gen(prompt: str):
            r = client.chat.completions.create(
                model=model_id,
                messages=[{'role': 'system', 'content': SYS_INSTRUCT},
                          {'role': 'user',   'content': prompt}],
                temperature=cfg['temp'],
                max_tokens=cfg['max_tokens'],
            )
            return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
        return gen, model_id

    if model == 'claude':
        try:
            from anthropic import Anthropic
        except ImportError:
            sys.exit("ERROR: pip install anthropic")
        if not os.environ.get('ANTHROPIC_API_KEY'):
            sys.exit("ERROR: ANTHROPIC_API_KEY not set")
        client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        model_id = cfg['model_id']

        def gen(prompt: str):
            r = client.messages.create(
                model=model_id,
                system=SYS_INSTRUCT,
                max_tokens=cfg['max_tokens'],
                temperature=cfg['temp'],
                messages=[{'role': 'user', 'content': prompt}],
            )
            text = "".join(b.text for b in r.content if hasattr(b, 'text'))
            return text, r.usage.input_tokens, r.usage.output_tokens
        return gen, model_id

    if model == 'gemini':
        try:
            import google.genai as genai
            from google.genai import types as genai_types
        except ImportError:
            sys.exit("ERROR: pip install google-genai")
        if not os.environ.get('GEMINI_API_KEY'):
            sys.exit("ERROR: GEMINI_API_KEY not set")
        client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        model_id = cfg['model_id']

        def gen(prompt: str):
            r = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYS_INSTRUCT,
                    temperature=cfg['temp'],
                    max_output_tokens=cfg['max_tokens'],
                ),
            )
            text = r.text or ""
            u = r.usage_metadata
            return text, (u.prompt_token_count or 0), (u.candidates_token_count or 0)
        return gen, model_id

    raise ValueError(f"Unknown model: {model}")


# ─── Save (atomic-ish: writes 3 files in order) ───────────────────────────────

def save_outputs(results_dir: Path, model: str, k: int, *,
                 model_results: pd.DataFrame, metrics_df: pd.DataFrame,
                 n_personas: int, seed: int, question_keys: list[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-(model, k) results CSV
    results_path = results_dir / f'results_{model}_k{k}.csv'
    model_results.to_csv(results_path)
    print(f"  ✓ {results_path}  {model_results.shape}")

    # 2. metrics_all.csv — dedup any prior (model, k) rows before appending
    metrics_path = results_dir / 'metrics_all.csv'
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        mask = ~((existing['model'] == model) & (existing['persona_k'] == k))
        combined = pd.concat([existing[mask], metrics_df], ignore_index=True)
    else:
        combined = metrics_df.copy()
    combined.to_csv(metrics_path, index=False)
    print(f"  ✓ {metrics_path}  ({len(metrics_df)} new rows, {len(combined)} total)")

    # 3. run_log.csv — append-only
    log_path = results_dir / 'run_log.csv'
    log_row = {
        'timestamp':   datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model':       model,
        'persona_k':   k,
        'questions':   ','.join(question_keys),
        'n_personas':  n_personas,
        'sample_seed': seed,
        'mean_jsd':    round(metrics_df['jsd'].mean(), 4),
        'valid_rate':  round((metrics_df['valid_answers'] / metrics_df['total_personas']).mean(), 3),
    }
    new_log = pd.DataFrame([log_row])
    if log_path.exists():
        pd.concat([pd.read_csv(log_path), new_log], ignore_index=True).to_csv(log_path, index=False)
    else:
        new_log.to_csv(log_path, index=False)
    print(f"  ✓ {log_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Run one LLM × one or more k simulations and save results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--model',       required=True, choices=list(MODEL_DEFAULTS.keys()))
    p.add_argument('--k',           required=True, type=int, nargs='+',
                   help='persona richness value(s) — e.g. --k 2 8 16; each must exist '
                        'in personas_by_k.json; runs sequentially, saves after each k')
    p.add_argument('--n',           type=int, default=266, help='persona sample size (default: 266)')
    p.add_argument('--seed',        type=int, default=42,  help='random sample seed (default: 42)')
    p.add_argument('--temperature', type=float, default=None, help='override temperature')
    p.add_argument('--max-tokens',  type=int,   default=None, help='override max_tokens')
    p.add_argument('--data-dir',    default='data',     help='source .sav lives here')
    p.add_argument('--results-dir', default='results',  help='personas + outputs live here')
    p.add_argument('--sav-file',    default='20172018_data_jap.sav')
    p.add_argument('--questions',   default=None,
                   help='comma-separated question subset (default: all 19)')
    args = p.parse_args()

    load_dotenv()

    # ── One-time setup (shared across all k values) ───────────────────────────

    # 1. Verify inputs exist
    data_dir    = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    sav_path      = data_dir / args.sav_file
    personas_path = data_dir / 'personas_by_k.json'
    if not sav_path.exists():
        sys.exit(f"ERROR: {sav_path} not found")
    if not personas_path.exists():
        sys.exit(f"ERROR: {personas_path} not found — run Appendix C in the notebook first")

    # 2. Resolve config (defaults + CLI overrides)
    cfg = dict(MODEL_DEFAULTS[args.model])
    if args.temperature is not None: cfg['temp']       = args.temperature
    if args.max_tokens  is not None: cfg['max_tokens'] = args.max_tokens

    # 3. Load source data
    print(f"Loading {sav_path} ...")
    df, meta = pyreadstat.read_sav(str(sav_path))
    for code in MISSING_CODES:
        df.replace(code, np.nan, inplace=True)

    # 4. Build question bank (from .sav value_labels — byte-correct options)
    question_keys = (args.questions.split(',') if args.questions
                     else list(QUESTION_TEXT.keys()))
    question_bank = build_question_bank(meta.variable_value_labels, question_keys)
    print(f"  {len(question_bank)} questions active")

    # 5. Load personas — validate all requested k values exist before starting
    with open(personas_path, encoding='utf-8') as f:
        personas_by_k = {int(k): v for k, v in json.load(f).items()}
    missing_k = [k for k in args.k if k not in personas_by_k]
    if missing_k:
        sys.exit(f"ERROR: k={missing_k} not in personas_by_k (have: {sorted(personas_by_k)})")

    # 6. Precompute real distributions from full N=2,660 (same for every k)
    real_dist = {
        v: df[v].dropna().astype(int).value_counts(normalize=True)
            .reindex(range(spec['scale_min'], spec['scale_max'] + 1), fill_value=0)
        for v, spec in question_bank.items()
    }

    # 7. Initialize LLM client once — reused across all k values
    print(f"\nInitializing {args.model} ...")
    generate_fn, model_id = make_generate_fn(args.model, cfg)
    print(f"  model_id    : {model_id}")
    print(f"  temperature : {cfg['temp']}")
    print(f"  max_tokens  : {cfg['max_tokens']}")
    print(f"  sleep       : {cfg['sleep']}s between calls")
    print(f"  k values    : {args.k}")
    print(f"  API calls   : {len(args.k) * args.n * len(question_bank):,} total "
          f"({len(args.k)} k × {args.n} personas × {len(question_bank)} questions)")

    # ── Per-k loop ────────────────────────────────────────────────────────────
    for k in args.k:
        print(f"\n{'━'*72}")
        print(f"  k = {k}  ({args.n} personas × {len(question_bank)} questions"
              f" = {args.n * len(question_bank):,} calls)")
        print(f"{'━'*72}")

        # Sample personas for this k (same seed → same respondents across all k)
        personas_full = personas_by_k[k]
        sample_idx = sample_indices(args.n, len(personas_full), args.seed)
        active_personas = [personas_full[i] for i in sample_idx]
        print(f"  sampled {len(active_personas)} personas  "
              f"(seed={args.seed}, indices {sample_idx[:3]}…{sample_idx[-3:]})")

        model_results = pd.DataFrame(index=[df.index[i] for i in sample_idx])
        metrics_rows  = []
        tot_in = tot_out = tot_calls = 0
        t_start = time.time()

        for var_name, spec in question_bank.items():
            all_values = list(range(spec['scale_min'], spec['scale_max'] + 1))
            answers = []
            print(f"\n[{args.model}  k={k}] {var_name}  "
                  f"(scale {spec['scale_min']}–{spec['scale_max']})")

            for i, persona in enumerate(active_personas):
                try:
                    raw, in_t, out_t = generate_fn(build_user_prompt(persona, spec))
                    answers.append(extract_answer(raw, spec['scale_min'], spec['scale_max']))
                    tot_in += in_t; tot_out += out_t; tot_calls += 1
                except Exception as e:
                    print(f"  respondent {i+1} error: {type(e).__name__}: {e}")
                    answers.append(None)
                if cfg['sleep'] > 0:
                    time.sleep(cfg['sleep'])
                if (i + 1) % 50 == 0 or (i + 1) == len(active_personas):
                    done = sum(1 for a in answers if a is not None)
                    print(f"  {i+1}/{len(active_personas)}  ({done} valid so far)")

            model_results[var_name] = answers

            # JSD
            sim_series = pd.Series(answers).dropna().astype(int)
            sim_dist = sim_series.value_counts(normalize=True).reindex(all_values, fill_value=0)
            valid_n = int(sim_series.notna().sum())
            if sim_dist.sum() == 0:
                jsd_score = None
                print(f"  {var_name}: no valid responses — JSD skipped")
            else:
                jsd_score = float(jensenshannon(real_dist[var_name], sim_dist))
                print(f"  {var_name}: JSD={jsd_score:.4f}  valid={valid_n}/{len(answers)}")

            metrics_rows.append({
                'model':          args.model,
                'question_var':   var_name,
                'persona_k':      k,
                'valid_answers':  valid_n,
                'total_personas': len(answers),
                'jsd':            jsd_score,
            })

        elapsed = time.time() - t_start
        metrics_df = pd.DataFrame(metrics_rows)

        # Per-k summary
        print(f"\n{'='*72}")
        print(f"  {args.model}  ·  k={k}  ·  n={args.n}  ·  {elapsed/60:.1f} min")
        print(f"{'='*72}")
        print(f"  Mean JSD      : {metrics_df['jsd'].mean():.4f}")
        print(f"  Valid rate    : {(metrics_df['valid_answers']/metrics_df['total_personas']).mean():.1%}")
        print(f"  API calls     : {tot_calls:,}")
        if tot_calls:
            print(f"  Input tokens  : {tot_in:,}  (avg {tot_in/tot_calls:.0f}/call)")
            print(f"  Output tokens : {tot_out:,}")
        cost = tot_in/1e6 * cfg['input_cost'] + tot_out/1e6 * cfg['output_cost']
        if cost > 0:
            print(f"  Est. cost     : ${cost:.4f}  "
                  f"(@ ${cfg['input_cost']}/1M in, ${cfg['output_cost']}/1M out)")
        print()

        # Persist after each k — safe to Ctrl-C between k runs
        save_outputs(results_dir, args.model, k,
                     model_results=model_results, metrics_df=metrics_df,
                     n_personas=args.n, seed=args.seed, question_keys=question_keys)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted before save. No partial results were written.")
        sys.exit(130)
