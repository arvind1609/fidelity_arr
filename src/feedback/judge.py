import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.append("../../")

from src.prompts.template import (
    SYSTEM_PROMPT_JUDGE_ACTIONABILITY,
    SYSTEM_PROMPT_JUDGE_CLARITY,
)
from src.models import make_gpt_oss_20b_judge_model, make_nemotron_30b_judge_model
from src.structures.structures_judge import JudgeDimension


PORT = 8000
CONCURRENCY = 6

JUDGE_MODELS = {
    "gpt_oss_20b": make_gpt_oss_20b_judge_model,
    "nemotron_30b": make_nemotron_30b_judge_model,
}

# Each dimension is judged with its own system prompt and its own model call,
# so a model never sees more than one dimension at a time.
DIMENSION_PROMPTS = {
    "actionability": SYSTEM_PROMPT_JUDGE_ACTIONABILITY,
    "clarity": SYSTEM_PROMPT_JUDGE_CLARITY,
}
ALL_DIMENSIONS = tuple(DIMENSION_PROMPTS.keys())


def judge_feedback(judge, df_combine, dimensions=ALL_DIMENSIONS, todo=None):
    results = {}
    if todo is None:
        todo = range(len(df_combine))

    def process(i):
        row = df_combine.iloc[i]
        # True = basic is A / rag is B; False = rag is A / basic is B
        # Fixed per row so "A"/"B" mean the same thing across that row's dimension calls.
        basic_is_a = random.random() < 0.5
        feedback_a = row.feedback_basic if basic_is_a else row.feedback_rag
        feedback_b = row.feedback_rag if basic_is_a else row.feedback_basic
        user_prompt = f"""
        Fidelity Assessment (targeted categories):
        {row.prompt_basic}

        ## Feedback A:
        {feedback_a}

        ## Feedback B:
        {feedback_b}
        """
        scores = {"basic_is_a": basic_is_a}
        for dimension in dimensions:
            messages = [
                ('system', DIMENSION_PROMPTS[dimension]),
                ('user', user_prompt),
            ]
            response = judge.invoke(messages)
            scores[dimension] = response.score
        return scores

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process, i): i for i in todo}
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Error on row {i}: {e}")

    df_results = df_combine.copy()
    for key in (*dimensions, "basic_is_a"):
        df_results[key] = [
            results[i][key] if i in results else df_combine.iloc[i].get(key)
            for i in range(len(df_combine))
        ]
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", choices=JUDGE_MODELS.keys(), default="gpt_oss_20b")
    parser.add_argument("--dimensions", nargs="+", choices=ALL_DIMENSIONS, default=list(ALL_DIMENSIONS),
                         help="Which dimensions to judge. Each is scored with its own model call "
                              "(default: all).")
    parser.add_argument("--basic_csv", type=str, default=None,
                         help="Feedback CSV from generation.py (required unless --resume_from is set)")
    parser.add_argument("--rag_csv", type=str, default=None,
                         help="Feedback CSV from generation_rag.py (required unless --resume_from is set)")
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Path to a previous judge_results csv; only rows where any of "
                              "--dimensions is NaN will be re-judged")
    args = parser.parse_args()
    dimensions = list(dict.fromkeys(args.dimensions))  # de-dupe, keep order

    if args.resume_from:
        df_combine = pd.read_csv(args.resume_from)
        todo = df_combine[df_combine[dimensions].isna().any(axis=1)].index.tolist()
        print(f"Resuming: {len(todo)}/{len(df_combine)} rows need re-judging on {dimensions}")
        output_path = args.resume_from
    else:
        if not (args.basic_csv and args.rag_csv):
            parser.error("--basic_csv and --rag_csv are required unless --resume_from is set")
        df = pd.read_csv(args.basic_csv)
        df_rag = pd.read_csv(args.rag_csv)
        df_combine = pd.merge(df, df_rag, on="filename", how="outer", suffixes=("_basic", "_rag"))
        todo = None
        suffix = "all" if dimensions == list(ALL_DIMENSIONS) else "-".join(dimensions)
        os.makedirs("../../results_judge", exist_ok=True)
        output_path = f"../../results_judge/{args.judge_model}_judge_results_{suffix}.csv"

    judge = JUDGE_MODELS[args.judge_model](port=PORT, schema_class=JudgeDimension)

    df_results = judge_feedback(judge, df_combine, dimensions=dimensions, todo=todo)
    df_results.to_csv(output_path, index=False)