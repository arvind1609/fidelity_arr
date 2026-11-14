import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.append("../../")
import feedback_prompt_construction as fpc
from src.models import make_qwen_9b_model
from src.prompts.template import SYSTEM_PROMPT_FEEDBACK, USER_PROMPT_FEEDBACK

PORT = 8000
CONCURRENCY = 6


def run_feedback_generation(model, df):
    results = {}

    def process(i):
        feedback_prompt = df.prompt.iloc[i]
        full_user_prompt = f"{feedback_prompt} \n\n {USER_PROMPT_FEEDBACK}"
        messages = [
            ("system", SYSTEM_PROMPT_FEEDBACK),
            ("user", full_user_prompt),
        ]
        response = model.invoke(messages)
        if "</think>" in response.content:
            return response.content.split("</think>")[-1].strip()
        return response.content

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process, i): i for i in range(len(df))}
        for future in tqdm(as_completed(futures), total=len(df)):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Error on row {i}: {e}")

    for i, feedback in results.items():
        df.loc[i, "feedback"] = feedback
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Merged evidential-estimation results CSV (from merge_results.py)")
    args = parser.parse_args()

    model = make_qwen_9b_model(port=PORT)
    input_path = args.results_csv
    df = pd.read_csv(input_path)
    df = fpc.generate_feedback_prompt(df)
    df = run_feedback_generation(model, df)
    run_name = os.path.basename(os.path.dirname(os.path.abspath(input_path)))
    output_path = os.path.join("../../results_feedback", run_name, os.path.basename(input_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
