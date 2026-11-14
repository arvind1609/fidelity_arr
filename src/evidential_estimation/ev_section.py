import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("../../")

from tqdm import tqdm

from src.structures import structures_reasoning
from src.prompts import cts_psy, template
from src.models import make_qwen_9b_model, make_gemma_4b_model, make_gemma_26b_model, make_safe_structured_output

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

PORT = 8000

MODEL_MAP = {
    "Qwen3.5-9B": make_qwen_9b_model,
    "gemma-4-E4B-it": make_gemma_4b_model,
    "gemma-4-26B-A4B-it": make_gemma_26b_model,
}
CONCURRENCY = 6


GROUP_CONFIG = {
    "general": (
        structures_reasoning.CtsPysItemsGeneral,
        template.GENERAL_CTS_PSY_USER_PROMPT,
        cts_psy.get_general_statements,
    ),
    "specific": (
        structures_reasoning.CtsPysItemsSpecific,
        template.SPECIFIC_CTS_PSY_USER_PROMPT,
        cts_psy.get_specific_statements,
    ),
}


def run(group, df, ids, model, run_dir):
    structure_class, user_prompt, get_statements = GROUP_CONFIG[group]
    llm = make_safe_structured_output(model, structure_class)
    statements = get_statements()
    transcript_map = df.set_index("filename")["text"].to_dict()

    run_dir = Path(run_dir)
    filename = f"{model.model_name.split('/')[-1]}_{group}_results.csv"
    results_path = run_dir / filename

    existing_df = None
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        already_done = set(existing_df["filename"].values)
        ids = [f for f in ids if f not in already_done]
        print(
            f"Resuming {group}: {len(already_done)} already done, {len(ids)} remaining."
        )

    print(f"Running for {group}...")
    results = []

    def process(f):
        transcript = transcript_map[f]
        full_user_prompt = f"\n\nTranscript: {transcript}\n\n{user_prompt}\n\n{statements}\n\n{template.INSTRUCTION_PROMPT_REASONING}"
        messages = [
            ("system", template.SYSTEM_PROMPT_REASONING),
            ("user", full_user_prompt),
        ]
        response = llm.invoke(messages)
        row = {"filename": f}
        for section_name in structure_class.model_fields:
            section = getattr(response, section_name)
            for field_name in section.__class__.model_fields:
                item = getattr(section, field_name)
                row[f"{field_name}_score"] = item.score.value
                row[f"{field_name}_relevant_spans"] = item.relevant_spans
                row[f"{field_name}_criterion_assessment"] = item.criterion_assessment
        return row

    if len(ids) > 0:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(process, f): f for f in ids}
            for future in tqdm(as_completed(futures), total=len(ids), desc=group):
                f = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"  Error on {f}: {e}")

    new_df = pd.DataFrame(results)
    if existing_df is not None:
        new_df = pd.concat([existing_df, new_df], ignore_index=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(results_path, index=False)
    print(f"Saved {len(new_df)} rows to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(GROUP_CONFIG.keys()),
        default=list(GROUP_CONFIG.keys()),
        help="One or more groups to run (default: both general and specific)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help="vLLM server port (default: 8000)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to an existing results directory to resume from (default: creates a new timestamped directory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["Qwen3.5-9B", "gemma-4-E4B-it", "gemma-4-26B-A4B-it"],
        default="Qwen3.5-9B",
        help="Model to use (default: Qwen3.5-9B)",
    )
    args = parser.parse_args()

    PORT = args.port
    results_base = "../../results"
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(results_base)
        / f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{Path(__file__).stem}"
    )

    df = pd.read_csv("../../data/synthetic_conversations.csv")

    ids = np.unique(df["filename"].values)
    print(len(ids))

    if args.model in ("gemma-4-E4B-it", "gemma-4-26B-A4B-it") and args.run_dir:
        model = MODEL_MAP[args.model](max_tokens=8192, port=args.port)
    else:
        model = MODEL_MAP[args.model](port=args.port)
    for group in args.groups:
        run(group, df, ids, model, run_dir)
