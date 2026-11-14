import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("../../")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tqdm import tqdm

from src.models import make_qwen_9b_model, make_qwen_9b_reflection_model, make_gemma_4b_model, make_gemma_4b_reflection_model, make_gemma_26b_model, make_gemma_26b_reflection_model, make_safe_structured_output
from src.prompts import cts_psy, template
from src.structures import structures_reasoning

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


MODEL_MAP = {
    "Qwen3.5-9B": make_qwen_9b_model,
    "gemma-4-E4B-it": make_gemma_4b_model,
    "gemma-4-26B-A4B-it": make_gemma_26b_model,
}

REFLECTION_MAP = {
    "Qwen3.5-9B": make_qwen_9b_reflection_model,
    "gemma-4-E4B-it": make_gemma_4b_reflection_model,
    "gemma-4-26B-A4B-it": make_gemma_26b_reflection_model,
}
PORT = 8000
CONCURRENCY = 6
STEPS = 2

CATEGORY_CONFIG = {
    "agenda": (structures_reasoning.AgendaItems, template.AGENDA_USER_PROMPT, cts_psy.get_agenda_statements),
    "feedback": (structures_reasoning.FeedbackItems, template.FEEDBACK_USER_PROMPT, cts_psy.get_feedback_statements),
    "understanding": (
        structures_reasoning.UnderstandingItems,
        template.UNDERSTANDING_USER_PROMPT,
        cts_psy.get_understanding_statements,
    ),
    "interpersonal_effectiveness": (
        structures_reasoning.InterpersonalEffectivenessItems,
        template.INTERPERSONAL_EFFECTIVENESS_USER_PROMPT,
        cts_psy.get_interpersonal_effectiveness_statements,
    ),
    "collaboration": (
        structures_reasoning.CollaborationItems,
        template.COLLABORATION_USER_PROMPT,
        cts_psy.get_collaboration_statements,
    ),
    "discovery": (structures_reasoning.DiscoveryItems, template.DISCOVERY_USER_PROMPT, cts_psy.get_discovery_statements),
    "cognition": (structures_reasoning.CognitionItems, template.COGNITION_USER_PROMPT, cts_psy.get_cognition_statements),
    "intervention": (
        structures_reasoning.InterventionItems,
        template.INTERVENTION_USER_PROMPT,
        cts_psy.get_intervention_statements,
    ),
    "homework": (structures_reasoning.HomeworkItems, template.HOMEWORK_USER_PROMPT, cts_psy.get_homework_statements),
    "social": (structures_reasoning.SocialItems, template.SOCIAL_USER_PROMPT, cts_psy.get_social_statements),
}

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template.SYSTEM_PROMPT_REASONING),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template.REFLECTION_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def construct_reflection(response, structure_class):
    scoring_list = []
    for section_name in structure_class.model_fields:
        question = getattr(response, section_name)
        assessment = getattr(question, 'criterion_assessment', "No assessment provided.")
        score = question.score
        scoring_list.append(
            f"For {section_name}, your score was: {score} and your reasoning was: {assessment}"
        )
    temp_prompt = "\n".join(scoring_list)

    reflection = f"Critique your rating in two sentences. \n\n{temp_prompt}"
    return reflection


def reflection_loop(
    generation_chain, reflection_chain, structure_class, full_user_prompt, steps
):
    messages = [("user", full_user_prompt)]

    # Initial entry point
    print("Initial LLM call")
    generation_response = generation_chain.invoke({"messages": messages})
    messages.append(("assistant", generation_response.model_dump_json()))

    for i in range(steps):
        print(f"Iteration {i+1}")
        # Reflection
        critique_prompt = construct_reflection(generation_response, structure_class)
        print("LLM reflecting...")
        reflection_response = reflection_chain.invoke(
            {"messages": [("user", critique_prompt)]}
        )  # Only give the critique
        critique = reflection_response.content.split("</think>")[-1].strip()

        messages.append(
            (
                "user",
                f"A supervisor critiqued your rating:\n\n{critique}\n\nRevise your rating accordingly if necessary.",
            )
        )
        print("LLM revising...")
        generation_response = generation_chain.invoke({"messages": messages})
        messages.append(("assistant", generation_response.model_dump_json()))

    return generation_response


def run(category, df, ids, model, reflection_model, run_dir, steps):
    structure_class, user_prompt, get_statements = CATEGORY_CONFIG[category]

    generation_chain = generation_prompt | make_safe_structured_output(model, structure_class)
    reflection_chain = reflection_prompt | reflection_model
    items = get_statements()
    transcript_map = df.set_index("filename")["text"].to_dict()

    run_dir = Path(run_dir)
    filename = f"{model.model_name.split('/')[-1]}_{category}_results.csv"
    results_path = run_dir / filename

    existing_df = None
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        already_done = set(existing_df["filename"].values)
        ids = [f for f in ids if f not in already_done]
        print(
            f"Resuming {category}: {len(already_done)} already done, {len(ids)} remaining."
        )

    print(f"Running for {category}...")
    results = []

    def process(f):
        transcript = transcript_map[f]
        full_user_prompt = f"\n\nTranscript: {transcript}\n\n{user_prompt}\n\n{items}\n\n{template.INSTRUCTION_PROMPT_REASONING}"
        response = reflection_loop(
            generation_chain, reflection_chain, structure_class, full_user_prompt, steps
        )
        row = {"filename": f}
        for field_name in structure_class.model_fields:
            item = getattr(response, field_name)
            row[f"{field_name}_score"] = item.score.value
            row[f"{field_name}_relevant_spans"] = item.relevant_spans
            row[f"{field_name}_criterion_assessment"] = getattr(item, 'criterion_assessment', None)
        return row

    if len(ids) > 0:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(process, f): f for f in ids}
            for future in tqdm(as_completed(futures), total=len(ids), desc=category):
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
        "--categories",
        nargs="+",
        choices=list(CATEGORY_CONFIG.keys()),
        default=list(CATEGORY_CONFIG.keys()),
        help="One or more categories to run (default: all)",
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
    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS,
        help=f"Number of reflection steps (default: {STEPS})",
    )

    args = parser.parse_args()

    # Load data
    df = pd.read_csv("../../data/synthetic_conversations.csv")
    ids = np.unique(df["filename"].values)
    print(len(ids))
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path("../../results") / f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{Path(__file__).stem}"
    )

    if args.model in ("gemma-4-E4B-it", "gemma-4-26B-A4B-it") and args.run_dir:
        model = MODEL_MAP[args.model](max_tokens=16384, port=PORT)
    else:
        model = MODEL_MAP[args.model](port=PORT)
    reflection_model = REFLECTION_MAP[args.model](port=PORT)
    for category in args.categories:
        run(category, df, ids, model, reflection_model, run_dir, args.steps)
