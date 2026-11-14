import pandas as pd
import sys 
import ast
sys.path.append("../")
import prompts.cts_psy as cts_psy


CBT_CATEGORIES = ['discovery', 'cognition', 'intervention', 'homework', 'social']
CBT_STATEMENTS = {
    'discovery':    {i: getattr(cts_psy, f'DISCOVERY_{i}') for i in range(1, 7)},
    'cognition':    {i: getattr(cts_psy, f'COGNITION_{i}') for i in range(1, 7)},
    'intervention': {i: getattr(cts_psy, f'INTERVENTION_{i}') for i in range(1, 7)},
    'homework':     {i: getattr(cts_psy, f'HOMEWORK_{i}') for i in range(1, 7)},
    'social':       {i: getattr(cts_psy, f'SOCIAL_{i}') for i in range(1, 7)},
}

def compute_bottom3_scores(df):
    # Compute mean score per CBT category per row
    for cat in CBT_CATEGORIES:
        score_cols = [f'{cat}_{i}_score' for i in range(1, 7)]
        df[f'{cat}_total_score'] = df[score_cols].sum(axis=1)

    # Select 3 least-scored CBT areas per row
    total_cols = [f'{cat}_total_score' for cat in CBT_CATEGORIES]
    df['bottom_3_cbt_areas'] = df[total_cols].apply(
        lambda row: row.nsmallest(3).index.str.replace('_total_score', '').tolist(),
        axis=1
    )
    return df

def get_bottom3_details(row):
    records = []
    for cat in row['bottom_3_cbt_areas']:
        for i in range(1, 7):
            records.append({
                'key': cat,
                'statement': CBT_STATEMENTS[cat][i],
                'score': row[f'{cat}_{i}_score'],
                'relevant_spans': row[f'{cat}_{i}_relevant_spans'],
                'criterion_assessment': row[f'{cat}_{i}_criterion_assessment'],
            })
    return records

def parse_spans(val):
    if pd.isna(val) or val == '' or val == '[]':
        return 'No explicit relevant span'
    try:
        spans = ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return 'No explicit relevant span'
    if not spans:
        return 'No explicit relevant span'
    return spans

def format_details_for_prompt(details):
    lines = []
    for i, statements in enumerate(details):
        if i % 6 == 0:
            lines.append(f"## {statements['key'].upper()} ##")
        lines.append(f"Fidelity Statement: {statements['statement']}")
        lines.append(f"Score: {statements['score']}")
        spans = parse_spans(statements['relevant_spans'])
        lines.append(f"Relevant Spans: {spans if isinstance(spans, str) else spans[0]}")
        lines.append(f"Criterion Assessment: {statements['criterion_assessment']}")
        lines.append("")
    return "\n".join(lines)

def generate_feedback_prompt(df):
    df = compute_bottom3_scores(df)
    df['bottom_3_details'] = df.apply(get_bottom3_details, axis=1)
    df['bottom_3_scores'] = df.apply(
        lambda row: {cat: row[f'{cat}_total_score'] for cat in row['bottom_3_cbt_areas']},
        axis=1
    )
    df = df.loc[:, ['filename', 'bottom_3_scores', 'bottom_3_details']]
    df['prompt'] = df.bottom_3_details.apply(lambda x: format_details_for_prompt(x))
    return df
