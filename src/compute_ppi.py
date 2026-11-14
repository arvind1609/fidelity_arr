import numpy as np
import pandas as pd
import joblib
import argparse
import sys
import os

import ppi_py

CATEGORIES = {
    "agenda": [f"agenda_{i}_score" for i in range(1, 7)],
    "feedback": [f"feedback_{i}_score" for i in range(1, 7)],
    "understanding": [f"understanding_{i}_score" for i in range(1, 7)],
    "interpersonal": [f"interpersonal_{i}_score" for i in range(1, 7)],
    "collaboration": [f"collaboration_{i}_score" for i in range(1, 7)],
    "discovery": [f"discovery_{i}_score" for i in range(1, 7)],
    "cognition": [f"cognition_{i}_score" for i in range(1, 7)],
    "intervention": [f"intervention_{i}_score" for i in range(1, 7)],
    "homework": [f"homework_{i}_score" for i in range(1, 7)],
    "social": [f"social_{i}_score" for i in range(1, 7)],
}

CATEGORIES['GENERAL'] = np.concatenate((CATEGORIES['agenda'],
                        CATEGORIES['feedback'],
                        CATEGORIES['understanding'],
                        CATEGORIES['interpersonal'],
                        CATEGORIES['collaboration'],
                        ))

CATEGORIES['SPECIFIC'] = np.concatenate((CATEGORIES['discovery'],
                        CATEGORIES['cognition'],
                        CATEGORIES['intervention'],
                        CATEGORIES['homework'],
                        CATEGORIES['social'],
                        ))

CATEGORIES['OVERALL'] = np.concatenate((CATEGORIES['GENERAL'], CATEGORIES['SPECIFIC']))

GT_TO_PRED = {
    "A": "agenda",
    "B": "feedback",
    "C": "understanding",
    "D": "interpersonal",  # d) Interpersonal Effectiveness
    "E": "collaboration",
    "F": "discovery",  # f) Guided Discovery
    "G": "cognition",  # g) Focus on key cognitions
    "H": "intervention",  # h) Choices of intervention
    "I": "homework",
    # J = Quality of Intervention (single 0-6 score, no items — not in df_pred)
    "K": "social",  # k) Social Skills Training
}

def compute_ppi(ground_truth, df_labelled_results, df_unlabelled_results, category):
    cat_cols = list(CATEGORIES[category])

    merged = ground_truth[['Recording ID'] + cat_cols].merge(
        df_labelled_results[['filename'] + cat_cols],
        left_on='Recording ID',
        right_on='filename',
        suffixes=('_gt', '_pred')
    ).dropna()

    gt_cols   = [f"{c}_gt"   for c in cat_cols]
    pred_cols = [f"{c}_pred" for c in cat_cols]

    y = merged[gt_cols].values.flatten().astype(int)
    yhat = merged[pred_cols].replace(9, 1).values.flatten().astype(int)
    yhat_unlabelled = df_unlabelled_results[cat_cols].dropna().replace(9, 1).values.flatten().astype(int)

    ci_low, ci_high = ppi_py.ppi_mean_ci(y, yhat, yhat_unlabelled, alpha=0.05)
    raw_unlabelled_mean = np.mean(yhat_unlabelled)
    ppi_rectified_mean = ppi_py.ppi_mean_pointestimate(y, yhat, yhat_unlabelled)
    delta = ppi_rectified_mean - raw_unlabelled_mean

    return round(y.mean(), 4), round(raw_unlabelled_mean, 4), round(ppi_rectified_mean[0], 4), round(delta[0], 4), round(ci_low[0], 4), round(ci_high[0], 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)
    args = parser.parse_args()

    df_all_results = pd.read_csv(args.results_csv)
    labelled_ids = joblib.load("../data/labelled_ids.pkl")

    df_labelled_results = df_all_results[df_all_results.filename.isin(labelled_ids)]
    df_unlabelled_results = df_all_results[~df_all_results.filename.isin(labelled_ids)]

    ground_truth = pd.read_csv("../data/ground_truth.csv")
    ground_truth = ground_truth[ground_truth['Recording ID'].isin(labelled_ids)]
    ground_truth = ground_truth.replace(9.0, 1.0)

    rename_map = {}
    for letter, name in GT_TO_PRED.items():
        rename_map[letter] = name
        for i in range(1, 7):
            rename_map[f"{letter}{i}"] = f"{name}_{i}_score"

    ground_truth = ground_truth.rename(columns=rename_map)

    rows = []
    for category in CATEGORIES:
        theta, raw_unlabelled_mean, ppi_rectified_mean, delta, ci_low, ci_high = compute_ppi(ground_truth, df_labelled_results, df_unlabelled_results, category)
        print(category, theta, ci_low, ci_high)
        rows.append({"category": category, "labelled_mean": theta, "raw_unlabelled_mean": raw_unlabelled_mean, "ppi_rectified_mean": ppi_rectified_mean, "delta": delta, "ci_low": ci_low, "ci_high": ci_high})

    base = os.path.splitext(os.path.basename(args.results_csv))[0]
    out_path = os.path.join(os.path.dirname(args.results_csv), f"{base}_ppi.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved PPI results to {out_path}")
