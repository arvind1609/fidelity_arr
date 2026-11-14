import argparse

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score

# Map ground truth column prefix → pred score column prefix
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

CATEGORY_ITEMS = list(range(1, 7))
CATEGORIES = {
    "agenda": CATEGORY_ITEMS,
    "feedback": CATEGORY_ITEMS,
    "understanding": CATEGORY_ITEMS,
    "interpersonal": CATEGORY_ITEMS,
    "collaboration": CATEGORY_ITEMS,
    "discovery": CATEGORY_ITEMS,
    "cognition": CATEGORY_ITEMS,
    "intervention": CATEGORY_ITEMS,
    "homework": CATEGORY_ITEMS,
    "social": CATEGORY_ITEMS,
}

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    if len(y_true) == 0:
        return dict(
            sensitivity=np.nan,
            specificity=np.nan,
            balanced_accuracy=np.nan,
            mcc=np.nan,
            n=0,
        )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    bal_acc = (
        (sensitivity + specificity) / 2
        if not (np.isnan(sensitivity) or np.isnan(specificity))
        else np.nan
    )
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "n": len(y_true),
    }


def bootstrap_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Return 95% CI (lower, upper) for each metric via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics_keys = ["sensitivity", "specificity", "balanced_accuracy", "mcc"]
    boot_stats = {k: [] for k in metrics_keys}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = binary_metrics(y_true[idx], y_pred[idx])
        for k in metrics_keys:
            boot_stats[k].append(m[k])
    ci = {}
    for k in metrics_keys:
        vals = np.array(boot_stats[k], dtype=float)
        ci[f"{k}_ci_low"] = np.nanpercentile(vals, 2.5)
        ci[f"{k}_ci_high"] = np.nanpercentile(vals, 97.5)
    return ci


def compute_kappa(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0 or len(np.unique(np.concatenate([y_true, y_pred]))) < 2:
        return np.nan
    return cohen_kappa_score(y_true, y_pred)


def _collect_gt_pred(merged, col_list):
    gt = (
        pd.concat([merged[c + "_gt"] for c in col_list if c + "_gt" in merged.columns])
        .reset_index(drop=True)
        .to_numpy()
    )
    pred = (
        pd.concat(
            [merged[c + "_pred"] for c in col_list if c + "_pred" in merged.columns]
        )
        .reset_index(drop=True)
        .to_numpy()
    )
    return gt.astype(int), pred.astype(int)


def compute_metrics(merged, score_cols):
    rows = []

    # Per-category (pool all items within each category)
    for cat in CATEGORIES:
        cat_items = [c for c in score_cols if c.startswith(cat + "_")]
        gt, pred = _collect_gt_pred(merged, cat_items)
        rows.append(
            {"category": cat, **binary_metrics(gt, pred), "kappa": compute_kappa(gt, pred), **bootstrap_ci(gt, pred)}
        )

    # General (first 5 categories) and Specific (last 5 categories)
    cat_names = list(CATEGORIES.keys())
    general_cats, specific_cats = cat_names[:5], cat_names[5:]
    for group_name, group_cats in [
        ("GENERAL", general_cats),
        ("SPECIFIC", specific_cats),
    ]:
        group_items = [
            c for c in score_cols if any(c.startswith(cat + "_") for cat in group_cats)
        ]
        gt, pred = _collect_gt_pred(merged, group_items)
        rows.append(
            {
                "category": group_name,
                **binary_metrics(gt, pred),
                "kappa": compute_kappa(gt, pred),
                **bootstrap_ci(gt, pred),
            }
        )

    # Overall
    all_gt, all_pred = _collect_gt_pred(merged, score_cols)
    rows.append(
        {
            "category": "OVERALL",
            **binary_metrics(all_gt, all_pred),
            "kappa": compute_kappa(all_gt, all_pred),
            **bootstrap_ci(all_gt, all_pred),
        }
    )

    return pd.DataFrame(rows).set_index("category")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)

    args = parser.parse_args()

    # Load results csv
    df = pd.read_csv(args.results_csv)
    ids = joblib.load("../data/labelled_ids.pkl")
    df = df[df.filename.isin(ids)]
    score_columns = [col for col in df.columns if "score" in col]
    df_pred = df.loc[:, ["filename"] + score_columns]
    print(len(df_pred))
    ground_truth = pd.read_csv("../data/ground_truth.csv")

    # Map ground truth column prefix and rename to match pred score columns
    col_map = {
        f"{letter}{i}": f"{name}_{i}_score"
        for letter, name in GT_TO_PRED.items()
        for i in range(1, 7)
    }

    columns_useful = ["Recording ID"] + list(col_map.keys())
    ground_truth = ground_truth[columns_useful]
    ground_truth = ground_truth.rename(columns=col_map)

    score_cols = [
        f"{cat}_{i}_score" for cat, items in CATEGORIES.items() for i in items
    ]

    print("ground_truth IDs:", ground_truth["Recording ID"].head().tolist())
    print("df_pred IDs:", df_pred["filename"].head().tolist())
    # Merge ground truth and predictions on recording ID
    merged = ground_truth.merge(
        df_pred.rename(columns={"filename": "Recording ID"}),
        on="Recording ID",
        suffixes=("_gt", "_pred"),
    )
    merged = (
        merged.drop(columns=["Recording ID"])
        .astype("Int64")
        .replace(9, 1)
    )
    print(len(merged))
    # Compute metrics
    metrics = compute_metrics(merged, score_cols)
    print(metrics)
    metrics.to_csv(f"{args.results_csv.split('.csv')[0]}_metrics.csv")
