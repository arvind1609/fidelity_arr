from pathlib import Path

import pandas as pd


def merge_results(results_dir, model, ignore_row_count=False):
    results_dir = Path(results_dir)
    csvs = sorted(
        csv for csv in results_dir.glob(f"{model}_*_results*.csv")
        if "merged" not in csv.name
    )
    if not csvs:
        raise FileNotFoundError(
            f"No result CSVs found in {results_dir} for model '{model}'"
        )
    merged = pd.read_csv(csvs[0])
    base_len = len(merged)
    for csv in csvs[1:]:
        df = pd.read_csv(csv)
        if not ignore_row_count and len(df) != base_len:
            raise ValueError(
                f"Row count mismatch: {csvs[0].name} has {base_len} rows, "
                f"{csv.name} has {len(df)} rows"
            )
        merged = merged.merge(df, on="filename", how="outer")
    output_path = results_dir / f"{model}_merged_results.csv"
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(csvs)} files ({len(merged)} rows) -> {output_path}")
    return merged
