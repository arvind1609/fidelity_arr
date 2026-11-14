import argparse

from utils import merge_results

parser = argparse.ArgumentParser()
parser.add_argument("results_dir")
parser.add_argument("model")
parser.add_argument("--ignore_row_count", action="store_true")
args = parser.parse_args()

merge_results(args.results_dir, args.model, args.ignore_row_count)
