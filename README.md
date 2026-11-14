## Installation

### Requirements

- Linux with NVIDIA GPUs (the serve scripts assume 4 GPUs by default)
- Python 3.11+ for the main environment, Python 3.12 for the vLLM environment
- [uv](https://docs.astral.sh/uv/)

### 1. Main environment

The packages used by the pipeline scripts are installed through uv. From the repository root:

```bash
uv sync                     # creates .venv from pyproject.toml / uv.lock
source .venv/bin/activate   # or prefix commands with `uv run`
```

### 2. vLLM environment

The models are served locally with [vLLM](https://docs.vllm.ai/) through an OpenAI-compatible API. vLLM goes in its own virtual environment, `.venv-vllm`, so its pinned dependencies don't conflict with the main environment. The serve scripts activate this environment automatically if it exists.

```bash
uv venv .venv-vllm --python 3.12
source .venv-vllm/bin/activate
uv pip install vllm==0.31.0
deactivate
```

Some models (e.g. Gemma) are gated on Hugging Face. Accept the license on the model page, then log in once:

```bash
uv run huggingface-cli login
```



### 3. Serve a model

Run a serve script from the repository root. It starts a vLLM server and writes its log to `logs/`.

```bash
bash serve/serve_gemma26b.sh
```


| Script                  | Model                                        | Port | GPUs |
| ----------------------- | -------------------------------------------- | ---- | ---- |
| `serve_qwen9b.sh`       | `Qwen/Qwen3.5-9B`                            | 8000 | 0–3  |
| `serve_gemma26b.sh`     | `google/gemma-4-26B-A4B-it`                  | 8000 | 0–3  |
| `serve_gemma4b.sh`      | `google/gemma-4-E4B-it`                      | 8000 | 1–2  |


Change `CUDA_VISIBLE_DEVICES` and `NUM_GPUS` in the script to fit your hardware.

## Fidelity Estimation



### 1. Run the pipelines

With the server running and the main environment active, run the scripts from inside their own folder, since they use paths relative to it. They read `data/synthetic_conversations.csv` and write results to `results/`.

```bash
cd src/rationalization
bash run.sh                                  # all rationalization (RZ) scripts
python rz_category.py --model Qwen3.5-9B     # or a single script
```

The `evidential_estimation` (EV) and `reflection` (RF) folders work the same way. Each folder has one script per decomposition (`category`, `section`, `clinical`, `observability`, `full`). `--model` takes `Qwen3.5-9B`, `gemma-4-E4B-it` or `gemma-4-26B-A4B-it`. All servers and scripts use port 8000, so serve one model at a time. Pass `--run_dir` to resume an earlier run.

### 2. Compute metrics

Run these from `src/`. The metric scripts compare the model's scores against the
binary labels in `data/ground_truth.csv`, for the recordings listed in
`data/labelled_ids.pkl`.

```bash
cd src
python merge_results.py ../results/<run_dir> <model>                       # -> <model>_merged_results.csv
python compute_performance.py --results_csv ../results/<run_dir>/<model>_merged_results.csv
python compute_ppi.py         --results_csv ../results/<run_dir>/<model>_merged_results.csv
```

1. `merge_results.py` joins the per-category result CSVs in a run directory into one file.
2. `compute_performance.py` reports various performance metrics and saves to `<model>_merged_results_metrics.csv`.
3. `compute_ppi.py` uses prediction-powered inference to estimate the mean fidelity
  score on the unlabelled recordings, correcting the model's bias with the labelled
   ones. Saves `<model>_merged_results_ppi.csv`.



## Feedback Pipeline

Run this from `src/feedback/`. Feedback generation takes a merged results CSV as input and provides actionable feedback for the therapist

```bash
cd src/feedback
python generation.py --results_csv ../../results/<run_dir>/<model>_merged_results.csv   # -> results_feedback/
```

`generation.py` selects the three lowest-scoring CBT-specific categories per session and generates supervisory feedback from their evidence.

> **Note:** The RAG feedback script has been removed from for anonymity. It will be added in the camera-ready version.

