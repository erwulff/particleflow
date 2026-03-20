# Running ONNX inference timing

These scripts can be run directly without modifying them if the environment already has the required Python packages, the TFDS CMS dataset cache, and a trained experiment directory containing both a checkpoint and the matching `model_kwargs.pkl`.

## Required inputs

- `--checkpoint`: trained checkpoint file, for example `<path/to/experiment>/checkpoints/checkpoint.pth`
- `--model-kwargs`: matching `model_kwargs.pkl` from the same experiment directory, for example `<path/to/experiment>/model_kwargs.pkl`
- `--data-dir`: TFDS CMS dataset root, for example `<path/to/tfds-datadir>`
- `--dataset`: TFDS dataset name such as `cms_pf_ttbar`, `cms_pf_qcd`, or `cms_pf_ztt`

Important optional arguments:

- `--num-events`: number of events to benchmark
- `--device`: `cpu` or `cuda`
- `--num-threads`: CPU threads for PyTorch and ONNX Runtime
- `--configs`: subset of benchmark configs to run; if omitted, all supported configs are run
- `--outdir`: output directory for ONNX exports, plots, and `summary.json`

Supported configs:

- `PT_MATH_FP32`
- `PT_MATH_FP16`
- `PT_FLASH_FP16`
- `ONNX_MATH_FP32`
- `ONNX_MATH_FP16`
- `ONNX_FLASH_FP32_FP16`
- `ONNX_FLASH_FP16`

## Smoke test

Run this first to verify the environment and dataset path:

```bash

python scripts/cms-validate-onnx.py \
  --checkpoint <path/to/experiment>/checkpoints/checkpoint-10000.pth \
  --model-kwargs <path/to/experiment>/model_kwargs.pkl \
  --data-dir <path/to/tfds-datadir> \
  --dataset cms_pf_ttbar \
  --num-events 10 \
  --device cpu \
  --num-threads 4 \
  --outdir <path/to/output-dir>
```

## Expected output

The benchmark run writes these to `--outdir`:

- exported ONNX models for the ONNX configs that were run
- comparison plots such as `runtime_vs_size.pdf`, `runtime_dist.pdf`, `runtime_ranking.pdf`, `model_error_ranking.pdf`, `jet_pt_distribution.pdf`, and `comp_abs_*.pdf`
- `summary.json` with per-event runtimes, OOM flags, MAE relative to the baseline, and system metadata

## Comparing multiple runs

Run `scripts/cms-validate-onnx.py` once per model or setup, each with its own output directory under a common parent:

```text
<path/to/benchmark-parent>/
  run_a/
    summary.json
  run_b/
    summary.json
```

Then aggregate them with:

```bash
python scripts/plot-onnx-summary.py \
  --indir <path/to/benchmark-parent> \
  --outdir <path/to/aggregate-output-dir>
```

This produces aggregate PDF and PNG plots plus `summary.txt` in `--outdir`.
