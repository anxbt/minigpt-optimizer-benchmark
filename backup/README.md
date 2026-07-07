# Backup and Recovery Notes

This repository is intended to be a recoverable backup for the MiniGPT optimizer benchmark project.

## What is stored directly in Git

- Source code and training scripts.
- Notebook-style learning labs.
- Benchmark configs, summaries, plots, reports, and paper artifacts.
- Raw metric JSONL files needed to regenerate plots.
- The local toy checkpoint: `gpu_benchmark/downloaded_runs/minigpt_toy_muon/checkpoints/latest.pt`.

## What is too large for normal Git

The real EC2 pretraining checkpoints are larger than GitHub's normal per-file limit. They are listed in:

- `backup/ec2_checkpoint_manifest.tsv`

At the time this manifest was generated, EC2 had 46 checkpoint files totaling roughly 39GB.

## Why not normal Git for all weights?

GitHub rejects normal Git files larger than 100MB. Most real checkpoint files are 564MB to 1.49GB each.
Use GitHub Releases, S3, Hugging Face Hub, or Git LFS for those files.

## Recommended restore model

1. Clone this repository.
2. Create the Python environment from the scripts/docs.
3. Download large checkpoint artifacts from the external backup location.
4. Place them back under the same relative run directories.

## EC2 source path

The current EC2 source path for full benchmark runs is:

```text
/home/anubrat/pretrain_runs
```
