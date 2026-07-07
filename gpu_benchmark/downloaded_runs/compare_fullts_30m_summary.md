# Full TinyStories 30-minute AdamW vs Hybrid Muon comparison

Dataset cache: full TinyStories, GPT-2 tokenizer, 512-token context. Effective batch fixed at 65,536 tokens/update.

Important rigor note: these two completed runs were launched before explicit `--seed` support was patched, so they are operational benchmark evidence, not yet a final paper-style claim. The script now supports `--seed` for the next controlled rerun.

| Run | Status | Steps | Tokens seen | Best val loss | Best val step | Last val loss | Last train loss | Median tok/s | Max GPU GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | stopped_max_minutes | 4439 | 290,914,304 | 1.4898 | 4200 | 1.4949 | 1.4389 | 165960 | 7.17 |
| Hybrid Muon | stopped_max_minutes | 4009 | 262,733,824 | 1.4282 | 4000 | 1.4282 | 1.4147 | 149605 | 7.05 |

## Nearest common-token comparison

Common token budget is approximately the smaller run: 262,733,824 tokens.

| Run | Nearest step | Nearest tokens | Val loss |
|---|---:|---:|---:|
| AdamW | 4000 | 262,144,000 | 1.5113 |
| Hybrid Muon | 4000 | 262,144,000 | 1.4282 |

## Interpretation

- AdamW was faster in wall-clock throughput: about 166k tok/s vs Muon about 150k tok/s.
- Muon reached a lower validation loss in this run, even though it processed fewer tokens in 30 minutes.
- The result is promising but not final because we need seed-matched reruns before making a strong claim.
- The full cache fixed the previous overfitting issue: validation loss kept improving instead of exploding upward.
