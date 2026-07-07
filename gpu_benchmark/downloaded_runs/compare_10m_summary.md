# TinyStories 60M 10-minute smoke comparison

Important: this used the small smoke cache: 1,000,000 train tokens and 131,072 validation tokens. The model saw the same tiny train set many times, so late validation loss rising is expected overfitting.

| Run | Status | Last step | Tokens seen | Best val loss | Best val step | Last val loss | Last train loss | Median tail tok/s | Max GPU alloc GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | stopped_max_minutes | 3377 | 55,328,768 | 2.5616 | 900 | 4.3172 | 0.1528 | 106073 | 2.53 |
| Hybrid Muon | stopped_max_minutes | 2725 | 44,646,400 | 2.6966 | 700 | 4.3913 | 0.1569 | 83602 | 2.41 |

## Reading the result

- Both runs learned: training loss dropped from about 10.9 to below 0.2.
- Both overfit: validation loss improved early, then rose while training loss kept falling.
- AdamW reached more steps/tokens in the same wall time, because this simple Muon implementation has extra matrix-update cost.
- On this tiny capped dataset, AdamW had a better best validation loss. This does not disprove Muon; this is a pipeline/sanity benchmark, not a paper-scale benchmark.
