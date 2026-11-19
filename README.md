# SimulacraBench submission — team junhyun-dev

Code for our final submission to the [NeurIPS 2026 SimulacraBench competition](https://www.codabench.org/competitions/17822/).
The method is described in the 4-page report in [`report/`](report/).

For every blank PREDICT cell, `predict(frame, schema)` returns a probability
vector over the item's codes (and `NA_GATED` where the item can be skipped).
All fitting happens inside `predict()` on the rows the grader passes in. The
respondent data are not released, so there is nothing to train offline.

## Files

| Path | What it is |
|---|---|
| `main.py` | The final submission (version 8F). Byte-identical to the uploaded ZIP's `main.py`. |
| `requirements.txt` | `tabicl==2.2.0`, `einops==0.8.2`. numpy, pandas, scikit-learn and torch come preinstalled in the grader's Python 3.13 image. |
| `models.txt` | `jingang/TabICL` (fetched by the grader from Hugging Face before the run). |
| `probe/main.py` | The diagnostic probe (Section 5 of the report). It runs the same computation and then raises an exception whose type says how far TabICL got. The type is one of eleven fixed values set only by the state of our own code; no respondent value, prediction, item name or score is encoded. It was submitted once. |
| `probe/make_probe.py` | Builds `probe/main.py` from `main.py` with exact-string edits. Run it from the repository root. |
| `report/` | LaTeX source of the report. |

## Base model

TabICLv2, used in context without fine-tuning:
`jingang/TabICL`, checkpoint `tabicl-classifier-v2-20260212.ckpt`,
revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` (BSD-3-Clause).
We use no language model, no question text and no external data.
Adding TabICLv2 followed the public README of
[exochard/simulacrabench-v33](https://github.com/exochard/simulacrabench-v33)
(Apache-2.0, read 2026-11-13); we read only its README and took no code from it.

## Running it

The evaluation data are not public. You can rerun the code on the
organisers' practice data with their tools
([SituatedEvals/public](https://github.com/SituatedEvals/public), commit `aa03186f`):

```bash
# in a clone of SituatedEvals/public, in one Python 3.13 environment
pip install numpy pandas scikit-learn torch -r /path/to/this/repo/requirements.txt
python make_sandbox.py --schema data/unicef.json --out _sandbox/unicef
mkdir my_submission && cp /path/to/this/repo/{main.py,requirements.txt,models.txt} my_submission/
python score.py --submission my_submission --data _sandbox/unicef \
                --schema data/unicef.json --phase 1
```

Practice data are synthetic, so these scores only show that the code runs.
Without a GPU the code skips TabICL; set `SB_TABICL_CPU=1` to run it on CPU
(slow). All randomness is seeded from `20261112`.

## Scores

Noised and rounded by the platform (one rounding step is about 0.0135, 0.0150
and 0.0193 skill on UNICEF, World Bank and UNHCR). The same v5 ZIP scored
0.8650 and 0.8457 on UNHCR on consecutive days, so these numbers do not rank
the versions. The Final row is the Test leaderboard score (held-out respondents,
so not directly comparable with the Development rows).

| Version | What changes | UNICEF | World Bank | UNHCR | Overall |
|---|---|---:|---:|---:|---:|
| v4 | backoff, recursion, network, trees | 0.3499 | 0.2335 | 0.8457 | 0.4764 |
| v5 | + TabICLv2 | 0.3499 | 0.2335 | 0.8650 | 0.4828 |
| v8 | v5 + refit on all visible rows for hidden rows | 0.3499 | 0.2335 | 0.8457 | 0.4764 |
| Final (Test) | v8 with the 3600 s budget, submission 1059398 | 0.3419 | 0.2343 | 0.8734 | 0.4832 |

## Use of AI tools

The code, this README and the report were written with an AI coding assistant
(Claude) working under the author's direction. All numbers come from our
submissions, the code, or the cited sources.

## Licence

MIT, see [LICENSE](LICENSE). TabICL is not included; it is downloaded at run time under its own licence.
