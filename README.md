# sieve-backdoors

A research harness for evaluating LLM backdoor detectors under a common, pre-registered adaptive
attacker. It builds a grid of (detector x attack) cells and scores each one with a ladder of gates
(detection, surface-confound control, adaptive re-test, and a causal-sufficiency check) so a
detector only counts as "caught" if it detects the backdoor for a reason that holds up.

This is experimental research code, not a product or a benchmark result to cite. See FINDINGS.md for
the honest write-up, including the limitations.

## Status

Null result at the scale tested. Across a 5x6 grid on Qwen2.5-7B and Llama-3.2-3B (N=10 models per
class), no attack cleanly evaded the detector set once measurement bugs were fixed, and the detectors
that did fire often failed the causal check. The results are confounded by small sample size, a
single payload family in the main grid, and reduced reimplementations of some detectors. Treat the
numbers as a demonstration of the harness, not as claims about the published detectors.

## Safety scope

Backdoors here are planted only as detection targets. Payloads are restricted in code to four benign,
inert markers (see `sieve_backdoors/payloads/benign.py`): a fixed canary string, a benign sentiment
label flip, refusal-to-compliance on an explicitly benign request pool, and a benign marker token.
Nothing else. No harmful content, and nothing is uploaded anywhere.

## Layout

```
configs/                pre-registration config + benign request pool
sieve_backdoors/
  models/               model loading, LoRA fine-tuning, activations, data
  payloads/             the four allowed benign payloads + hard guard
  attacks/              A1..A8 backdoor constructions
  detectors/            D1,D2,D4,D5,D6 detector implementations
  attacker/             common attacker config, eval-trigger helper
  sieve/                vendored validity-audit gate/control/verdict code
  grid/                 runner, scorecard, causal gate, verdict ladder
tests/                  interface, safety, and verdict-logic tests
scripts/                smoke, positive control, grid runner, offline re-analysis
results/                emitted scorecards (fixed and pre-fix, for the record)
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .[dev]
```

## Run

```bash
.venv/bin/pytest -q                          # interface + safety + logic tests
.venv/bin/python scripts/smoke_rig.py        # plant one backdoor, verify it fires
.venv/bin/python scripts/positive_control.py # positive control on a small model
.venv/bin/python scripts/run_grid.py --help  # full grid options
```

Trained adapters are cached on disk and reused, so re-scoring with a new detector or threshold does
not retrain. `scripts/reanalyze.py` recomputes verdicts under alternate thresholds offline from a
saved scorecard.

## Notes

The gate/control/verdict machinery under `sieve_backdoors/sieve/` is vendored from a separate
validity-audit project and carries provenance headers; it is not modified here.
