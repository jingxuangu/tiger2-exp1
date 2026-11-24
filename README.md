# tiger2-exp1 · Terra Top 3 with NeuriCo

One request and one fixed literature review. Generate exactly eight Terra v7
ideas, rank all eight by the published Terra RM's calibrated combined
`reward`, and run NeuriCo Standard on ranks 1–3 with paper writing off.
The RM sees the original idea, request, and literature review. The release's
Python scorer also returns the raw reward and four supervisor ratings. The
HTTP server's additional format and length costs are outside this ranking.
An invalid Top 3 proposal blocks execution and leaves the scores available
for inspection.

## Inputs and environments

The pipeline lives in `TIGER-2.0/tiger2-experiments/tiger2-exp1`. It uses
the workspace's existing `NeuriCo` and `rl-pipline` checkouts. For a
standalone clone, set `TIGER2_WORKSPACE` to the absolute path containing
those two checkouts and `TIGER2_EXP1_DIR` to the absolute clone path. The
TIGER 1.0 code, restricted RM, model weights, and run artifacts are separate
from this source repository.

- Query file is a one-line JSONL object with nonempty `id`, `request`, and
  `literature_review`, matching TIGER 1.0's data contract.
- [config.json](config.json) controls the base model, seeds, token limit,
  research domain, Codex model, and stage timeouts. The default base is the
  Qwen checkpoint listed in the policy release metadata. The original
  training setup calls its cluster copy Qwen3-8B-Nonthinking. Generation
  uses the tokenizer shipped with the policy and disables thinking.
- NeuriCo runs natively through its locked `.venv` with the Codex CLI.
  Initialize it with `uv sync --project "$TIGER2_WORKSPACE/NeuriCo" --locked`. Its runner uses an isolated registry, workspace and Codex
  session directory for each attempt. The existing Codex login is copied into
  a temporary directory for the run.
- The generation and scoring jobs use the existing DSI
  `ideation-rm-py312` environment (PyTorch 2.13.0+cu130 and Transformers 5.15.1).
  Scoring loads the pinned private Terra RM snapshot and existing Flash Linear
  Attention dependencies. `TIGER2_EXP1_MODEL_PYTHON`, `TIGER2_EXP1_FLA_DEPS` and
  `TIGER2_EXP1_RM_SNAPSHOT` override those locations. The published scoring setup
  uses one H200 with BF16, and the job wrapper uses the shared cache for the
  RM's pinned base weights.
- [prepare-policy.sh](prepare-policy.sh) downloads the pinned four-way FSDP
  checkpoint and uses the existing verl 0.9 environment to export PEFT LoRA
  and stage the original tokenizer. `TIGER2_EXP1_VERL_PYTHON` overrides that interpreter.

## Submission

From the workspace root, after reviewing the query and configuration:

```bash
export TIGER2_WORKSPACE="$PWD"
export TIGER2_EXP1_DIR="$PWD/tiger2-experiments/tiger2-exp1"
run="$TIGER2_EXP1_DIR/runs/query-01"
query=/absolute/path/to/query.jsonl
mkdir -p -m 700 "$run"
bash "$TIGER2_EXP1_DIR/prepare-policy.sh" "$TIGER2_EXP1_DIR/models/terra-policy"
adapter="$TIGER2_EXP1_DIR/models/terra-policy/lora_adapter"

generate=$(sbatch --parsable --partition=general --gres=gpu:a100:1 -c 4 \
    --mem=48G -t 01:00:00 -o "$run/generate-%j.log" \
    "$TIGER2_EXP1_DIR/job.sh" generate "$run" --query "$query" --adapter "$adapter")
score=$(sbatch --parsable --partition=general --gres=gpu:h200:1 -c 8 \
    --mem=96G -t 01:00:00 --dependency="afterok:${generate%%;*}" \
    -o "$run/score-%j.log" "$TIGER2_EXP1_DIR/job.sh" score "$run")
sbatch --partition=general --array=1-3 --gres=gpu:a100:1 -c 8 \
    --mem=48G -t 02:00:00 --dependency="afterok:${score%%;*}" \
    -o "$run/neurico-%A_%a.log" "$TIGER2_EXP1_DIR/job.sh" execute "$run"
```

The research GPU allocation is a starting budget. Set the same allocation
for each candidate, based on the selected query. To rerun a failed candidate,
submit the execute job with its rank as the only array index.

## Outputs

- `input.json` records the query, full TIGER v7 prompt and generation settings.
- `candidates.json` stores all eight raw outputs, sample seeds and token counts.
- `scores.json` stores every reward and returned RM detail and `selection.json`
  records the exact Top 3 in order. Scores tie-break by candidate ID.
- Each selected candidate has a separate attempt directory with the original
  NeuriCo input, runner log, research workspace, and private trace archive.
  The workspace contains `REPORT.md`, code, logs and results. The trace
  archive indexes native Codex sessions and available per-invocation
  records. A missing report, absent sessions, or a recorded archive gap fails
  the attempt. `complete.json` points to the accepted attempt, report and
  trace manifest. Failed attempts remain available for diagnosis.
