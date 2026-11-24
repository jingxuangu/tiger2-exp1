#!/usr/bin/env python3
"""Generate eight Terra ideas, rank them with Terra RM, and execute the top three."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TIGER2_WORKSPACE", HERE.parent.parent)).resolve()
NEURICO = ROOT / "NeuriCo"
TIGER = ROOT / "rl-pipline"
sys.path.insert(0, str(TIGER / "src"))
POLICY = "chailabideation/qwen3-8b-v7-from0-step50"
POLICY_REVISION = "b7c5c201ceb54a2a3aa56fac21be8279298c1431"
RM = "chailabideation/qwen3.8-27b-rm-terra-with-literature-2000-combined-no-coherence"
RM_REVISION = "d515befe7b58ba617793c551b30ad959f723b580"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    Path(stream.name).replace(path)


def generate(args, run):
    from ideation_rl.data import read_rows, to_verl

    config = read(args.config)
    rows = read_rows(args.query)
    if len(rows) != 1:
        raise ValueError("The query JSONL must contain exactly one request.")
    query = rows[0]
    prompt = to_verl(query, "tiger2-exp1", "v7")["prompt"]
    adapter = args.adapter.resolve()
    tokenizer_path = adapter.parent / "tokenizer"
    required = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors",
                tokenizer_path / "tokenizer_config.json"]
    if not all(path.is_file() for path in required):
        raise ValueError("Expected a PEFT adapter exported from the Terra FSDP checkpoint.")
    manifest = dict(config=config, query=query, prompt=prompt, policy=POLICY,
                    policy_revision=POLICY_REVISION, adapter=str(adapter))
    manifest_path = run / "input.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("Inputs changed. Choose a new run directory.")
    output = run / "candidates.json"
    if output.exists():
        return
    write(manifest_path, manifest)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Run generation in a GPU allocation.")
    base = config["policy_base"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokens = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, enable_thinking=False,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to("cuda")
    prompt_length = tokens.input_ids.shape[-1]
    if prompt_length > 16384:
        raise ValueError(f"Prompt exceeds the TIGER limit: {prompt_length} tokens.")
    base_model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": 0},
        revision=config.get("policy_base_revision"),
    )
    model = PeftModel.from_pretrained(base_model, adapter).eval()
    candidates = []
    for index in range(8):
        seed = config["seed"] + index
        torch.manual_seed(seed)
        with torch.inference_mode():
            result = model.generate(
                **tokens, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = result[0, prompt_length:]
        candidates.append(dict(id=f"idea-{index + 1:02d}", seed=seed,
                               tokens=len(completion),
                               raw=tokenizer.decode(completion, skip_special_tokens=True)))
    write(output, candidates)


def score(args, run):
    from terra_combined_rm import TerraCombinedRM
    from ideation_reward_model.idea_format import parse_json_idea

    query = read(run / "input.json")["query"]
    candidates = read(run / "candidates.json")
    output = run / "scores.json"
    adapter = os.environ.get("TIGER2_EXP1_RM_SNAPSHOT", RM)
    run_config = dict(adapter=RM, revision=RM_REVISION, source=adapter)
    if output.exists():
        scored = read(output)
        if scored["model"] != run_config:
            raise ValueError("Scoring inputs changed. Choose a new run directory.")
        ranked = scored["ranking"]
    else:
        model = TerraCombinedRM(adapter, revision=RM_REVISION)
        values = model.score_pairs([
            dict(idea=row["raw"], request=query["request"],
                 literature_review=query["literature_review"])
            for row in candidates
        ])
        ranked = sorted(
            [dict(id=row["id"], rm_score=value["reward"], details=value)
             for row, value in zip(candidates, values, strict=True)],
            key=lambda row: (-row["rm_score"], row["id"]),
        )
        write(output, dict(model=run_config, ranking=ranked))
    by_id = {row["id"]: row for row in candidates}
    for row in ranked[:3]:
        parse_json_idea(by_id[row["id"]]["raw"])
    write(run / "selection.json", [row["id"] for row in ranked[:3]])


def execute(args, run):
    sys.path.insert(0, str(TIGER / "reward_model" / "src"))
    from ideation_reward_model.idea_format import parse_json_idea

    selected = read(run / "selection.json")
    candidate_id = selected[args.rank - 1]
    candidate = next(row for row in read(run / "candidates.json") if row["id"] == candidate_id)
    inputs = read(run / "input.json")
    query, config = inputs["query"], inputs["config"]
    idea = parse_json_idea(candidate["raw"])
    destination = run / candidate_id
    destination.mkdir(exist_ok=True)
    with (destination / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (destination / "complete.json").exists():
            return
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=destination))
        specification = {"idea": dict(
            title=idea["title"], domain=config["domain"], hypothesis=idea["description"],
            background={"description": (f"Original research request:\n{query['request']}\n\n"
                        f"Supplied literature review:\n{query['literature_review']}")},
        )}
        write(attempt / "idea.json", specification)
        interpreter = NEURICO / ".venv/bin/python"
        with (attempt / "runner.log").open("wb") as log:
            result = subprocess.run(
                [str(interpreter), "-u", __file__, "_research", str(attempt),
                 "--config", str(run / "input.json")],
                cwd=NEURICO, stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"NeuriCo or artifact validation failed. See {attempt / 'runner.log'}")
        write(destination / "complete.json", dict(attempt=attempt.name, **read(attempt / "result.json")))


def research(args):
    from contextlib import redirect_stdout

    sys.path[:0] = [str(NEURICO / "src"), str(NEURICO / "integrations/harbor")]
    from core.idea_manager import IdeaManager
    from core.runner import main as runner_main
    from neurico_harbor_agent.internal_traces import InternalTraceArchive, private_trace_redactor
    from neurico_harbor_agent.retention_hooks import install_retention_hooks

    attempt = args.run.resolve()
    config = read(args.config)["config"]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=NEURICO, text=True,
    ).strip()
    original_codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    auth = original_codex_home / "auth.json"
    if not auth.is_file():
        raise FileNotFoundError(f"Codex login file missing: {auth}")
    trace = attempt / "trace"
    trace.mkdir(mode=0o700)
    workspaces = attempt / "workspaces"
    registry = attempt / "ideas"
    argv = sys.argv[:]
    with tempfile.TemporaryDirectory(prefix="tiger2-exp1-codex-") as temporary:
        home = Path(temporary)
        shutil.copyfile(auth, home / "auth.json")
        (home / "auth.json").chmod(0o600)
        os.environ.update(CODEX_HOME=str(home), NEURICO_IDEAS=str(registry),
                          NEURICO_WORKSPACE=str(workspaces), NEURICO_RESEARCH_TRACE_DIR=str(trace))
        idea_id = IdeaManager(registry).submit_idea(read(attempt / "idea.json"))
        install_retention_hooks()
        archive = InternalTraceArchive(
            trace, private_trace_redactor(os.environ, credential_file=home / "auth.json"),
        )
        class RunnerOutput:
            tail = ""

            def write(self, text):
                self.tail = (self.tail + text[-8192:])[-8192:]
                return sys.__stdout__.write(text)

            def flush(self):
                sys.__stdout__.flush()

        output = RunnerOutput()
        sys.argv = ["runner.py", idea_id, "--provider", "codex", "--no-github",
                    "--no-write-paper", "--timeout", str(config["research_timeout_seconds"]),
                    "--resource-finder-timeout", str(config["resource_timeout_seconds"])]
        if config.get("codex_model"):
            sys.argv += ["--model", config["codex_model"]]
        succeeded = False
        try:
            with redirect_stdout(output):
                runner_main()
            if "SUCCESS! Research execution completed." not in output.tail:
                raise RuntimeError("NeuriCo did not report successful completion.")
            succeeded = True
        finally:
            sys.argv = argv
            archive.provider(home)
            archive.retained_records()
            reports = [path for path in workspaces.glob("*/REPORT.md") if path.stat().st_size]
            if len(reports) != 1:
                archive.gap("workspace", "Expected exactly one nonempty REPORT.md")
            if not any(path for folder in ("sessions", "archived_sessions")
                       for path in (trace / "provider" / folder).rglob("*.jsonl")):
                archive.gap("provider", "No native Codex session JSONL was captured")
            archive.finalize(run_id=attempt.name,
                             run_configuration={**config, "neurico_revision": revision,
                                                "workflow": "standard", "paper": False},
                             outcome="completed" if succeeded else "failed",
                             return_code=0 if succeeded else 1)
    if archive.gaps:
        raise RuntimeError("Expected one nonempty REPORT.md and a trace archive without recorded gaps.")
    write(attempt / "result.json",
          dict(report=str(reports[0].relative_to(attempt)), trace="trace/manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("generate", "score", "execute", "_research"):
        stages.add_parser(stage).add_argument("run", type=Path)
    stages.choices["generate"].add_argument("--query", type=Path, required=True)
    stages.choices["generate"].add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    stages.choices["generate"].add_argument("--adapter", type=Path, required=True)
    stages.choices["execute"].add_argument("--rank", type=int, choices=(1, 2, 3), required=True)
    stages.choices["_research"].add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    run = args.run.resolve()
    if args.stage == "_research":
        research(args)
    elif args.stage == "execute":
        execute(args, run)
    else:
        run.mkdir(parents=True, exist_ok=True)
        with (run / ".inputs.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            {"generate": generate, "score": score}[args.stage](args, run)


if __name__ == "__main__":
    main()
