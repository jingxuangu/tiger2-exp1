#!/usr/bin/env bash
set -euo pipefail

exp_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
target=${1:?Usage: prepare-policy.sh EXPORT_DIRECTORY}
scratch=${TIGER2_EXP1_SCRATCH:-/net/spaces/scratch/$USER}
verl_python=${TIGER2_EXP1_VERL_PYTHON:-$scratch/venvs/ideation-verl090-ahc-h200/bin/python}
if [[ -f "$target/lora_adapter/adapter_model.safetensors" &&
      -f "$target/lora_adapter/adapter_config.json" &&
      -f "$target/tokenizer/tokenizer_config.json" ]]; then
    exit 0
fi
uv_bin="$exp_dir/.tools/uv"
if [[ ! -x "$uv_bin" ]]; then
    uv_bin=uv
fi
source_dir=$("$uv_bin" tool run --from 'huggingface_hub>=1,<2' hf download \
    chailabideation/qwen3-8b-v7-from0-step50 \
    --revision b7c5c201ceb54a2a3aa56fac21be8279298c1431 \
    model_world_size_4_rank_{0,1,2,3}.pt huggingface/ lora_train_meta.json fsdp_config.json \
    --quiet)
uv_bin="$exp_dir/.tools/uv"
if [[ ! -x "$uv_bin" ]]; then
    uv_bin=uv
fi
source_dir=${source_dir#path=}
if [[ ! -s "$target/lora_adapter/adapter_model.safetensors" ||
      ! -s "$target/lora_adapter/adapter_config.json" ]]; then
# Reuse verl's FSDP merge and LoRA serializer. The release contains only
# adapter weights, so skip the full-model initialization and save step.
"$verl_python" -u - merge --backend fsdp --local_dir "$source_dir" --target_dir "$target" <<'PYTHON'
from verl.model_merger.base_model_merger import generate_config_from_args, parse_args
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


class AdapterMerger(FSDPModelMerger):
    def save_hf_model_and_tokenizer(self, state_dict):
        if self.save_lora_adapter(state_dict) is None:
            raise ValueError("The Terra checkpoint did not contain LoRA weights.")


merger = AdapterMerger(generate_config_from_args(parse_args()))
try:
    merger.merge_and_save()
finally:
    merger.cleanup()
PYTHON
fi
test -f "$target/lora_adapter/adapter_model.safetensors"
test -f "$target/lora_adapter/adapter_config.json"
python3 - "$source_dir/huggingface" "$target/tokenizer" <<'PYTHON'
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
destination.mkdir(parents=True, exist_ok=True)
for path in sorted(source.iterdir(), key=lambda p: (p.name == "tokenizer_config.json", p.name)):
    if path.is_file():
        target = destination / path.name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(path.read_bytes())
        temporary.replace(target)
PYTHON
