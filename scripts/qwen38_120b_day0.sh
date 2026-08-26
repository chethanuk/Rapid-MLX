#!/usr/bin/env bash
# Prepare a Qwen3.8 120B-class checkpoint without guessing its architecture.
# Every remote write is separately gated. Run from the repository root.
set -euo pipefail

action="${1:-audit}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_repo="${SOURCE_REPO:-}"
source_revision="${SOURCE_REVISION:-main}"
work_dir="${WORK_DIR:-${TMPDIR:-/tmp}/qwen38-120b-day0}"
source_dir="${work_dir}/source"
q4_dir="${work_dir}/q4"
q8_dir="${work_dir}/q8"

require_source() {
  if [[ -z "${source_repo}" ]]; then
    echo 'SOURCE_REPO must name the published official checkpoint.' >&2
    exit 2
  fi
}

audit_config() {
  require_source
  python3 - "${source_repo}" "${source_revision}" <<'PY'
import json
import sys

from huggingface_hub import hf_hub_download

repo, revision = sys.argv[1:]
path = hf_hub_download(repo, "config.json", revision=revision)
config = json.load(open(path, encoding="utf-8"))
text = config.get("text_config", config)
architectures = set(config.get("architectures") or [])
allowed_architectures = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}
model_types = {str(config.get("model_type", "")), str(text.get("model_type", ""))}
allowed_types = {"qwen3_5", "qwen3_5_moe", "qwen3_5_text"}
layer_types = text.get("layer_types") or []
mtp_layers = int(text.get("mtp_num_hidden_layers", 0) or 0)

if not architectures or not architectures <= allowed_architectures:
    raise SystemExit(f"BLOCK: unsupported architectures={sorted(architectures)}")
if not model_types & allowed_types:
    raise SystemExit(f"BLOCK: unsupported model types={sorted(model_types)}")
if not any(layer in {"linear_attention", "mamba", "recurrent"} for layer in layer_types):
    raise SystemExit("BLOCK: checkpoint does not declare the expected hybrid layers")
if mtp_layers > 1:
    raise SystemExit(f"BLOCK: {mtp_layers} MTP layers exceed the validated chain-MTP contract")

print(json.dumps({
    "repo": repo,
    "revision": revision,
    "architectures": sorted(architectures),
    "model_types": sorted(model_types),
    "num_hidden_layers": text.get("num_hidden_layers"),
    "num_experts": text.get("num_experts", 0),
    "num_experts_per_tok": text.get("num_experts_per_tok", 0),
    "hybrid": True,
    "mtp_num_hidden_layers": mtp_layers,
}, indent=2))
PY
}

case "${action}" in
  audit)
    audit_config
    ;;
  download)
    audit_config
    mkdir -p "${work_dir}"
    python3 - "${source_repo}" "${source_revision}" "${source_dir}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, revision, target = sys.argv[1:]
print(snapshot_download(repo, revision=revision, local_dir=target))
PY
    ;;
  convert-q4)
    audit_config
    [[ -f "${source_dir}/config.json" ]] || { echo 'Run download first.' >&2; exit 2; }
    mlx_lm.convert --hf-path "${source_dir}" --mlx-path "${q4_dir}" \
      --quantize --q-bits 4 --q-group-size 64
    python3 "${repo_root}/scripts/extract_mtp_weights.py" \
      --hf-model "${source_dir}" --mlx-model "${q4_dir}" --bits 4 --group-size 64
    ;;
  convert-q8)
    audit_config
    [[ -f "${source_dir}/config.json" ]] || { echo 'Run download first.' >&2; exit 2; }
    mlx_lm.convert --hf-path "${source_dir}" --mlx-path "${q8_dir}" \
      --quantize --q-bits 8 --q-group-size 64
    python3 "${repo_root}/scripts/extract_mtp_weights.py" \
      --hf-model "${source_dir}" --mlx-model "${q8_dir}" --bits 8 --group-size 64
    ;;
  validate-q4)
    [[ -f "${q4_dir}/config.json" ]] || { echo 'Run convert-q4 first.' >&2; exit 2; }
    rapid-mlx bench "${q4_dir}" --tier smoke
    rapid-mlx bench "${q4_dir}" --num-prompts 3 --max-tokens 256 \
      --max-num-seqs 1 --long-prompt-tokens 1024
    rapid-mlx bench "${q4_dir}" --num-prompts 3 --max-tokens 256 \
      --max-num-seqs 1 --long-prompt-tokens 4096
    ;;
  panel)
    [[ -n "${MODEL_NAME:-}" ]] || { echo 'Set MODEL_NAME to the served model ID.' >&2; exit 2; }
    python3 "${repo_root}/scripts/qwen38_120b_panel.py" \
      --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}" \
      --model "${MODEL_NAME}" --output "${work_dir}/panel.jsonl"
    ;;
  publish-q4)
    [[ "${PUBLISH_WRITE:-NO}" == YES ]] || { echo 'Set PUBLISH_WRITE=YES.' >&2; exit 2; }
    [[ -n "${Q4_REPO:-}" ]] || { echo 'Set Q4_REPO.' >&2; exit 2; }
    [[ -f "${q4_dir}/config.json" ]] || { echo 'Run convert-q4 first.' >&2; exit 2; }
    hf upload "${Q4_REPO}" "${q4_dir}" . \
      --commit-message "Convert ${source_repo}@${source_revision} to MLX q4/group-64"
    ;;
  publish-q8)
    [[ "${PUBLISH_WRITE:-NO}" == YES ]] || { echo 'Set PUBLISH_WRITE=YES.' >&2; exit 2; }
    [[ -n "${Q8_REPO:-}" ]] || { echo 'Set Q8_REPO.' >&2; exit 2; }
    [[ -f "${q8_dir}/config.json" ]] || { echo 'Run convert-q8 first.' >&2; exit 2; }
    hf upload "${Q8_REPO}" "${q8_dir}" . \
      --commit-message "Convert ${source_repo}@${source_revision} to MLX q8/group-64"
    ;;
  mirror-q4)
    [[ "${MIRROR_WRITE:-NO}" == YES ]] || { echo 'Set MIRROR_WRITE=YES.' >&2; exit 2; }
    [[ -n "${Q4_REPO:-}" ]] || { echo 'Set Q4_REPO.' >&2; exit 2; }
    python3 "${repo_root}/scripts/mirror_to_r2.py" "${Q4_REPO}"
    ;;
  alias-candidate)
    [[ -n "${Q4_REPO:-}" ]] || { echo 'Set Q4_REPO.' >&2; exit 2; }
    [[ -f "${q4_dir}/config.json" ]] || { echo 'Run convert-q4 first.' >&2; exit 2; }
    [[ -n "${TOOL_CALL_PARSER:-}" ]] || {
      echo 'Set TOOL_CALL_PARSER only after inspecting the published tokenizer template.' >&2
      exit 2
    }
    python3 - "${q4_dir}/config.json" "${Q4_REPO}" "${TOOL_CALL_PARSER}" <<'PY'
import json
import sys

config_path, repo, parser = sys.argv[1:]
config = json.load(open(config_path, encoding="utf-8"))
text = config.get("text_config", config)
entry = {
    "hf_path": repo,
    "tool_call_parser": parser,
    "reasoning_parser": "qwen3",
    "is_hybrid": True,
    "supports_spec_decode": False,
    "is_moe": bool(text.get("num_experts", 0)),
}
if int(text.get("mtp_num_hidden_layers", 0) or 0) == 1:
    entry.update(mtp_draft_model=repo, mtp_speculative_tokens=3)
print(json.dumps({"qwen3.8-120b-4bit": entry}, indent=2))
PY
    ;;
  serve-q4)
    [[ -f "${q4_dir}/config.json" ]] || { echo 'Run convert-q4 first.' >&2; exit 2; }
    exec rapid-mlx serve "${q4_dir}" --host 127.0.0.1 \
      --port "${PORT:-8000}" --enable-auto-tool-choice \
      --tool-call-parser "${TOOL_CALL_PARSER:?set TOOL_CALL_PARSER after template audit}" \
      --reasoning-parser qwen3
    ;;
  *)
    echo "Unknown action: ${action}" >&2
    echo 'Actions: audit download convert-q4 convert-q8 validate-q4 panel publish-q4 publish-q8 mirror-q4 alias-candidate serve-q4' >&2
    exit 2
    ;;
esac
