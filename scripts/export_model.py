"""Export the final RL checkpoint from Tinker as a PEFT adapter and a merged Hugging Face model.

Follows the Tinker cookbook deployment path (tinker_cookbook.weights):

  1. download   the LoRA weights of a training checkpoint (tinker://<run>/weights/<name>). Tinker loads the
                checkpoint into a short training session and exports it, so the account must be in good
                standing. Needs tinker >= 0.39.1 (0.39.0 is refused by the service).
  2. adapter    convert to a PEFT adapter (adapter_config.json + adapter_model.safetensors) for vLLM or
                SGLang with the base model. Small, no base weights needed.
  3. merge      fold the adapter into Qwen/Qwen3.8-27B and write a standalone bf16 model. This downloads the
                base model (about 54 GB) and writes a model of the same size, so it needs about 120 GB free.
  4. publish    optional, uploads the merged model (or the adapter) to a private Hugging Face repo.

    export TINKER_API_KEY=...            # never pass the key on the command line
    python scripts/export_model.py --tinker-path tinker://<run>/weights/final --out exports/final \
        [--steps download,adapter,merge] [--hf-repo org/name]

The final checkpoint of the reported run is recorded in the campaign ledger; it is not part of this package.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.8-27B"
MERGE_FREE_GB = 120


def free_gb(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tinker-path", required=True, help="tinker://<run>/weights/<name> (a training checkpoint)")
    ap.add_argument("--out", required=True, help="export directory (gets raw/, peft/, merged/)")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--steps", default="download,adapter,merge")
    ap.add_argument("--hf-repo", default=None, help="private Hugging Face repo for the merged model (step 4)")
    args = ap.parse_args()
    steps = set(args.steps.split(","))
    if args.hf_repo:
        steps.add("publish")
    from tinker_cookbook import weights

    out = Path(args.out)
    raw, peft, merged = out / "raw", out / "peft", out / "merged"
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"tinker_path": args.tinker_path, "base_model": args.base_model, "steps": sorted(steps)}

    if "download" in steps:
        if not os.environ.get("TINKER_API_KEY"):
            sys.exit("TINKER_API_KEY is not set")
        weights.download(tinker_path=args.tinker_path, output_dir=str(raw))
        receipt["raw_files"] = sorted(p.name for p in raw.iterdir())
    if "adapter" in steps:
        weights.build_lora_adapter(base_model=args.base_model, adapter_path=str(raw), output_path=str(peft))
        receipt["peft_files"] = sorted(p.name for p in peft.iterdir())
    if "merge" in steps:
        gb = free_gb(merged)
        if gb < MERGE_FREE_GB:
            sys.exit(f"merge needs about {MERGE_FREE_GB} GB free for the base model and the output; {gb:.0f} GB "
                     "available. Run the merge on a larger disk or set HF_HOME to one.")
        weights.build_hf_model(base_model=args.base_model, adapter_path=str(raw), output_path=str(merged),
                               dtype="bfloat16")
        receipt["merged_files"] = sorted(p.name for p in merged.iterdir())
    if "publish" in steps:
        target = merged if merged.exists() else peft
        receipt["hf_url"] = weights.publish_to_hf_hub(model_path=str(target), repo_id=args.hf_repo, private=True)
    (out / "export-receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
