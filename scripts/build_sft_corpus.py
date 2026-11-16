"""Build the day-packed, economically selected SFT corpus from archived frontier runs.

Zero model calls (tokenizer and renderer only). See ``smb_buyer/packed_sft.py`` for
the packing and masking rules. The corpus is rendered for Qwen3.6-35B-A3B, the
backbone of the reported warm start.

  --selection  JSON ``{"archives": [{trial_id, model, task_id, reward, path}]}``; ``path``
               is an archived run ``*.tar.gz`` that contains model_transcript.jsonl with
               its model_request records. TRAIN businesses (b500-b503) only.
  --holdout    JSON list of trial ids kept out of training for held-out NLL.

    python scripts/build_sft_corpus.py --selection selected-archives.json \\
        --holdout holdout.json --out runs/sft-corpus/packed-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smb_buyer import packed_sft, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--prompt", default=str(paths.PROMPT), help="system prompt rendered into every segment")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=131072)
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text())
    for row in selection["archives"]:
        if row["task_id"][3:7] not in ("b500", "b501", "b502", "b503"):
            raise SystemExit(f"{row['trial_id']} is not a TRAIN-business archive ({row['task_id']})")
    manifest = packed_sft.build_parallel(
        args.selection,
        args.out,
        args.prompt,
        holdout=set(json.loads(Path(args.holdout).read_text())),
        workers=args.workers,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(manifest["totals"], indent=1))


if __name__ == "__main__":
    main()
