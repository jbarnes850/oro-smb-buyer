"""Economically selected, day-packed full-buyer SFT with the official Cookbook trainer.

Data: a corpus directory from ``scripts/build_sft_corpus.py`` (top-reward frontier
TRAIN episodes, day-packed public actions; see ``smb_buyer/packed_sft.py``). Parent:
frozen Qwen3.6-35B-A3B (256K), fresh LoRA rank 32 and fresh AdamW. One pass, linear
LR decay, held-out NLL on the corpus's held-out episodes. The only change to the
official runner is a state-only final save (no duplicate sampler export); samplers
are created ephemerally from the saved state. Evaluate the result with
``scripts/campaign.py eval --policy state:<path> --backbone qwen3.6``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import chz
import numpy as np
import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

from smb_buyer import paths, storage
from smb_buyer.backbones import BACKBONES
from smb_buyer.packed_sft import SFT_BACKBONE

MODEL, _, RENDERERS = BACKBONES[SFT_BACKBONE]


def datum(ids, weights):
    ids = np.asarray(ids, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float32)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(ids[:-1].tolist()),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(ids[1:]),
            "weights": tinker.TensorData.from_numpy(w[1:]),
        },
    )


class PackedDataset(SupervisedDataset):
    def __init__(self, data, split, batch_size, seed, fraction=1.0, max_segments=None):
        data = Path(data)
        self.tokens = np.load(data / f"{split}-tokens.npy", mmap_mode="r")
        self.weights = np.load(data / f"{split}-weights.npy", mmap_mode="r")
        offsets = json.loads((data / f"{split}-offsets.json").read_text())
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(offsets))
        keep = int(round(len(order) * fraction))
        if max_segments is not None:
            keep = min(keep, max_segments)
        self.offsets = [offsets[i] for i in order[:keep]]
        self.batch_size = batch_size

    def segment(self, index):
        start, length = self.offsets[index][:2]
        return datum(self.tokens[start : start + length], self.weights[start : start + length])

    def get_batch(self, index):
        lo = index * self.batch_size
        return [self.segment(i) for i in range(lo, min(lo + self.batch_size, len(self.offsets)))]

    def __len__(self):
        return len(self.offsets) // self.batch_size

    def set_epoch(self, seed=0):
        if seed:
            raise ValueError("One declared pass only")


@chz.chz
class PackedBuilder(SupervisedDatasetBuilder):
    data: str
    batch_size: int = 8
    seed: int = 20261114
    fraction: float = 1.0

    def __call__(self):
        return PackedDataset(self.data, "train", self.batch_size, self.seed, self.fraction), None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="corpus directory from scripts/build_sft_corpus.py")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--heldout-segments", type=int, default=48)
    parser.add_argument("--owned-checkpoints", default=None,
                        help="optional owned-checkpoint budget registry; the final save waits until it fits")
    parser.add_argument("--dry-run", action="store_true",
                        help="load the corpus, build the config and the first batch, then exit; no Tinker calls")
    args = parser.parse_args()
    data = Path(args.data).resolve()
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("backbone", SFT_BACKBONE) != SFT_BACKBONE:
        sys.exit(f"corpus was rendered for {manifest['backbone']}, not {SFT_BACKBONE}")
    if not args.dry_run:
        paths.require_tinker_key()
    heldout = PackedDataset(data, "heldout", 1, 7, max_segments=args.heldout_segments)
    held_data = [heldout.segment(i) for i in range(len(heldout.offsets))]
    config = train.Config(
        log_path=args.log_path,
        model_name=MODEL,
        recipe_name="full-buyer-econ-packed-sft-v1",
        renderer_name=RENDERERS["on"],
        load_checkpoint_path=None,
        dataset_builder=PackedBuilder(data=str(data), batch_size=args.batch_size, fraction=args.fraction),
        learning_rate=args.learning_rate,
        lr_schedule="linear",
        num_epochs=1,
        lora_rank=32,
        save_every=0,
        rolling_save_every=0,
        eval_every=args.eval_every,
        evaluator_builders=[lambda: NLLEvaluator(held_data, name="heldout")],
        submit_ahead=1,
    )
    if args.dry_run:
        dataset, _ = config.dataset_builder()
        first = dataset.get_batch(0)
        print(json.dumps({
            "model": MODEL,
            "train_batches": len(dataset),
            "first_batch_datums": len(first),
            "first_batch_tokens": sum(d.model_input.length for d in first),
            "heldout_segments": len(held_data),
            "corpus_totals": manifest["totals"],
        }, indent=1))
        return
    Path(args.log_path).mkdir(parents=True, exist_ok=False)
    (Path(args.log_path) / "launch.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "data": str(data),
                "data_manifest": manifest["totals"],
                "args": vars(args),
                "parent": "frozen base, fresh LoRA rank 32, fresh AdamW",
                "final_save": "state only",
            },
            indent=1,
        )
    )
    checkpoint_utils.CheckpointManager = storage.state_only_final(
        checkpoint_utils, backbone=SFT_BACKBONE, owned_registry=args.owned_checkpoints
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    try:
        main()
    except paths.ConfigError as error:
        sys.exit(f"configuration: {error}")
