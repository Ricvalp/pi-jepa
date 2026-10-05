"""Prepare immutable derived learning/target caches before submitting training."""
import argparse
from pathlib import Path

import torch

from pi_jepa.initial_conditions import load_fixed_training_initial_conditions
from pi_jepa.training_cache import prepare_learning_cache, prepare_fixed_target_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Parent containing passive/controlled datasets")
    parser.add_argument("--cache-root", required=True, help="Separate parent directory for derived caches")
    parser.add_argument("--dataset", choices=("passive", "controlled", "both"), default="both")
    parser.add_argument("--fixed-targets", action="store_true",
                        help="Also compute targets for the explicit fixed true training-reset diagnostic")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64, help="Bounded simulator preparation batch")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.batch_size < 1:
        parser.error("CPU threads and simulator batch size must be positive")
    torch.set_num_threads(args.cpu_threads)
    datasets = ("passive", "controlled") if args.dataset == "both" else (args.dataset,)
    for dataset in datasets:
        root = Path(args.data_root) / dataset
        print(f"Preparing {dataset} learning cache ...", flush=True)
        path = prepare_learning_cache(root, args.cache_root)
        print(f"Learning cache ready: {path}", flush=True)
        if args.fixed_targets:
            table, metadata = load_fixed_training_initial_conditions(root)
            print(f"Preparing {dataset} fixed-reset targets ...", flush=True)
            path = prepare_fixed_target_cache(root, args.cache_root, table, metadata, args.batch_size)
            print(f"Fixed-reset target cache ready: {path}", flush=True)


if __name__ == "__main__":
    main()
