"""Explicit campaign: controlled joint/post-hoc comparison, or passive PI-JEPA."""
import argparse
import json
from pathlib import Path

from pi_jepa.data import generate
from pi_jepa.runs import run_directory, utc_stamp, write_provenance
from pi_jepa.train import add_training_arguments, config_from_arguments, same_experiment, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--run-dir", help="New campaign directory, or existing campaign with --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="Skip final held-out prediction evaluation")
    args = parser.parse_args()
    config = config_from_arguments(args)
    for key in config["paths"]:
        config["paths"][key] = str(Path(config["paths"][key]).resolve())
    generate(config)
    modes = ("joint",) if config["dataset"] == "passive" else ("joint", "jepa", "readout")
    with run_directory(config["paths"]["runs"], f"{config['dataset']}-comparison-seed{config['seed']}", args.run_dir, args.resume) as output:
        manifest_path = output / "stages.json"
        if args.resume:
            saved = json.loads((output / "config.json").read_text())
            if not same_experiment(saved, config):
                raise ValueError("Resume campaign scientific configuration differs from the original")
            stage_paths = json.loads(manifest_path.read_text())
        else:
            stage_paths = {mode: str((output / f"{mode}-seed{config['seed']}-{utc_stamp()}").resolve())
                           for mode in modes}
            manifest_path.write_text(json.dumps(stage_paths, indent=2) + "\n")
        if not config.setdefault("wandb", {}).get("group"):
            config["wandb"]["group"] = output.name
        write_provenance(output, config, extra={"stage": "comparison"}, resume=args.resume)
        if not args.quiet:
            print(f"Campaign directory: {output.resolve()}", flush=True)
        checkpoints, failures = {}, {}
        for mode in modes:
            if mode == "readout" and "jepa" not in checkpoints:
                failures[mode] = "Skipped because JEPA training failed"
                continue
            destination = Path(stage_paths[mode])
            try:
                checkpoints[mode] = train(config, mode, args.device,
                    resume=args.resume and destination.exists(), run_dir=destination,
                    pretrained=checkpoints.get("jepa") if mode == "readout" else None, quiet=args.quiet)
            except (FloatingPointError, RuntimeError) as exc:
                failures[mode] = str(exc)
                print(f"{mode} failed: {exc}", flush=True)
        (output / "completion.json").write_text(json.dumps({
            "checkpoints": {key: str(value.resolve()) for key, value in checkpoints.items()},
            "failures": failures}, indent=2) + "\n")
        if failures:
            raise SystemExit("The comparison has incomplete stages; inspect completion.json and stage logs")
        if not args.train_only:
            from pi_jepa.evaluate import run
            selected = {"passive": checkpoints["joint"]} if config["dataset"] == "passive" else {
                "joint": checkpoints["joint"], "posthoc": checkpoints["readout"]}
            run(config, args.device, checkpoints=selected)


if __name__ == "__main__":
    main()
