"""Run collision, single-writer and immutable-data provenance contracts."""
import json
import re

import pytest
import torch

from pi_jepa.runs import file_digest, nonsecret, run_directory, write_provenance
from pi_jepa.train import same_experiment


def config_at(tmp_path):
    data = tmp_path / "data/controlled"
    data.mkdir(parents=True, exist_ok=True)
    (data / "manifest.json").write_text(json.dumps({"schema_version": 2, "dataset": "controlled",
        "train": [{"path": "train/a.npz", "trajectory_id": 0}], "validation": [], "test": []}))
    return {"seed": 42, "dataset": "controlled", "paths": {"data": str(data.parent)}, "wandb": {"api_key": "hidden"}}


def test_new_runs_are_named_unique_and_reject_collisions(tmp_path):
    with run_directory(tmp_path, "joint-seed42") as first:
        assert re.fullmatch(r"joint-seed42-\d{8}T\d{12}Z(?:-\d+)?", first.name)
        with pytest.raises(FileExistsError):
            with run_directory(tmp_path, "ignored", run_dir=first):
                pass
        with pytest.raises(RuntimeError, match="Another writer"):
            with run_directory(tmp_path, "ignored", run_dir=first, resume=True):
                pass
    with run_directory(tmp_path, "joint-seed42") as second:
        assert first != second
    with run_directory(tmp_path, "ignored", run_dir=first, resume=True) as reopened:
        assert reopened == first


def test_resume_requires_existing_explicit_run(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        with run_directory(tmp_path, "joint", resume=True):
            pass
    with pytest.raises(FileNotFoundError):
        with run_directory(tmp_path, "joint", tmp_path / "absent", resume=True):
            pass


def test_provenance_records_checkpoint_and_rejects_data_or_checkpoint_changes(tmp_path):
    config = config_at(tmp_path)
    checkpoint = tmp_path / "weights.pt"
    torch.save({"step": 17, "mode": "joint", "config": {"seed": 42}}, checkpoint)
    with run_directory(tmp_path, "example") as output:
        first = write_provenance(output, config, {"joint": checkpoint})
        assert first["checkpoints"]["joint"] == {"path": str(checkpoint), "sha256": file_digest(checkpoint),
            "step": 17, "mode": "joint", "seed": 42, "weights": "raw"}
        assert first["data"]["splits"]["train"]["entries"] == 1
        assert "hidden" not in (output / "config.json").read_text()
        resumed = write_provenance(output, config, {"joint": checkpoint}, resume=True)
        assert resumed["utc"] == first["utc"] and len(resumed["resumes"]) == 1
        torch.save({"step": 18, "mode": "joint"}, checkpoint)
        with pytest.raises(ValueError, match="checkpoint differs"):
            write_provenance(output, config, {"joint": checkpoint}, resume=True)
        (tmp_path / "data/controlled/manifest.json").write_text('{"train": []}')
        with pytest.raises(ValueError, match="manifest differs"):
            write_provenance(output, config, {"joint": checkpoint}, resume=True)


def test_science_comparison_allows_relocation_and_logging_changes():
    first = {"seed": 42, "paths": {"data": "/old"}, "training": {"lr": .1, "cpu_threads": 2}}
    second = {"seed": 42, "paths": {"data": "/new"}, "training": {"lr": .1, "cpu_threads": 8},
              "wandb": {"enabled": True}, "diagnostics": {"enabled": True}}
    assert same_experiment(first, second)
    second["training"]["lr"] = .2
    assert not same_experiment(first, second)
    assert nonsecret({"token": "secret", "batch_size": 4}) == {"token": "<redacted>", "batch_size": 4}
