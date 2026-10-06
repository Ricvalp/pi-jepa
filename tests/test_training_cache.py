"""Derived caches preserve crop/action/solver semantics and truth boundaries."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pi_jepa.data import CAMERA, CART_LIMIT, SCHEMA_VERSION, TrainDataset, validate_clocks
from pi_jepa.initial_conditions import FixedInitialConditions
from pi_jepa.losses import simulate_window
from pi_jepa.train import batch_from, load_config
from pi_jepa.training_cache import (LearningCache, FixedTargetCache,
    prepare_learning_cache, prepare_fixed_target_cache)


@pytest.fixture(params=["passive", "controlled"])
def corpus(tmp_path, request):
    torch.set_num_threads(2)
    kind = request.param
    root = tmp_path / "data" / kind
    root.mkdir(parents=True)
    config = load_config()
    manifest = {"schema_version": SCHEMA_VERSION, "dataset": kind,
        "physics": config["physics"], "collection_settings": config["data"],
        "camera": CAMERA, "cart_limit_m": CART_LIMIT, "clocks": validate_clocks(config),
        "train": [], "validation": []}
    rng = np.random.default_rng(914)
    for split, lengths in [("train", [83, 75]), ("validation", [81])]:
        (root / split).mkdir()
        for index, length in enumerate(lengths):
            rgb = rng.integers(0, 256, (length, 8, 8, 3), dtype=np.uint8)
            # Nonexact float32 values exercise public action/parameter rounding.
            force = (rng.uniform(-.7, .9, (length - 1, 1)) if kind == "controlled"
                     else np.zeros((length - 1, 1)))
            theta = np.array([.987654321, .231234567])
            relative = f"{split}/episode-{index}.npz"
            np.savez_compressed(root / relative, rgb=rgb, force=force, theta=theta)
            manifest[split].append({"path": relative, "truth": f"truth/{split}/{index}.npz",
                "reset_mode": index, "apparatus_id": str(index), "valid_length": length})
    (root / "manifest.json").write_text(json.dumps(manifest))
    table = FixedInitialConditions(torch.tensor([[0., .13, 1.87, -.29],
        [0., -.07, -.93, .19]], dtype=torch.float64), [0, 1])
    metadata = {"mode": "true_fixed", "split": "train", "source_sha256": "test-reset-digest"}
    return root, tmp_path / "cache", table, metadata


def test_learning_cache_matches_ragged_crops_and_preserves_disk(corpus, monkeypatch):
    root, cache_root, _, _ = corpus
    original_load = np.load
    accesses = []

    def learning_only(path, **kwargs):
        assert "truth" not in Path(path).parts
        accesses.append(Path(path))
        return original_load(path, **kwargs)

    monkeypatch.setattr(np, "load", learning_only)
    path = prepare_learning_cache(root, cache_root)
    assert prepare_learning_cache(root, cache_root) == path
    plain = TrainDataset(root, cache_size=0)
    mapped = TrainDataset(root, cache_size=0, cache_root=cache_root)
    before = len(accesses)
    ids, starts = torch.tensor([1, 0]), [3, 17]
    actual = batch_from(mapped, ids, "cpu", starts=starts)
    cache_accesses = accesses[before:]
    assert cache_accesses and all(path.suffix == ".npy" for path in cache_accesses)
    expected = batch_from(plain, ids, "cpu", starts=starts)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key
    item = mapped[0]
    unchanged = item["frames"].clone()
    item["frames"].zero_()
    assert torch.equal(mapped[0]["frames"], unchanged)
    assert len(TrainDataset(root, "validation", cache_root=cache_root)) == 1


def test_learning_cache_rejects_changed_source_and_incomplete_files(corpus):
    root, cache_root, _, _ = corpus
    path = prepare_learning_cache(root, cache_root)
    index = json.loads((path / "index.json").read_text())
    force_file = path / index["episodes"]["train"][0]["files"]["forces"]
    original = force_file.read_bytes()
    force_file.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="incomplete"):
        LearningCache(root, "train", cache_root)
    force_file.write_bytes(original)
    source = root / "train/episode-0.npz"
    source.write_bytes(source.read_bytes() + b"source changed")
    with pytest.raises(ValueError, match="identity differs"):
        LearningCache(root, "train", cache_root)
    with pytest.raises(ValueError, match="identity differs"):
        prepare_learning_cache(root, cache_root)


def test_fixed_targets_match_float64_solver_without_private_truth(corpus, monkeypatch):
    root, cache_root, table, metadata = corpus
    original = np.load

    class LearningArchive:
        def __init__(self, archive):
            self.archive = archive
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __getitem__(self, key):
            assert key in ("force", "theta")
            return self.archive[key]

    def no_truth_or_images(path, **kwargs):
        path = Path(path)
        assert "truth" not in path.parts
        value = original(path, **kwargs)
        return LearningArchive(value) if path.suffix == ".npz" else value

    with monkeypatch.context() as guarded:
        guarded.setattr(np, "load", no_truth_or_images)
        cache_path = prepare_fixed_target_cache(root, cache_root, table, metadata, batch_size=2)
        assert prepare_fixed_target_cache(root, cache_root, table, metadata) == cache_path
        cache = FixedTargetCache(root, cache_root, table, metadata)
    data = TrainDataset(root)
    ids = torch.tensor([1, 0])
    batch = batch_from(data, ids, "cpu", starts=[3, 17])
    initial = table(ids)
    expected = simulate_window(initial, batch["theta"], batch["prefix_forces"], batch["raw_endpoints"])
    actual = cache.gather(ids, batch["raw_endpoints"], device="cpu")
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float64 and not actual.requires_grad
    # Ragged storage contains only actual episode states; no post-exit padding.
    assert cache.offsets.tolist() == [0, 83, 158]
    assert torch.equal(cache.gather(torch.tensor([0, 1]), torch.tensor([[0], [0]]))[:, 0], table.states)
    with pytest.raises(ValueError, match="extrapolate"):
        cache.gather(torch.tensor([1]), torch.tensor([[75]]))
    with pytest.raises(ValueError, match="training split"):
        cache.gather(torch.tensor([2]), torch.tensor([[0]]))
    assert not list(table.parameters())


def test_fixed_cache_rejects_reset_solver_and_file_changes(corpus, monkeypatch):
    from pi_jepa import physics
    root, cache_root, table, metadata = corpus
    path = prepare_fixed_target_cache(root, cache_root, table, metadata)
    changed = FixedInitialConditions(table.states + .001, table.reset_modes)
    with pytest.raises(ValueError, match="identity differs"):
        FixedTargetCache(root, cache_root, changed, metadata)
    with monkeypatch.context() as patch:
        patch.setattr(physics, "ELL", .86)
        with pytest.raises(ValueError, match="identity differs"):
            FixedTargetCache(root, cache_root, table, metadata)
    with (path / "states.npy").open("ab") as stream:
        stream.write(b"modified cache")
    with pytest.raises(ValueError, match="incomplete or changed"):
        FixedTargetCache(root, cache_root, table, metadata)


def test_caches_are_explicit_and_outside_production_data(corpus):
    from pi_jepa.train import InitialConditions
    root, cache_root, _, metadata = corpus
    with pytest.raises(FileNotFoundError, match="Prepare"):
        TrainDataset(root, cache_root=cache_root)
    with pytest.raises(ValueError, match="outside"):
        prepare_learning_cache(root, root / "derived")
    with pytest.raises(ValueError, match="fixed true"):
        prepare_fixed_target_cache(root, cache_root, InitialConditions([0, 1]), metadata)
    with pytest.raises(ValueError, match="splits only"):
        prepare_learning_cache(root, cache_root, splits=("test",))


def test_interrupted_preparation_does_not_publish_partial_cache(corpus, monkeypatch):
    root, cache_root, _, _ = corpus
    def fail_save(*args, **kwargs):
        raise OSError("synthetic write failure")
    monkeypatch.setattr(np, "save", fail_save)
    with pytest.raises(OSError, match="synthetic write failure"):
        prepare_learning_cache(root, cache_root)
    assert not (cache_root / root.name / "learning").exists()
    assert not list((cache_root / root.name).glob(".learning-*"))
