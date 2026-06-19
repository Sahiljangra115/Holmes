"""The image-manifest split must be deterministic and the three splits disjoint.
We synthesise a tiny REAL/FAKE tree so the test needs no real dataset."""
from PIL import Image

from src.data import prepare_images


def _make_tree(root, n=40):
    for cls in ("REAL", "FAKE"):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (8, 8), (i, i, i)).save(d / f"{i}.png")


def test_split_deterministic_and_disjoint(tmp_path):
    root = tmp_path / "cifake"
    _make_tree(root)
    out = tmp_path / "m.parquet"

    a = prepare_images.build_manifest(root, out=out)
    b = prepare_images.build_manifest(root, out=out)
    assert a.sort_values("path")["split"].tolist() == b.sort_values("path")["split"].tolist()
    paths = {s: set(a[a["split"] == s]["path"]) for s in ("train", "val", "test")}
    assert paths["train"].isdisjoint(paths["val"])
    assert paths["train"].isdisjoint(paths["test"])
    assert paths["val"].isdisjoint(paths["test"])
