"""Metadata reader degrades gracefully on a missing file (absence != fake)."""
from src.models.metadata_provenance import extract_metadata


def test_missing_file_returns_unknown_not_fake(tmp_path):
    out = extract_metadata(str(tmp_path / "does_not_exist.jpg"))
    assert out["has_c2pa"] is False
    assert out["camera_make"] is None
    assert isinstance(out["inconsistencies"], list)
