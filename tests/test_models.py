"""Tests for smart_albums.core.models."""

from __future__ import annotations

from datetime import datetime

from smart_albums.core.models import Asset


class TestAsset:
    """Tests for the Asset dataclass."""

    def test_create_minimal(self):
        asset = Asset(
            id="abc-123",
            filename="photo.jpg",
            captured_at=datetime(2024, 1, 15, 10, 30, 0),
            mime_type="image/jpeg",
        )
        assert asset.id == "abc-123"
        assert asset.filename == "photo.jpg"
        assert asset.mime_type == "image/jpeg"
        assert asset.metadata == {}
        assert asset.latitude is None
        assert asset.longitude is None

    def test_create_with_metadata(self):
        asset = Asset(
            id="xyz",
            filename="img.png",
            captured_at=datetime(2024, 6, 1),
            mime_type="image/png",
            metadata={"score": 0.9, "phash": "abcdef1234567890"},
        )
        assert asset.metadata["score"] == 0.9
        assert asset.metadata["phash"] == "abcdef1234567890"

    def test_create_with_gps(self):
        asset = Asset(
            id="gps-1",
            filename="loc.jpg",
            captured_at=datetime(2024, 3, 10),
            mime_type="image/jpeg",
            latitude=48.8566,
            longitude=2.3522,
        )
        assert asset.latitude == 48.8566
        assert asset.longitude == 2.3522

    def test_metadata_is_mutable(self):
        asset = Asset(
            id="mut",
            filename="x.jpg",
            captured_at=datetime(2024, 1, 1),
            mime_type="image/jpeg",
        )
        asset.metadata["score"] = 0.75
        assert asset.metadata["score"] == 0.75
