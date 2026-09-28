"""Tests for smart_albums.nodes.publish stages."""

from __future__ import annotations

import pytest

from smart_albums.nodes.publish.create_album import PublishCreateAlbum
from smart_albums.nodes.publish.replace_album import PublishReplaceAlbum
from protocols_system.protocols import AlbumSummary

from conftest import make_asset, make_context, FakeImageClient


class TestPublishCreateAlbum:
    """Tests for the PublishCreateAlbum stage."""

    @pytest.mark.asyncio
    async def test_dry_run(self):
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        client = FakeImageClient()
        ctx = make_context(
            assets=assets,
            image_client=client,
            config={"publish.create_album.name": "Best of 2024", "dry_run": True},
        )
        node = PublishCreateAlbum({"name": "Best of 2024", "dry_run": True})
        result = await node.run(ctx)
        assert result[0].stats["publish.create_album.dry_run"] is True
        assert result[0].stats["publish.create_album.would_create"] == 2
        assert len(client._created_albums) == 0

    @pytest.mark.asyncio
    async def test_creates_album(self):
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        client = FakeImageClient()
        ctx = make_context(
            assets=assets,
            image_client=client,
            config={"publish.create_album.name": "Best of 2024", "dry_run": False},
        )
        node = PublishCreateAlbum({"name": "Best of 2024", "dry_run": False})
        result = await node.run(ctx)
        assert result[0].stats["publish.create_album.created"] is True
        assert len(client._created_albums) == 1
        assert client._created_albums[0] == ("Best of 2024", ["a1", "a2"])

    @pytest.mark.asyncio
    async def test_conflict_detection(self):
        client = FakeImageClient(albums=[AlbumSummary(id="existing", name="Best of 2024")])
        assets = [make_asset(id="a1")]
        ctx = make_context(
            assets=assets,
            image_client=client,
            config={"publish.create_album.name": "Best of 2024", "dry_run": False},
        )
        node = PublishCreateAlbum({"name": "Best of 2024", "dry_run": False})
        result = await node.run(ctx)
        assert result[0].stats["publish.create_album.conflict"] is True
        assert result[0].stats["publish.create_album.created"] is False
        assert len(client._created_albums) == 0


class TestPublishReplaceAlbum:
    """Tests for the PublishReplaceAlbum stage."""

    @pytest.mark.asyncio
    async def test_dry_run(self):
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        client = FakeImageClient()
        ctx = make_context(
            assets=assets,
            image_client=client,
            config={"publish.replace_album.name": "Rotation", "dry_run": True},
        )
        node = PublishReplaceAlbum({"name": "Rotation", "dry_run": True})
        result = await node.run(ctx)
        assert result[0].stats["publish.replace_album.dry_run"] is True

    @pytest.mark.asyncio
    async def test_creates_new_album_when_not_exists(self):
        assets = [make_asset(id="a1")]
        client = FakeImageClient()
        ctx = make_context(
            assets=assets,
            image_client=client,
            config={"publish.replace_album.name": "New Album", "dry_run": False},
        )
        node = PublishReplaceAlbum({"name": "New Album", "dry_run": False})
        result = await node.run(ctx)
        assert result[0].stats["publish.replace_album.created"] is True
        assert len(client._created_albums) == 1
