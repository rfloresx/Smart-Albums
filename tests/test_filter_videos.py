"""Tests for smart_albums.nodes.filter.videos (RetainImages)."""

from __future__ import annotations

import pytest

from smart_albums.nodes.filter.videos import RetainImages

from conftest import make_asset, make_context


class TestRetainImages:
    """Tests for the RetainImages filter stage."""

    @pytest.mark.asyncio
    async def test_keeps_jpeg(self):
        ctx = make_context(assets=[make_asset(id="img", mime_type="image/jpeg")])
        node = RetainImages({})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1

    @pytest.mark.asyncio
    async def test_keeps_png(self):
        ctx = make_context(assets=[make_asset(id="img", mime_type="image/png")])
        node = RetainImages({})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1

    @pytest.mark.asyncio
    async def test_removes_video(self):
        ctx = make_context(assets=[make_asset(id="vid", mime_type="video/mp4")])
        node = RetainImages({})
        result = await node.run(ctx)
        assert len(result[0].assets) == 0

    @pytest.mark.asyncio
    async def test_removes_audio(self):
        ctx = make_context(assets=[make_asset(id="aud", mime_type="audio/mpeg")])
        node = RetainImages({})
        result = await node.run(ctx)
        assert len(result[0].assets) == 0

    @pytest.mark.asyncio
    async def test_mixed_keeps_only_images(self):
        assets = [
            make_asset(id="img1", mime_type="image/jpeg"),
            make_asset(id="vid1", mime_type="video/mp4"),
            make_asset(id="img2", mime_type="image/heic"),
            make_asset(id="app1", mime_type="application/pdf"),
        ]
        ctx = make_context(assets=assets)
        node = RetainImages({})
        result = await node.run(ctx)
        assert len(result[0].assets) == 2
        ids = [a.id for a in result[0].assets]
        assert "img1" in ids
        assert "img2" in ids

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        ctx = make_context(assets=[make_asset(id="img", mime_type="Image/JPEG")])
        node = RetainImages({})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1

    @pytest.mark.asyncio
    async def test_stats_excluded_count(self):
        assets = [
            make_asset(id="img", mime_type="image/jpeg"),
            make_asset(id="vid", mime_type="video/mp4"),
        ]
        ctx = make_context(assets=assets)
        node = RetainImages({})
        result = await node.run(ctx)
        assert result[0].stats["filter.non_images.excluded"] == 1

    @pytest.mark.asyncio
    async def test_empty_assets(self):
        ctx = make_context(assets=[])
        node = RetainImages({})
        result = await node.run(ctx)
        assert len(result[0].assets) == 0
        assert result[0].stats["filter.non_images.excluded"] == 0
