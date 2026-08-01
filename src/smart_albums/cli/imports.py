"""Lazy import helpers — trigger stage and client registration on demand."""

from __future__ import annotations


def import_all_stages() -> None:
    """Import all stage modules to trigger @stage registration."""
    import smart_albums.nodes.retrieve.by_year  # noqa: F401
    import smart_albums.nodes.retrieve.source  # noqa: F401
    import smart_albums.nodes.filter.videos  # noqa: F401
    import smart_albums.nodes.filter.min_score  # noqa: F401
    import smart_albums.nodes.filter.sensitive  # noqa: F401
    import smart_albums.nodes.filter.smart_search  # noqa: F401
    import smart_albums.nodes.filter.on_this_day  # noqa: F401
    import smart_albums.nodes.filter.people  # noqa: F401
    import smart_albums.nodes.filter.random  # noqa: F401
    import smart_albums.nodes.filter.none  # noqa: F401
    import smart_albums.nodes.partition.time  # noqa: F401
    import smart_albums.nodes.partition.time_gps  # noqa: F401
    import smart_albums.nodes.partition.phash  # noqa: F401
    import smart_albums.nodes.partition.faiss  # noqa: F401
    import smart_albums.nodes.partition.cosine  # noqa: F401
    import smart_albums.nodes.dedup.phash  # noqa: F401
    import smart_albums.nodes.select.best  # noqa: F401
    import smart_albums.nodes.select.balanced  # noqa: F401
    import smart_albums.nodes.select.diverse  # noqa: F401
    import smart_albums.nodes.select.avoid_repeat  # noqa: F401
    import smart_albums.nodes.analyze.score  # noqa: F401
    import smart_albums.nodes.analyze.embedding  # noqa: F401
    import smart_albums.nodes.publish.create_album  # noqa: F401
    import smart_albums.nodes.publish.replace_album  # noqa: F401
    import smart_albums.nodes.merge  # noqa: F401
    import smart_albums.nodes.fork  # noqa: F401


def import_clients() -> None:
    """Import client modules to trigger ProtocolsRegistry registration."""
    import smart_albums.clients  # noqa: F401
    import smart_albums.core.cache  # noqa: F401
    import smart_albums.core.no_cache  # noqa: F401
