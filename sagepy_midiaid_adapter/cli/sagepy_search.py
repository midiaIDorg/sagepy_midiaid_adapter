from pathlib import Path

import click


@click.command(context_settings={"show_default": True})
@click.argument("precursor_stats", type=Path)
@click.argument("fragment_stats", type=Path)
@click.argument("edges", type=Path)
@click.argument("config", type=Path)
def sagepy_search(
    precursor_stats: Path,
    fragment_stats: Path,
    edges: Path,
    config: Path,
) -> None:
    """Run sagepy search"""
    raise NotImplementedError
