"""Command-line entry point. Subcommands are added by later issues."""

import typer

from kinship import __version__

app = typer.Typer(help="Kinship command-line tools.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Kinship command-line tools."""


@app.command()
def version() -> None:
    """Print the Kinship version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
