"""Command-line entry point. Subcommands are added by later issues."""

from pathlib import Path
from typing import Annotated

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


@app.command()
def synth(
    out_dir: Annotated[Path, typer.Argument(help="Directory to write the export into.")],
    seed: Annotated[int, typer.Option(help="Seed; the same seed gives the same export.")] = 42,
    version: Annotated[
        int, typer.Option(min=1, max=2, help="1 = first export, 2 = a later export.")
    ] = 1,
    variant: Annotated[str, typer.Option(help="Export variant: basic or complete.")] = "basic",
    as_zip: Annotated[bool, typer.Option("--zip", help="Write a .zip archive.")] = False,
    connections: Annotated[
        int | None, typer.Option(help="Number of connections in export version 1.")
    ] = None,
    no_messages: Annotated[
        bool, typer.Option("--no-messages", help="Write messages.csv as a placeholder file.")
    ] = False,
) -> None:
    """Generate a synthetic (entirely fake) LinkedIn data export."""
    from kinship.synth import DEFAULT_CONNECTIONS, generate_export

    if variant not in ("basic", "complete"):
        raise typer.BadParameter("must be 'basic' or 'complete'", param_hint="--variant")
    try:
        path = generate_export(
            out_dir,
            seed=seed,
            version=version,
            variant=variant,  # type: ignore[arg-type]
            n_connections=DEFAULT_CONNECTIONS if connections is None else connections,
            as_zip=as_zip,
            include_messages=not no_messages,
        )
    except (FileExistsError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Wrote synthetic {variant} export v{version} (seed {seed}) to {path}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Address to bind."),
    port: int = typer.Option(8000, help="Port to listen on."),
) -> None:
    """Run the MCP server (Streamable HTTP at /mcp, health check at /healthz)."""
    import uvicorn

    from kinship.server.app import create_app

    uvicorn.run(create_app(host=host), host=host, port=port)


if __name__ == "__main__":
    app()
