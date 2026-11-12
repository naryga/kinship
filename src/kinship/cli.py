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
