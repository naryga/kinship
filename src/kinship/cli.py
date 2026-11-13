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


@app.command()
def embed(
    owner: Annotated[str, typer.Option(help="Owner id, or the owner's exact display name.")],
    include_messages: Annotated[
        bool, typer.Option("--include-messages", help="Also chunk and embed message bodies.")
    ] = False,
    purge_messages: Annotated[
        bool, typer.Option("--purge-messages", help="Delete this owner's message chunks.")
    ] = False,
    embedder: Annotated[
        str | None,
        typer.Option(help="openai or fake (test vectors). Default: KINSHIP_EMBEDDER, else openai."),
    ] = None,
    batch_size: Annotated[int, typer.Option(min=1, help="Chunks embedded per request.")] = 64,
) -> None:
    """Chunk an owner's data and embed the chunks that are new or changed."""
    from kinship.config import get_settings
    from kinship.db.engine import get_sessionmaker
    from kinship.rag.embedder import EmbedderError, embedder_name, get_embedder
    from kinship.rag.indexer import OwnerNotFoundError, resolve_owner_id, sync_chunks

    def fail(message: str) -> typer.Exit:
        typer.echo(f"Error: {message}", err=True)
        return typer.Exit(1)

    if include_messages and purge_messages:
        raise fail("--include-messages and --purge-messages cannot be used together.")
    settings = get_settings()
    try:
        if embedder is None and embedder_name(settings) == "fake":
            # Fake vectors in a real database are useless, so an environment
            # variable alone is not enough to select them.
            raise fail("KINSHIP_EMBEDDER=fake: pass --embedder fake to confirm test vectors.")
        chosen = get_embedder(settings, embedder, batch_size=batch_size)
    except EmbedderError as error:
        raise fail(str(error)) from error

    session = get_sessionmaker()()
    try:
        owner_id = resolve_owner_id(session, owner)
        summary = sync_chunks(
            session,
            owner_id,
            chosen,
            include_messages=include_messages,
            purge_messages=purge_messages,
            batch_size=batch_size,
            # Commit as we go: a failure late in a long run keeps the vectors already paid for.
            after_batch=session.commit,
        )
        session.commit()
    except (OwnerNotFoundError, EmbedderError) as error:
        session.rollback()
        raise fail(str(error)) from error
    finally:
        session.close()
    typer.echo(summary.format())


if __name__ == "__main__":
    app()
