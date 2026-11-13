.PHONY: eval test lint

# Retrieval quality (recall@5, MRR) over the synthetic export with the fake
# embedder. Pass options through ARGS, e.g. `make eval ARGS="--embedder openai"`.
eval:
	uv run kinship eval $(ARGS)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
