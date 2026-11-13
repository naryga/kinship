"""Embedders: text in, vectors out.

- `Embedder`: the protocol the rest of Kinship depends on.
- `OpenAIEmbedder`: the real one (`text-embedding-3-small`, 1536 dimensions),
  with batching and retries.
- `FakeEmbedder`: deterministic, offline, and good enough for retrieval tests:
  texts that share words get similar vectors.
- `get_embedder(settings)`: picks one from configuration. It never falls back
  to the fake embedder on its own.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kinship.config import Settings

OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIM = 1536
FAKE_MODEL = "fake-hash-v1"

EMBEDDER_NAMES = ("openai", "fake")


class EmbedderError(RuntimeError):
    """An embedder could not be built or could not produce vectors."""


class EmbedderConfigError(EmbedderError):
    """The embedder configuration is unusable (unknown name, missing API key)."""


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors of length `dim`, labelled with `model`."""

    model: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in the order given."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """The vector for a search query."""
        return self.embed([text])[0]


# -- fake ---------------------------------------------------------------------

_WORD = re.compile(r"\w+")
_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "were",
        "with",
        "you",
    ]
)
_BIGRAM_WEIGHT = 0.5


class FakeEmbedder(Embedder):
    """Feature-hashed bag of words (and word bigrams), L2-normalised.

    Deterministic across processes and machines (it hashes with `hashlib`, not
    the per-process `hash()`), and never touches the network.
    """

    def __init__(self, dim: int = OPENAI_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.model = FAKE_MODEL
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        words = [w for w in _WORD.findall(text.casefold()) if w not in _STOP_WORDS]
        features: defaultdict[str, float] = defaultdict(float)
        for word in words:
            features[word] += 1.0
        for first, second in itertools.pairwise(words):
            features[f"{first} {second}"] += _BIGRAM_WEIGHT

        vector = [0.0] * self.dim
        for feature, weight in features.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            vector[(value >> 1) % self.dim] += sign * weight

        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            # No usable words. A zero vector has no cosine distance (NaN in pgvector).
            vector = [0.0] * self.dim
            vector[0] = 1.0
            return vector
        return [x / norm for x in vector]


# -- OpenAI -------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    """A cheap token estimate: about four characters per token."""
    return max(1, math.ceil(len(text) / 4))


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings with batching and retry.

    Inputs are sent in batches of at most `batch_size` texts and about
    `max_batch_tokens` tokens. Rate limits (429), 5xx responses, timeouts and
    connection errors are retried up to `max_retries` times with exponential
    backoff and jitter; a `Retry-After` header is honoured. The SDK's own
    retries are switched off so that this is the only retry loop.

    `client` (anything with `.embeddings.create(...)`), `sleep` and `jitter`
    can be injected for tests.
    """

    def __init__(
        self,
        api_key: str = "",
        *,
        model: str = OPENAI_MODEL,
        dim: int = OPENAI_DIM,
        batch_size: int = 128,
        max_batch_tokens: int = 100_000,
        max_input_chars: int = 24_000,
        max_retries: int = 6,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if client is None:
            if not api_key:
                raise EmbedderConfigError(
                    "OPENAI_API_KEY is not set, so the OpenAI embedder cannot be used. "
                    "Set it in the environment or in .env."
                )
            import openai

            client = openai.OpenAI(api_key=api_key, max_retries=0)
        self._client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_input_chars = max_input_chars
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._jitter = jitter

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in self._batches([self._prepare(text) for text in texts]):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _prepare(self, text: str) -> str:
        # The API rejects empty input; over-long input is cut well below the
        # model's 8191-token limit (chunks are far shorter than this anyway).
        text = text[: self.max_input_chars]
        return text if text.strip() else " "

    def _batches(self, texts: Sequence[str]) -> Iterator[list[str]]:
        batch: list[str] = []
        tokens = 0
        for text in texts:
            size = approx_tokens(text)
            if batch and (len(batch) >= self.batch_size or tokens + size > self.max_batch_tokens):
                yield batch
                batch, tokens = [], 0
            batch.append(text)
            tokens += size
        if batch:
            yield batch

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._create_with_retry(batch)
        items = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in items]
        if len(vectors) != len(batch):
            raise EmbedderError(
                f"OpenAI returned {len(vectors)} embeddings for {len(batch)} inputs"
            )
        for vector in vectors:
            if len(vector) != self.dim:
                raise EmbedderError(
                    f"OpenAI model {self.model} returned a vector of length {len(vector)}, "
                    f"expected {self.dim}"
                )
        return vectors

    def _create_with_retry(self, batch: list[str]) -> Any:
        kwargs: dict[str, Any] = {"input": batch, "model": self.model, "encoding_format": "float"}
        if self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dim
        attempt = 0
        while True:
            try:
                return self._client.embeddings.create(**kwargs)
            except Exception as error:
                if not _is_retryable(error):
                    if _is_openai_error(error):
                        raise EmbedderError(f"OpenAI embeddings request failed: {error}") from error
                    raise
                if attempt >= self.max_retries:
                    raise EmbedderError(
                        f"OpenAI embeddings request failed after {attempt + 1} attempts: {error}"
                    ) from error
                self._sleep(self._delay(attempt, error))
                attempt += 1

    def _delay(self, attempt: int, error: Exception) -> float:
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            # Wait what the server asked for, plus a little so batches do not line up.
            return retry_after + self._jitter() * 0.25
        backoff = min(self.max_delay, self.base_delay * (2**attempt))
        return backoff * (0.5 + self._jitter() / 2)


def _is_openai_error(error: Exception) -> bool:
    import openai

    return isinstance(error, openai.OpenAIError)


def _is_retryable(error: Exception) -> bool:
    import openai

    if isinstance(error, openai.RateLimitError):
        # Out of credit is also a 429, but waiting does not fix it.
        return getattr(error, "code", None) != "insufficient_quota"
    if isinstance(error, openai.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code >= 500 or error.status_code in (408, 409)
    return False


def _retry_after_seconds(error: Exception) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            seconds = float(raw) * scale
        except ValueError:
            continue  # an HTTP date; fall back to exponential backoff
        if 0 <= seconds <= 300:
            return seconds
    return None


# -- factory ------------------------------------------------------------------


def embedder_name(settings: Settings, name: str | None = None) -> str:
    """The embedder to use: `name` if given, else `KINSHIP_EMBEDDER`, else `openai`."""
    choice = (name or settings.kinship_embedder or "openai").strip().lower()
    if choice not in EMBEDDER_NAMES:
        raise EmbedderConfigError(
            f"Unknown embedder {choice!r}; expected one of: {', '.join(EMBEDDER_NAMES)}"
        )
    return choice


def get_embedder(
    settings: Settings, name: str | None = None, *, batch_size: int | None = None
) -> Embedder:
    """Build the configured embedder.

    `openai` needs `OPENAI_API_KEY` and raises `EmbedderConfigError` without it.
    There is no fallback to `fake`: fake vectors in a real database would make
    semantic search quietly useless, so `fake` must be chosen by name.
    """
    choice = embedder_name(settings, name)
    if choice == "fake":
        return FakeEmbedder()
    if not settings.openai_api_key:
        raise EmbedderConfigError(
            "OPENAI_API_KEY is not set, so the OpenAI embedder cannot be used. "
            "Set it in the environment or in .env, or choose the fake embedder explicitly "
            "(test vectors only)."
        )
    if batch_size is None:
        return OpenAIEmbedder(settings.openai_api_key)
    return OpenAIEmbedder(settings.openai_api_key, batch_size=batch_size)
