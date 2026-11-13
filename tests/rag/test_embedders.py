"""The fake embedder, the OpenAI embedder against a stub client, and the factory.

Nothing here touches the network or needs an API key.
"""

import math
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest

from kinship.config import Settings
from kinship.rag.embedder import (
    Embedder,
    EmbedderConfigError,
    EmbedderError,
    FakeEmbedder,
    OpenAIEmbedder,
    get_embedder,
)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def settings(**values: Any) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


# -- fake ----------------------------------------------------------------------------


def test_fake_embedder_is_deterministic_and_normalised() -> None:
    one, two = FakeEmbedder(), FakeEmbedder()
    texts = ["Led the data platform team", "Écrit en français", ""]
    assert one.embed(texts) == two.embed(texts)
    assert one.model == "fake-hash-v1" and one.dim == 1536
    for vector in one.embed(texts):
        assert len(vector) == 1536
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)
    assert isinstance(one, Embedder)


def test_fake_embedder_values_are_stable_across_processes() -> None:
    # Pinned: a change here means stored fake vectors and eval baselines moved.
    vector = FakeEmbedder(dim=32).embed(["data pipelines"])[0]
    nonzero = {i: round(x, 6) for i, x in enumerate(vector) if x}
    assert nonzero == {4: 0.666667, 9: -0.333333, 20: -0.666667}
    assert vector == FakeEmbedder(dim=32).embed_query("Data   PIPELINES!")


def test_fake_embedder_empty_text_is_a_fixed_unit_vector() -> None:
    fake = FakeEmbedder(dim=8)
    assert fake.embed(["", "   ", "the of and"]) == [[1.0, 0, 0, 0, 0, 0, 0, 0]] * 3


def test_fake_embedder_similarity_follows_shared_words() -> None:
    fake = FakeEmbedder()
    query, close, closer, far = fake.embed(
        [
            "kubernetes platform engineering",
            "She ran platform engineering for the payments group",
            "Deep kubernetes platform engineering experience on our clusters",
            "Organised the annual charity bake sale and raffle",
        ]
    )
    assert cosine(query, closer) > cosine(query, close) > cosine(query, far)
    assert cosine(query, close) > 0.2
    assert abs(cosine(query, far)) < 0.1


# -- OpenAI with a stub client ---------------------------------------------------------


class StubClient:
    """Stands in for `openai.OpenAI`: records calls, fails on cue, returns shuffled data."""

    def __init__(self, dim: int = 4, failures: list[Exception] | None = None) -> None:
        self.dim = dim
        self.failures = list(failures or [])
        self.calls: list[dict[str, Any]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def vector(self, text: str) -> list[float]:
        return [float(len(text)), *([0.0] * (self.dim - 1))]

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        data = [
            SimpleNamespace(index=i, embedding=self.vector(text))
            for i, text in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=list(reversed(data)))  # order must come from `index`


def _response(status: int, headers: dict[str, str] | None = None) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.openai.invalid/v1/embeddings")
    return httpx2.Response(status, headers=headers, request=request)


def rate_limit(headers: dict[str, str] | None = None, code: str | None = None) -> Exception:
    body = {"code": code} if code else None
    return openai.RateLimitError("slow down", response=_response(429, headers), body=body)


def make(client: StubClient, **kwargs: Any) -> tuple[OpenAIEmbedder, list[float]]:
    sleeps: list[float] = []
    embedder = OpenAIEmbedder(
        client=client, dim=client.dim, sleep=sleeps.append, jitter=lambda: 0.0, **kwargs
    )
    return embedder, sleeps


def test_openai_embedder_needs_a_key() -> None:
    with pytest.raises(EmbedderConfigError, match="OPENAI_API_KEY"):
        OpenAIEmbedder("")


def test_openai_batches_by_count_and_keeps_order() -> None:
    client = StubClient()
    embedder, sleeps = make(client, batch_size=3)
    texts = [f"text {'x' * i}" for i in range(8)]

    vectors = embedder.embed(texts)

    assert [len(call["input"]) for call in client.calls] == [3, 3, 2]
    assert [text for call in client.calls for text in call["input"]] == texts
    assert vectors == [client.vector(text) for text in texts]
    assert client.calls[0]["model"] == "text-embedding-3-small"
    assert client.calls[0]["dimensions"] == 4
    assert client.calls[0]["encoding_format"] == "float"
    assert sleeps == []
    assert embedder.embed([]) == [] and len(client.calls) == 3


def test_openai_batches_by_token_budget() -> None:
    client = StubClient()
    embedder, _ = make(client, batch_size=100, max_batch_tokens=100)
    texts = ["a" * 160, "b" * 160, "c" * 160, "d" * 1000, "e" * 40]  # 40, 40, 40, 250, 10 tokens

    embedder.embed(texts)

    assert [[t[0] for t in call["input"]] for call in client.calls] == [
        ["a", "b"],
        ["c"],
        ["d"],
        ["e"],
    ]


def test_openai_never_sends_empty_or_huge_input() -> None:
    client = StubClient()
    embedder, _ = make(client, max_input_chars=10)
    embedder.embed(["", "x" * 50])
    assert client.calls[0]["input"] == [" ", "x" * 10]


def test_openai_default_dimension_matches_the_column() -> None:
    client = StubClient(dim=1536)
    embedder = OpenAIEmbedder(client=client)
    assert (embedder.model, embedder.dim) == ("text-embedding-3-small", 1536)
    assert len(embedder.embed_query("hello")) == 1536


def test_openai_retries_rate_limits_with_backoff() -> None:
    client = StubClient(failures=[rate_limit(), rate_limit(), rate_limit()])
    embedder, sleeps = make(client, base_delay=1.0)

    assert embedder.embed(["a", "bb"]) == [client.vector("a"), client.vector("bb")]
    assert len(client.calls) == 4
    assert sleeps == [0.5, 1.0, 2.0]  # exponential; jitter stubbed to its minimum


def test_openai_backoff_is_jittered_and_capped() -> None:
    client = StubClient(failures=[rate_limit()] * 5)
    sleeps: list[float] = []
    embedder = OpenAIEmbedder(
        client=client, dim=4, sleep=sleeps.append, jitter=lambda: 1.0, base_delay=1.0, max_delay=5.0
    )
    embedder.embed(["a"])
    assert sleeps == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_openai_honours_retry_after() -> None:
    client = StubClient(
        failures=[rate_limit({"retry-after": "7"}), rate_limit({"retry-after-ms": "1500"})]
    )
    embedder, sleeps = make(client)
    embedder.embed(["a"])
    assert sleeps == [7.0, 1.5]


@pytest.mark.parametrize(
    "error",
    [
        openai.InternalServerError("boom", response=_response(503), body=None),
        openai.APITimeoutError(request=_response(500).request),
        openai.APIConnectionError(message="reset", request=_response(500).request),
    ],
    ids=["5xx", "timeout", "connection"],
)
def test_openai_retries_transient_errors(error: Exception) -> None:
    client = StubClient(failures=[error])
    embedder, sleeps = make(client)
    assert embedder.embed(["a"]) == [client.vector("a")]
    assert len(sleeps) == 1


def test_openai_gives_up_after_max_retries() -> None:
    client = StubClient(failures=[rate_limit()] * 10)
    embedder, sleeps = make(client, max_retries=2)
    with pytest.raises(EmbedderError, match="after 3 attempts") as caught:
        embedder.embed(["a"])
    assert isinstance(caught.value.__cause__, openai.RateLimitError)
    assert len(client.calls) == 3 and len(sleeps) == 2


@pytest.mark.parametrize(
    "error",
    [
        openai.AuthenticationError("bad key", response=_response(401), body=None),
        openai.BadRequestError("bad input", response=_response(400), body=None),
        rate_limit(code="insufficient_quota"),
    ],
    ids=["auth", "bad-request", "out-of-quota"],
)
def test_openai_does_not_retry_permanent_errors(error: Exception) -> None:
    client = StubClient(failures=[error])
    embedder, sleeps = make(client)
    with pytest.raises(EmbedderError, match="OpenAI embeddings request failed"):
        embedder.embed(["a"])
    assert len(client.calls) == 1 and sleeps == []


def test_openai_rejects_a_wrong_sized_answer() -> None:
    client = StubClient(dim=4)
    embedder = OpenAIEmbedder(client=client, dim=8)
    with pytest.raises(EmbedderError, match="length 4"):
        embedder.embed(["a"])


# -- factory ---------------------------------------------------------------------------


def test_get_embedder_defaults_to_openai_and_needs_a_key() -> None:
    with pytest.raises(EmbedderConfigError, match="OPENAI_API_KEY"):
        get_embedder(settings(openai_api_key="", kinship_embedder=""))
    with pytest.raises(EmbedderConfigError, match="OPENAI_API_KEY"):
        get_embedder(settings(openai_api_key="", kinship_embedder="fake"), "openai")


def test_get_embedder_builds_openai_when_a_key_is_set() -> None:
    embedder = get_embedder(settings(openai_api_key="not-a-real-key"), batch_size=7)
    assert isinstance(embedder, OpenAIEmbedder)
    assert (embedder.model, embedder.dim, embedder.batch_size) == (
        "text-embedding-3-small",
        1536,
        7,
    )


def test_get_embedder_fake_only_by_name() -> None:
    assert isinstance(get_embedder(settings(kinship_embedder="fake")), FakeEmbedder)
    assert isinstance(get_embedder(settings(openai_api_key="k"), "FAKE"), FakeEmbedder)
    with pytest.raises(EmbedderConfigError, match="Unknown embedder"):
        get_embedder(settings(), "voyage")


def test_settings_read_kinship_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KINSHIP_EMBEDDER", "fake")
    assert settings().kinship_embedder == "fake"
