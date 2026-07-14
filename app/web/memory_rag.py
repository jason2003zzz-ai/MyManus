import hashlib
import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

from app.web.skill_matcher import tokenize


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    conversation_id: str
    run_id: str
    created_at: str
    task: str
    answer: str
    observations: str = ""
    status: str = "completed"
    quality: str = "completed"
    retrieval_eligible: bool = True

    def retrieval_text(self, max_chars: int = 12000) -> str:
        text = (
            f"Historical task: {self.task}\n"
            f"Historical answer: {self.answer}\n"
            f"Key observations: {self.observations}"
        )
        return text[:max_chars]


@dataclass(frozen=True)
class MemoryMatch:
    record: MemoryRecord
    score: float
    retrieval_method: str
    embedding_model: str = ""


class EmbeddingBackend(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedBackend:
    def __init__(self, model_name: str, cache_dir: Path | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "FastEmbed is required for semantic Agent Memory retrieval."
                ) from exc
            kwargs = {"model_name": self.model_name}
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                kwargs["cache_dir"] = str(self.cache_dir)
            self._model = TextEmbedding(**kwargs)
            return self._model

    def _prefix(self, text: str, kind: str) -> str:
        return f"{kind}: {text}" if "e5" in self.model_name.lower() else text

    @staticmethod
    def _vector(value) -> list[float]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [float(item) for item in value]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._get_model()
        return [
            self._vector(vector)
            for vector in model.embed([self._prefix(text, "passage") for text in texts])
        ]

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        vectors = list(model.query_embed(self._prefix(text, "query")))
        if not vectors:
            raise RuntimeError("Embedding model returned no query vector.")
        return self._vector(vectors[0])


class OpenAIEmbeddingBackend:
    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        dimensions: int | None = None,
    ):
        resolved_key = api_key or os.getenv(api_key_env, "")
        if not resolved_key:
            raise RuntimeError(
                f"Embedding API key is missing; set `{api_key_env}` or memory.api_key."
            )
        from openai import OpenAI

        self.model_name = model_name
        self.dimensions = dimensions
        kwargs = {"api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs = {"model": self.model_name, "input": list(texts)}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        items = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in items]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text])
        if not vectors:
            raise RuntimeError("Embedding API returned no query vector.")
        return vectors[0]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class SemanticMemoryStore:
    """Persistent long-term Agent Memory with dense semantic recall."""

    CACHE_VERSION = 1

    def __init__(
        self,
        *,
        storage_path: Path,
        cache_path: Path,
        provider: str = "fastembed",
        model_name: str = "BAAI/bge-small-zh-v1.5",
        model_cache_dir: Path | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        dimensions: int | None = None,
        max_records: int = 500,
        max_content_chars: int = 12000,
        fallback_to_sparse: bool = True,
        backend: EmbeddingBackend | None = None,
    ):
        self.storage_path = storage_path
        self.cache_path = cache_path
        self.provider = provider.lower().strip()
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.dimensions = dimensions
        self.max_records = max_records
        self.max_content_chars = max_content_chars
        self.fallback_to_sparse = fallback_to_sparse
        self._backend = backend
        self._lock = threading.RLock()

    def _get_backend(self) -> EmbeddingBackend:
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is not None:
                return self._backend
            if self.provider == "fastembed":
                self._backend = FastEmbedBackend(
                    self.model_name, cache_dir=self.model_cache_dir
                )
            elif self.provider in {"openai", "openai-compatible"}:
                self._backend = OpenAIEmbeddingBackend(
                    self.model_name,
                    api_key=self.api_key,
                    api_key_env=self.api_key_env,
                    base_url=self.base_url,
                    dimensions=self.dimensions,
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
            return self._backend

    def load(self) -> list[MemoryRecord]:
        if not self.storage_path.is_file():
            return []
        try:
            values = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if not isinstance(values, list):
                return []
            return [MemoryRecord(**value) for value in values if isinstance(value, dict)]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []

    def _write_records(self, records: list[MemoryRecord]) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.storage_path)

    def upsert(self, record: MemoryRecord) -> None:
        with self._lock:
            records = [item for item in self.load() if item.id != record.id]
            records.append(record)
            records.sort(key=lambda item: item.created_at, reverse=True)
            self._write_records(records[: self.max_records])

    def delete_run_ids(self, run_ids: set[str]) -> None:
        if not run_ids:
            return
        with self._lock:
            self._write_records(
                [record for record in self.load() if record.run_id not in run_ids]
            )
            if not self.cache_path.is_file():
                return
            try:
                cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            documents = cache.get("documents")
            if not isinstance(documents, dict):
                return
            cache["documents"] = {
                record_id: value
                for record_id, value in documents.items()
                if record_id not in run_ids
            }
            self._write_cache(cache)

    def _fingerprint(self, record: MemoryRecord) -> str:
        return hashlib.sha256(
            record.retrieval_text(self.max_content_chars).encode("utf-8")
        ).hexdigest()

    def _load_cache(self, backend: EmbeddingBackend) -> dict:
        empty = {
            "version": self.CACHE_VERSION,
            "model": backend.model_name,
            "documents": {},
        }
        if not self.cache_path.is_file():
            return empty
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty
        if (
            data.get("version") != self.CACHE_VERSION
            or data.get("model") != backend.model_name
            or not isinstance(data.get("documents"), dict)
        ):
            return empty
        return data

    def _write_cache(self, cache: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(cache, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)

    def _vectors(
        self, records: list[MemoryRecord], backend: EmbeddingBackend
    ) -> dict[str, list[float]]:
        with self._lock:
            cache = self._load_cache(backend)
            entries = cache["documents"]
            missing = [
                record
                for record in records
                if entries.get(record.id, {}).get("fingerprint")
                != self._fingerprint(record)
                or not entries.get(record.id, {}).get("embedding")
            ]
            if missing:
                vectors = backend.embed_documents(
                    [
                        record.retrieval_text(self.max_content_chars)
                        for record in missing
                    ]
                )
                if len(vectors) != len(missing):
                    raise RuntimeError("Embedding count does not match Memory records.")
                for record, vector in zip(missing, vectors):
                    entries[record.id] = {
                        "fingerprint": self._fingerprint(record),
                        "embedding": vector,
                    }
                active_ids = {record.id for record in self.load()}
                cache["documents"] = {
                    record_id: value
                    for record_id, value in entries.items()
                    if record_id in active_ids
                }
                self._write_cache(cache)
            return {record.id: cache["documents"][record.id]["embedding"] for record in records}

    @staticmethod
    def _sparse_search(
        query: str, records: list[MemoryRecord], top_k: int
    ) -> list[MemoryMatch]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        matches = []
        for record in records:
            record_tokens = set(tokenize(record.retrieval_text()))
            union = query_tokens | record_tokens
            score = len(query_tokens & record_tokens) / len(union) if union else 0.0
            if score:
                matches.append(
                    MemoryMatch(
                        record=record,
                        score=round(score, 4),
                        retrieval_method="keyword_fallback",
                    )
                )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:top_k]

    def search(
        self,
        query: str,
        *,
        exclude_run_ids: set[str] | None = None,
        top_k: int = 3,
        min_score: float = 0.5,
    ) -> list[MemoryMatch]:
        if not query.strip() or top_k <= 0:
            return []
        exclude_run_ids = exclude_run_ids or set()
        records = [
            record
            for record in self.load()
            if record.run_id not in exclude_run_ids and record.retrieval_eligible
        ]
        if not records:
            return []
        try:
            backend = self._get_backend()
            query_vector = backend.embed_query(query)
            vectors = self._vectors(records, backend)
            matches = [
                MemoryMatch(
                    record=record,
                    score=round(_cosine(query_vector, vectors[record.id]), 4),
                    retrieval_method="embedding",
                    embedding_model=backend.model_name,
                )
                for record in records
            ]
            matches = [match for match in matches if match.score >= min_score]
            matches.sort(
                key=lambda item: (item.score, item.record.created_at), reverse=True
            )
            return matches[:top_k]
        except Exception:
            if not self.fallback_to_sparse:
                raise
            return self._sparse_search(query, records, top_k)
