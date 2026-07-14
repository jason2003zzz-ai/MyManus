from app.web.memory_rag import MemoryRecord, SemanticMemoryStore


class FakeMemoryBackend:
    model_name = "fake-memory-embedding"

    def __init__(self):
        self.document_batches = []

    @staticmethod
    def _vector(text):
        value = text.lower()
        if any(term in value for term in ("邮件", "导师", "gmail", "回信")):
            return [1.0, 0.0, 0.0]
        if any(term in value for term in ("报告", "word", "文档")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def make_store(tmp_path, backend=None):
    return SemanticMemoryStore(
        storage_path=tmp_path / "memories.json",
        cache_path=tmp_path / "memory-embeddings.json",
        backend=backend or FakeMemoryBackend(),
        fallback_to_sparse=False,
    )


def seed_store(store):
    store.upsert(
        MemoryRecord(
            id="run-mail",
            conversation_id="thread-a",
            run_id="run-mail",
            created_at="2026-01-01T00:00:00+00:00",
            task="检查导师发来的邮件",
            answer="已经读取邮件并起草回信。",
        )
    )
    store.upsert(
        MemoryRecord(
            id="run-report",
            conversation_id="thread-b",
            run_id="run-report",
            created_at="2026-01-02T00:00:00+00:00",
            task="整理调研材料",
            answer="已经生成正式 Word 报告。",
        )
    )


def test_memory_rag_semantically_recalls_relevant_history(tmp_path):
    store = make_store(tmp_path)
    seed_store(store)

    matches = store.search("继续处理之前给导师的回信", top_k=2, min_score=0.5)

    assert matches
    assert matches[0].record.run_id == "run-mail"
    assert matches[0].retrieval_method == "embedding"
    assert matches[0].embedding_model == "fake-memory-embedding"


def test_memory_rag_persists_records_reuses_vectors_and_supports_exclusion(tmp_path):
    backend = FakeMemoryBackend()
    store = make_store(tmp_path, backend)
    seed_store(store)

    store.search("生成 Word 文档", min_score=0.5)
    matches = store.search(
        "给导师回复邮件", exclude_run_ids={"run-mail"}, min_score=0.0
    )

    assert len(backend.document_batches) == 1
    assert all(match.record.run_id != "run-mail" for match in matches)
    reloaded = make_store(tmp_path)
    assert {record.run_id for record in reloaded.load()} == {"run-mail", "run-report"}

    store.delete_run_ids({"run-mail"})
    assert {record.run_id for record in store.load()} == {"run-report"}


def test_memory_rag_skips_records_marked_ineligible_for_retrieval(tmp_path):
    store = make_store(tmp_path)
    store.upsert(
        MemoryRecord(
            id="bad-negative",
            conversation_id="thread-bad",
            run_id="bad-negative",
            created_at="2026-01-03T00:00:00+00:00",
            task="搜索公开毕业去向",
            answer="公开渠道未找到相关数据。",
            quality="unverified_negative",
            retrieval_eligible=False,
        )
    )
    store.upsert(
        MemoryRecord(
            id="verified",
            conversation_id="thread-good",
            run_id="verified",
            created_at="2026-01-04T00:00:00+00:00",
            task="搜索公开毕业去向",
            answer="已打开官网并核验毕业生页面。",
        )
    )

    matches = store.search("搜索公开毕业去向", top_k=3, min_score=0.0)

    assert {match.record.run_id for match in matches} == {"verified"}
