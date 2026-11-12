"""One isolated, persistent retrieval process per approach for an experiment."""

import contextlib
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def execute(request):
    system = request["system"]
    project = {"baseline": "pdf-rag-baseline", "okf": "pdf-to-okf-auto", "hybrid": "hybrid-rag"}[system]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / project))
    started = time.perf_counter()
    import chromadb
    from src.embeddings.embedder import Embedder
    from retrieval_shared.pdf_chunking import chunk_pages
    from retrieval_shared.pdf_loader import load_pdf

    embedder = Embedder(model_name=request["embedding_model"])
    client = chromadb.PersistentClient(path=request["index_directory"])
    collections = {}
    kinds = ["pdf"] if system == "baseline" else ["okf"] if system == "okf" else ["pdf", "okf"]
    for kind in kinds:
        name = "experiment_" + kind
        if name in [c.name for c in client.list_collections()]:
            raise ValueError("Experiment indexes must be new and empty")
        collections[kind] = client.create_collection(
            name, metadata={"hnsw:space": "cosine"}, embedding_function=None,
        )
    initialization_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dimensions = request["dimensions"]
    pdf_fingerprint = hashlib.sha256()

    def store(kind, embedded):
        ids = [getattr(r, "record_id", getattr(r, "chunk_id", None)) for r in embedded]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate index record IDs")
        if any(len(r.embedding) != dimensions for r in embedded):
            raise ValueError("Embedding dimensions differ from configuration")
        if kind == "pdf":
            for record_id, record in zip(ids, embedded):
                pdf_fingerprint.update(json.dumps([record_id, record.text, record.metadata.get("page_numbers")]).encode())
        for offset in range(0, len(embedded), 100):
            batch = embedded[offset:offset + 100]
            collections[kind].upsert(
                ids=ids[offset:offset + 100], documents=[r.text for r in batch],
                embeddings=[list(r.embedding) for r in batch],
                metadatas=[{**r.metadata, "source_type": kind} for r in batch],
            )

    if "pdf" in kinds:
        for path in sorted(Path(request["pdf_directory"]).rglob("*.pdf")):
            chunks = chunk_pages(load_pdf(path), chunk_size=request["chunk_size"], overlap=request["chunk_overlap"])
            store("pdf", embedder.embed_chunks(chunks) if system == "baseline" else embedder.embed_pdf_chunks(chunks))
    if "okf" in kinds:
        from src.ingestion.okf_loader import load_okf_directory
        concepts = load_okf_directory(request["okf_directory"])
        store("okf", embedder.embed_okf_concepts(concepts))
    counts = {kind: collection.count() for kind, collection in collections.items()}
    if not all(counts.values()):
        raise ValueError("Empty experiment collection")
    index_seconds = time.perf_counter() - started

    def retrieve(question):
        start = time.perf_counter()
        vector = list(embedder.embed_query(question))
        if len(vector) != dimensions:
            raise ValueError("Query embedding dimensions differ from configuration")
        embedded_at = time.perf_counter()
        rankings = {}
        for kind, collection in collections.items():
            raw = collection.query(query_embeddings=[vector],
                                   n_results=min(request["top_k"], counts[kind]),
                                   include=["documents", "metadatas", "distances"])
            rankings[kind] = [dict(record_id=i, text=t, metadata=m, distance=d)
                              for i, t, m, d in zip(raw["ids"][0], raw["documents"][0],
                                                    raw["metadatas"][0], raw["distances"][0])]
        searched_at = time.perf_counter()
        if system == "hybrid":
            from src.fusion.ranker import reciprocal_rank_fusion
            from src.retrieval.vector_store import RetrievedRecord
            fused = reciprocal_rank_fusion(
                [RetrievedRecord(**r) for r in rankings["pdf"]],
                [RetrievedRecord(**r) for r in rankings["okf"]], rrf_k=request["rrf_k"],
            )
            records = [asdict(r) for r in fused[:request["top_k"]]]
        else:
            records = rankings[kinds[0]]
        finished = time.perf_counter()
        return records, dict(embedding=embedded_at-start, search=searched_at-embedded_at,
                             fusion=finished-searched_at, total=finished-start)

    started = time.perf_counter()
    retrieve(request["questions"][0]["question"])
    warmup_seconds = time.perf_counter() - started
    records = []
    for question in request["questions"]:
        runs = [retrieve(question["question"]) for _ in range(request["repetitions"])]
        records.append(dict(question_id=question["id"], question_type=question["type"],
                            arm=request["arm"], results=runs[0][0],
                            timings=[r[1] for r in runs],
                            repeat_rankings=[[r["record_id"] for r in result] for result, _ in runs]))
    return dict(status="success", counts=counts, initialization_seconds=initialization_seconds,
                pdf_chunk_sha256=pdf_fingerprint.hexdigest() if "pdf" in kinds else None,
                index_seconds=index_seconds, warmup_seconds=warmup_seconds, retrieval_records=records,
                dependencies={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                python=sys.version)


if __name__ == "__main__":
    request = json.load(sys.stdin)
    # Library progress must not corrupt the JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        response = execute(request)
    print(json.dumps(response))
