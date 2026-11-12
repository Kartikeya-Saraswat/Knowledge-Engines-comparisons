"""Load, index, retrieve, and answer questions over a converted OKF corpus."""

from pathlib import Path

from dotenv import load_dotenv

from src.embeddings.embedder import DEFAULT_EMBEDDING_MODEL, Embedder
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.okf_loader import load_okf_directory
from src.ingestion.fingerprints import fingerprint_concepts
from src.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OKFRAGPipeline:
    def __init__(self, persist_directory=PROJECT_ROOT / "data" / "chroma_db", *,
                 embedder=None, vector_store=None, answer_generator=None,
                 embedding_model=DEFAULT_EMBEDDING_MODEL, answer_model=None):
        load_dotenv(PROJECT_ROOT / ".env")
        self.embedder = embedder if embedder is not None else Embedder(embedding_model)
        self.vector_store = vector_store if vector_store is not None else VectorStore(
            persist_directory, model_name=self.embedder.model_name,
        )
        self._answer_generator = answer_generator
        self.answer_model = answer_model

    def index_okf_directory(self, directory: str | Path) -> int:
        fingerprint = fingerprint_concepts(directory)
        concepts = load_okf_directory(directory)
        if not concepts:
            raise ValueError(f"No OKF concepts found in {directory}")
        ids = [concept.concept_id for concept in concepts]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate OKF concept IDs")
        # Complete parsing and embedding before modifying the stored corpus.
        records = self.embedder.embed_okf_concepts(concepts)
        count = self.vector_store.replace_concepts(records)
        collection = self.vector_store.collection
        collection.modify(metadata={
            **{key: value for key, value in (collection.metadata or {}).items()
               if not key.startswith("hnsw:")},
            "corpus_fingerprint": fingerprint,
        })
        return count

    def ensure_okf_directory(self, directory: str | Path) -> int:
        """Reuse stored embeddings when the saved concept files are unchanged."""
        fingerprint = fingerprint_concepts(directory)
        collection = self.vector_store.collection
        count = collection.count()
        if count and (collection.metadata or {}).get("corpus_fingerprint") == fingerprint:
            print(f"Reusing indexed OKF concepts: {count}")
            return count
        count = self.index_okf_directory(directory)
        print(f"OKF concepts indexed: {count}")
        return count

    def retrieve(self, question: str, top_k=5):
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        return self.vector_store.search(self.embedder.embed_query(question), top_k=top_k)

    def answer_question(self, question: str, top_k=5):
        evidence = self.retrieve(question, top_k=top_k)
        if self._answer_generator is None:
            self._answer_generator = AnswerGenerator(model=self.answer_model)
        return self._answer_generator.generate(question, evidence)
