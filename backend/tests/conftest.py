import sys
import os
from unittest.mock import MagicMock

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_tests_dir)

# Make backend modules importable (e.g. vector_store, ai_generator, models)
sys.path.insert(0, _backend_dir)
# Make tests/ importable so test files can do `from helpers import ...`
sys.path.insert(0, _tests_dir)

import pytest
import chromadb


class MockEmbeddingFunction:
    """
    Fast deterministic mock — avoids downloading SentenceTransformer during tests.
    ChromaDB 1.x requires a name() method on embedding functions.
    """

    def name(self) -> str:
        return "mock-embedding-function"

    def __call__(self, input: list) -> list:
        import hashlib
        vectors = []
        for text in input:
            digest = hashlib.sha256(text.encode()).digest()
            # Repeat digest to fill 384 dims (matching all-MiniLM-L6-v2 output size)
            extended = (digest * 12)[:384]
            vectors.append([b / 255.0 for b in extended])
        return vectors


@pytest.fixture
def ephemeral_chroma():
    """
    Fresh in-memory ChromaDB client, isolated per test.
    chromadb 1.x EphemeralClient shares its backing store within a process,
    so we enable allow_reset and call reset() after each test to guarantee isolation.
    """
    from chromadb.config import Settings
    client = chromadb.EphemeralClient(settings=Settings(allow_reset=True))
    yield client
    try:
        client.reset()
    except Exception:
        pass


@pytest.fixture
def vector_store(ephemeral_chroma):
    """
    VectorStore wired to ephemeral ChromaDB with a mock embedding function.
    Bypasses __init__ so no PersistentClient or SentenceTransformer is created.
    """
    from vector_store import VectorStore

    ef = MockEmbeddingFunction()
    vs = object.__new__(VectorStore)
    vs.max_results = 5
    vs.client = ephemeral_chroma
    vs.embedding_function = ef
    vs.course_catalog = ephemeral_chroma.get_or_create_collection(
        name="course_catalog", embedding_function=ef
    )
    vs.course_content = ephemeral_chroma.get_or_create_collection(
        name="course_content", embedding_function=ef
    )
    return vs


@pytest.fixture
def sample_course():
    from models import Course, Lesson
    return Course(
        title="Introduction to Python",
        course_link="https://example.com/python",
        instructor="Dr. Smith",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Variables and Types",
                lesson_link="https://example.com/python/1",
            ),
            Lesson(
                lesson_number=2,
                title="Control Flow",
                lesson_link="https://example.com/python/2",
            ),
        ],
    )


@pytest.fixture
def sample_chunks():
    from models import CourseChunk
    return [
        CourseChunk(
            content="Python variables store data values. Use integers, strings, and floats.",
            course_title="Introduction to Python",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Control flow in Python uses if, elif, and else to make decisions.",
            course_title="Introduction to Python",
            lesson_number=2,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def populated_vector_store(vector_store, sample_course, sample_chunks):
    """VectorStore seeded with two lessons and their chunks."""
    vector_store.add_course_metadata(sample_course)
    vector_store.add_course_content(sample_chunks)
    return vector_store


