"""
Tests for CourseSearchTool.execute() in search_tools.py.

Three bugs under investigation:
  Bug 2 – ChromaDB 1.x rejects None metadata values; CourseChunk.lesson_number is
           Optional[int], so chunks without a lesson assignment can't be stored.
  Bug 3 – Requesting n_results=5 when the collection holds <5 documents triggers a
           ChromaDB error; the try/except catches it but returns an error string
           instead of content.
"""

import pytest
from search_tools import CourseSearchTool


class TestCourseSearchToolExecute:

    # ── happy-path ─────────────────────────────────────────────────────────────

    def test_execute_returns_formatted_results(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(query="Python variables")

        assert isinstance(result, str)
        assert "Introduction to Python" in result

    def test_execute_result_contains_lesson_header(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(query="Python variables")

        # Formatted header format is "[Course - Lesson N]"
        assert "Lesson" in result

    def test_execute_populates_last_sources(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)
        assert tool.last_sources == []

        tool.execute(query="Python variables")

        assert len(tool.last_sources) > 0

    def test_execute_sources_have_label_and_url_keys(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)
        tool.execute(query="Python variables")

        for source in tool.last_sources:
            assert "label" in source
            assert "url" in source

    # ── empty collection ───────────────────────────────────────────────────────

    def test_execute_empty_collection_returns_no_content_message(self, vector_store):
        tool = CourseSearchTool(vector_store)
        result = tool.execute(query="Python variables")

        assert "No relevant content found" in result

    def test_execute_empty_collection_last_sources_stays_empty(self, vector_store):
        tool = CourseSearchTool(vector_store)
        tool.execute(query="Python variables")

        assert tool.last_sources == []

    # ── Bug 2: None lesson_number in ChromaDB metadata ─────────────────────────

    def test_execute_with_none_lesson_number_does_not_raise(self, vector_store):
        """
        Bug 2 (fixed) – CourseChunk.lesson_number=None must not crash ChromaDB.
        The fix omits the lesson_number key from metadata when the value is None.
        """
        from models import Course, Lesson, CourseChunk

        course = Course(
            title="Test Course",
            lessons=[Lesson(lesson_number=1, title="Lesson 1")],
        )
        vector_store.add_course_metadata(course)

        chunk_with_null_lesson = CourseChunk(
            content="Content without a lesson number",
            course_title="Test Course",
            lesson_number=None,
            chunk_index=0,
        )

        # After fix: no exception — None lesson_number is simply omitted from metadata
        vector_store.add_course_content([chunk_with_null_lesson])
        assert vector_store.course_content.count() == 1

    # ── Bug 3: n_results > collection size ────────────────────────────────────

    def test_execute_n_results_exceeds_collection_count_does_not_crash(
        self, populated_vector_store
    ):
        """
        Bug 3 – max_results=5 but only 2 documents exist.
        ChromaDB may raise a ValueError; VectorStore.search() catches it and
        returns a SearchResults error.  The tool must always return a str.
        """
        tool = CourseSearchTool(populated_vector_store)
        # collection has 2 docs; max_results=5
        result = tool.execute(query="Python")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_n_results_exceeded_returns_results_or_clean_error(
        self, populated_vector_store
    ):
        """When n_results > doc count the tool returns either real content or a
        graceful error message — never a raw traceback or empty string."""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(query="Python")

        has_content = "Introduction to Python" in result
        has_clean_error = "Search error:" in result or "No relevant content" in result
        assert has_content or has_clean_error, (
            f"Expected results or clean error, got: {result!r}"
        )

    # ── course_name filter ────────────────────────────────────────────────────

    def test_execute_filters_by_course_name(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(
            query="programming concepts", course_name="Introduction to Python"
        )

        assert "Introduction to Python" in result

    def test_execute_unknown_course_name_returns_no_course_found(
        self, populated_vector_store
    ):
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(query="Python", course_name="Nonexistent Course XYZ")

        assert "No course found" in result

    # ── lesson_number filter ──────────────────────────────────────────────────

    def test_execute_filters_by_lesson_number(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(query="Python", lesson_number=1)

        # Lesson 1 chunk contains "variables"; lesson 2 chunk contains "control flow"
        assert "Lesson 1" in result

    def test_execute_lesson_filter_excludes_other_lessons(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)
        result_l1 = tool.execute(query="Python", lesson_number=1)
        result_l2 = tool.execute(query="Python", lesson_number=2)

        # Results from different lesson filters should differ
        assert result_l1 != result_l2
