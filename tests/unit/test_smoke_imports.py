"""Smoke tests for verifying package imports.

This module provides basic import validation for all core packages
to ensure the project structure is correctly set up and importable.
"""

import pytest


class TestSmokeImports:
    """Smoke tests to verify all core packages can be imported."""

    @pytest.mark.unit
    def test_import_mcp_server(self):
        """Verify mcp_server package is importable."""
        import src.mcp_server
        assert src.mcp_server is not None

    @pytest.mark.unit
    def test_import_core(self):
        """Verify core package is importable."""
        import src.core
        assert src.core is not None

    @pytest.mark.unit
    def test_import_ingestion(self):
        """Verify ingestion package is importable."""
        import src.ingestion
        assert src.ingestion is not None

    @pytest.mark.unit
    def test_import_libs(self):
        """Verify libs package is importable."""
        import src.libs
        assert src.libs is not None

    @pytest.mark.unit
    def test_import_observability(self):
        """Verify observability package is importable."""
        import src.observability
        assert src.observability is not None

    @pytest.mark.unit
    def test_import_core_submodules(self):
        """Verify core submodules are importable."""
        from src.core import trace
        from src.core import response
        assert trace is not None
        assert response is not None

    @pytest.mark.unit
    def test_import_libs_submodules(self):
        """Verify libs submodules are importable."""
        from src.libs import llm
        from src.libs import embedding
        from src.libs import splitter
        from src.libs import vector_store
        from src.libs import reranker
        from src.libs import evaluator
        from src.libs import loader
        assert all([llm, embedding, splitter, vector_store, reranker, evaluator, loader])

    @pytest.mark.unit
    def test_import_ingestion_submodules(self):
        """Verify ingestion submodules are importable."""
        from src.ingestion import chunking
        from src.ingestion import transform
        from src.ingestion import embedding
        from src.ingestion import storage
        assert all([chunking, transform, embedding, storage])

    @pytest.mark.unit
    def test_import_observability_submodules(self):
        """Verify observability submodules are importable."""
        from src.observability import dashboard
        assert dashboard is not None
