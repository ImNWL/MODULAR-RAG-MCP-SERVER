"""Tests for configuration loading and validation (A3)."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.settings import (
    Settings,
    SettingsError,
    load_settings,
    validate_settings,
)


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_default_settings_file(self):
        """Should load settings from default config/settings.yaml."""
        settings = load_settings()
        
        assert isinstance(settings, Settings)
        assert settings.llm.provider
        assert settings.llm.model
        assert settings.embedding.provider
        assert settings.embedding.model
        assert settings.embedding.dimensions > 0

    def test_load_settings_from_custom_path(self):
        """Should load settings from a custom path."""
        config = {
            "llm": {"provider": "openai", "model": "gpt-4"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small", "dimensions": 1536},
            "vector_store": {"provider": "chroma"},
            "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 5, "rrf_k": 60},
            "rerank": {"enabled": False, "provider": "none"},
            "evaluation": {"enabled": False},
            "observability": {"log_level": "DEBUG"},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 100},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            settings = load_settings(temp_path)
            
            assert settings.llm.provider == "openai"
            assert settings.llm.model == "gpt-4"
            assert settings.embedding.dimensions == 1536
            assert settings.observability.log_level == "DEBUG"
            assert settings.ingestion.chunk_size == 500
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file_raises_error(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_settings("/nonexistent/path/settings.yaml")
        
        assert "not found" in str(exc_info.value).lower()

    def test_missing_llm_provider_raises_error(self):
        """Should raise SettingsError when llm.provider is missing."""
        config = {
            "llm": {"model": "gpt-4"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small", "dimensions": 1536},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(SettingsError) as exc_info:
                load_settings(temp_path)
            
            assert "llm.provider" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_missing_embedding_provider_raises_error(self):
        """Should raise SettingsError when embedding.provider is missing."""
        config = {
            "llm": {"provider": "openai", "model": "gpt-4"},
            "embedding": {"model": "text-embedding-3-small", "dimensions": 1536},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(SettingsError) as exc_info:
                load_settings(temp_path)
            
            assert "embedding.provider" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_missing_embedding_dimensions_raises_error(self):
        """Should raise SettingsError when embedding.dimensions is missing."""
        config = {
            "llm": {"provider": "openai", "model": "gpt-4"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(SettingsError) as exc_info:
                load_settings(temp_path)
            
            assert "embedding.dimensions" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_multiple_missing_fields_reported(self):
        """Should report all missing fields in error message."""
        config = {
            "embedding": {"dimensions": 1536},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(SettingsError) as exc_info:
                load_settings(temp_path)
            
            error_msg = str(exc_info.value)
            assert "llm.provider" in error_msg
            assert "llm.model" in error_msg
            assert "embedding.provider" in error_msg
            assert "embedding.model" in error_msg
        finally:
            Path(temp_path).unlink()

    def test_empty_provider_treated_as_missing(self):
        """Should treat empty string provider as missing."""
        config = {
            "llm": {"provider": "", "model": "gpt-4"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small", "dimensions": 1536},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(SettingsError) as exc_info:
                load_settings(temp_path)
            
            assert "llm.provider" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()


class TestValidateSettings:
    """Tests for validate_settings function."""

    def test_valid_settings_passes(self):
        """Should pass validation for valid settings."""
        settings = load_settings()
        validate_settings(settings)

    def test_invalid_embedding_dimensions_raises_error(self):
        """Should raise error for non-positive embedding dimensions."""
        config = {
            "llm": {"provider": "openai", "model": "gpt-4"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small", "dimensions": -1},
            "vector_store": {"provider": "chroma"},
            "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 5, "rrf_k": 60},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 100},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(SettingsError) as exc_info:
                load_settings(temp_path)
            
            assert "embedding.dimensions" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_invalid_chunk_overlap_raises_error(self):
        """Should raise error when chunk_overlap >= chunk_size."""
        config = {
            "llm": {"provider": "openai", "model": "gpt-4"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small", "dimensions": 1536},
            "vector_store": {"provider": "chroma"},
            "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 5, "rrf_k": 60},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 600},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(SettingsError) as exc_info:
                load_settings(temp_path)
            
            assert "chunk_overlap" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()


class TestSettingsDataclass:
    """Tests for Settings dataclass structure."""

    def test_settings_has_all_required_sections(self):
        """Should have all required configuration sections."""
        settings = load_settings()
        
        assert hasattr(settings, "llm")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "vision_llm")
        assert hasattr(settings, "vector_store")
        assert hasattr(settings, "retrieval")
        assert hasattr(settings, "rerank")
        assert hasattr(settings, "evaluation")
        assert hasattr(settings, "observability")
        assert hasattr(settings, "ingestion")

    def test_llm_settings_fields(self):
        """Should have correct LLM settings fields."""
        settings = load_settings()
        llm = settings.llm
        
        assert hasattr(llm, "provider")
        assert hasattr(llm, "model")
        assert hasattr(llm, "api_key")
        assert hasattr(llm, "temperature")
        assert hasattr(llm, "max_tokens")

    def test_embedding_settings_fields(self):
        """Should have correct embedding settings fields."""
        settings = load_settings()
        embed = settings.embedding
        
        assert hasattr(embed, "provider")
        assert hasattr(embed, "model")
        assert hasattr(embed, "dimensions")

    def test_ingestion_nested_settings(self):
        """Should correctly parse nested ingestion settings."""
        settings = load_settings()
        
        assert hasattr(settings.ingestion, "chunk_refiner")
        assert hasattr(settings.ingestion, "metadata_enricher")
        assert hasattr(settings.ingestion.chunk_refiner, "use_llm")
        assert hasattr(settings.ingestion.metadata_enricher, "use_llm")
