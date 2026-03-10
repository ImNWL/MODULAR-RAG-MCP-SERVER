"""Configuration loading and validation module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


class SettingsError(Exception):
    """Raised when settings validation fails."""
    pass


@dataclass
class LLMSettings:
    provider: str
    model: str
    api_key: str = ""
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    api_key: str = ""
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = ""
    base_url: str = ""


@dataclass
class VisionLLMSettings:
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = ""
    base_url: str = ""
    max_image_size: int = 2048


@dataclass
class VectorStoreSettings:
    provider: str = "chroma"
    persist_directory: str = "./data/db/chroma"
    collection_name: str = "knowledge_hub"


@dataclass
class RetrievalSettings:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    rrf_k: int = 60


@dataclass
class RerankSettings:
    enabled: bool = False
    provider: str = "none"
    model: str = ""
    top_k: int = 5


@dataclass
class EvaluationSettings:
    enabled: bool = False
    provider: str = "custom"
    metrics: list[str] = field(default_factory=lambda: ["hit_rate", "mrr"])


@dataclass
class ObservabilitySettings:
    log_level: str = "INFO"
    trace_enabled: bool = True
    trace_file: str = "./logs/traces.jsonl"
    structured_logging: bool = True


@dataclass
class ChunkRefinerSettings:
    use_llm: bool = True


@dataclass
class MetadataEnricherSettings:
    use_llm: bool = True


@dataclass
class IngestionSettings:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter: str = "recursive"
    batch_size: int = 100
    chunk_refiner: ChunkRefinerSettings = field(default_factory=ChunkRefinerSettings)
    metadata_enricher: MetadataEnricherSettings = field(default_factory=MetadataEnricherSettings)


@dataclass
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vision_llm: VisionLLMSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings


def _get_nested_value(data: dict, path: str) -> Any:
    """Get a nested value from a dict using dot notation."""
    keys = path.split(".")
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _validate_required_fields(data: dict) -> None:
    """Validate that required fields are present in raw config dict."""
    required_fields = [
        "llm.provider",
        "llm.model",
        "embedding.provider",
        "embedding.model",
        "embedding.dimensions",
    ]
    
    missing = []
    for field_path in required_fields:
        value = _get_nested_value(data, field_path)
        if value is None or value == "":
            missing.append(field_path)
    
    if missing:
        raise SettingsError(f"Missing required configuration fields: {', '.join(missing)}")


def validate_settings(settings: "Settings") -> None:
    """Validate a Settings object for required fields and constraints.
    
    This function provides centralized validation for Settings objects,
    ensuring all required fields are present and valid.
    
    Args:
        settings: The Settings object to validate
        
    Raises:
        SettingsError: If validation fails, with clear error message indicating
                      which field is missing or invalid
    """
    errors = []
    
    if not settings.llm.provider:
        errors.append("llm.provider")
    if not settings.llm.model:
        errors.append("llm.model")
    if not settings.embedding.provider:
        errors.append("embedding.provider")
    if not settings.embedding.model:
        errors.append("embedding.model")
    if settings.embedding.dimensions <= 0:
        errors.append("embedding.dimensions (must be positive)")
    
    if not settings.vector_store.provider:
        errors.append("vector_store.provider")
    
    if settings.retrieval.dense_top_k <= 0:
        errors.append("retrieval.dense_top_k (must be positive)")
    if settings.retrieval.sparse_top_k <= 0:
        errors.append("retrieval.sparse_top_k (must be positive)")
    if settings.retrieval.fusion_top_k <= 0:
        errors.append("retrieval.fusion_top_k (must be positive)")
    
    if settings.ingestion.chunk_size <= 0:
        errors.append("ingestion.chunk_size (must be positive)")
    if settings.ingestion.chunk_overlap < 0:
        errors.append("ingestion.chunk_overlap (must be non-negative)")
    if settings.ingestion.chunk_overlap >= settings.ingestion.chunk_size:
        errors.append("ingestion.chunk_overlap (must be less than chunk_size)")
    
    if errors:
        raise SettingsError(f"Invalid settings: {', '.join(errors)}")


def _parse_llm_settings(data: dict) -> LLMSettings:
    """Parse LLM settings from config dict."""
    llm_data = data.get("llm", {})
    return LLMSettings(
        provider=llm_data.get("provider", ""),
        model=llm_data.get("model", ""),
        api_key=llm_data.get("api_key", ""),
        azure_endpoint=llm_data.get("azure_endpoint", ""),
        deployment_name=llm_data.get("deployment_name", ""),
        api_version=llm_data.get("api_version", ""),
        base_url=llm_data.get("base_url", ""),
        temperature=llm_data.get("temperature", 0.0),
        max_tokens=llm_data.get("max_tokens", 4096),
    )


def _parse_embedding_settings(data: dict) -> EmbeddingSettings:
    """Parse embedding settings from config dict."""
    embed_data = data.get("embedding", {})
    return EmbeddingSettings(
        provider=embed_data.get("provider", ""),
        model=embed_data.get("model", ""),
        dimensions=embed_data.get("dimensions", 1536),
        api_key=embed_data.get("api_key", ""),
        azure_endpoint=embed_data.get("azure_endpoint", ""),
        deployment_name=embed_data.get("deployment_name", ""),
        api_version=embed_data.get("api_version", ""),
        base_url=embed_data.get("base_url", ""),
    )


def _parse_vision_settings(data: dict) -> VisionLLMSettings:
    """Parse vision LLM settings from config dict."""
    vision_data = data.get("vision_llm", {})
    return VisionLLMSettings(
        enabled=vision_data.get("enabled", False),
        provider=vision_data.get("provider", "openai"),
        model=vision_data.get("model", "gpt-4o"),
        api_key=vision_data.get("api_key", ""),
        azure_endpoint=vision_data.get("azure_endpoint", ""),
        deployment_name=vision_data.get("deployment_name", ""),
        api_version=vision_data.get("api_version", ""),
        base_url=vision_data.get("base_url", ""),
        max_image_size=vision_data.get("max_image_size", 2048),
    )


def _parse_ingestion_settings(data: dict) -> IngestionSettings:
    """Parse ingestion settings from config dict."""
    ing_data = data.get("ingestion", {})
    chunk_refiner_data = ing_data.get("chunk_refiner", {})
    metadata_enricher_data = ing_data.get("metadata_enricher", {})
    
    return IngestionSettings(
        chunk_size=ing_data.get("chunk_size", 1000),
        chunk_overlap=ing_data.get("chunk_overlap", 200),
        splitter=ing_data.get("splitter", "recursive"),
        batch_size=ing_data.get("batch_size", 100),
        chunk_refiner=ChunkRefinerSettings(
            use_llm=chunk_refiner_data.get("use_llm", True)
        ),
        metadata_enricher=MetadataEnricherSettings(
            use_llm=metadata_enricher_data.get("use_llm", True)
        ),
    )


def load_settings(path: Optional[str] = None) -> Settings:
    """Load and validate settings from YAML file.
    
    Args:
        path: Path to settings file. Defaults to config/settings.yaml
        
    Returns:
        Validated Settings object
        
    Raises:
        SettingsError: If required fields are missing or invalid
        FileNotFoundError: If settings file doesn't exist
    """
    if path is None:
        path = "config/settings.yaml"
    
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    
    with open(settings_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    _validate_required_fields(data)
    
    vector_store_data = data.get("vector_store", {})
    retrieval_data = data.get("retrieval", {})
    rerank_data = data.get("rerank", {})
    eval_data = data.get("evaluation", {})
    obs_data = data.get("observability", {})
    
    settings = Settings(
        llm=_parse_llm_settings(data),
        embedding=_parse_embedding_settings(data),
        vision_llm=_parse_vision_settings(data),
        vector_store=VectorStoreSettings(
            provider=vector_store_data.get("provider", "chroma"),
            persist_directory=vector_store_data.get("persist_directory", "./data/db/chroma"),
            collection_name=vector_store_data.get("collection_name", "knowledge_hub"),
        ),
        retrieval=RetrievalSettings(
            dense_top_k=retrieval_data.get("dense_top_k", 20),
            sparse_top_k=retrieval_data.get("sparse_top_k", 20),
            fusion_top_k=retrieval_data.get("fusion_top_k", 10),
            rrf_k=retrieval_data.get("rrf_k", 60),
        ),
        rerank=RerankSettings(
            enabled=rerank_data.get("enabled", False),
            provider=rerank_data.get("provider", "none"),
            model=rerank_data.get("model", ""),
            top_k=rerank_data.get("top_k", 5),
        ),
        evaluation=EvaluationSettings(
            enabled=eval_data.get("enabled", False),
            provider=eval_data.get("provider", "custom"),
            metrics=eval_data.get("metrics", ["hit_rate", "mrr"]),
        ),
        observability=ObservabilitySettings(
            log_level=obs_data.get("log_level", "INFO"),
            trace_enabled=obs_data.get("trace_enabled", True),
            trace_file=obs_data.get("trace_file", "./logs/traces.jsonl"),
            structured_logging=obs_data.get("structured_logging", True),
        ),
        ingestion=_parse_ingestion_settings(data),
    )
    
    validate_settings(settings)
    return settings
