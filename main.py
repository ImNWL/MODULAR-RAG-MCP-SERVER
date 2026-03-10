#!/usr/bin/env python3
"""MCP Server entry point for Modular RAG system."""

import sys


def main():
    """Main entry point for the MCP server."""
    try:
        from src.core.settings import load_settings
        
        settings = load_settings()
        print(f"[INFO] Config loaded: LLM={settings.llm.provider}/{settings.llm.model}")
        print(f"[INFO] Embedding: {settings.embedding.provider}/{settings.embedding.model}")
        print(f"[INFO] Vision: {'enabled' if settings.vision_llm.enabled else 'disabled'}")
        print(f"[INFO] Rerank: {'enabled' if settings.rerank.enabled else 'disabled'}")
        print("[INFO] MCP Server starting...")
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to start: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
