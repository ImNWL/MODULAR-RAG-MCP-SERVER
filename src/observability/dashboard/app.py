"""Modular RAG Dashboard - Main Application."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.settings import load_settings, SettingsError


st.set_page_config(
    page_title="Modular RAG Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def show_overview():
    """Display system overview page."""
    st.title("🔍 Modular RAG Dashboard")
    st.markdown("---")
    
    try:
        settings = load_settings()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("LLM Provider", settings.llm.provider.upper())
            st.caption(f"Model: {settings.llm.model}")
        
        with col2:
            st.metric("Embedding", settings.embedding.provider.upper())
            st.caption(f"Model: {settings.embedding.model}")
        
        with col3:
            st.metric("Vision", "ON" if settings.vision_llm.enabled else "OFF")
            if settings.vision_llm.enabled:
                st.caption(f"Model: {settings.vision_llm.model}")
        
        with col4:
            st.metric("Rerank", "ON" if settings.rerank.enabled else "OFF")
            if settings.rerank.enabled:
                st.caption(f"Provider: {settings.rerank.provider}")
        
        st.markdown("---")
        st.subheader("📊 Configuration Details")
        
        with st.expander("LLM Configuration", expanded=True):
            st.json({
                "provider": settings.llm.provider,
                "model": settings.llm.model,
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens,
            })
        
        with st.expander("Embedding Configuration"):
            st.json({
                "provider": settings.embedding.provider,
                "model": settings.embedding.model,
                "dimensions": settings.embedding.dimensions,
            })
        
        with st.expander("Retrieval Configuration"):
            st.json({
                "dense_top_k": settings.retrieval.dense_top_k,
                "sparse_top_k": settings.retrieval.sparse_top_k,
                "fusion_top_k": settings.retrieval.fusion_top_k,
                "rrf_k": settings.retrieval.rrf_k,
            })
        
        with st.expander("Ingestion Configuration"):
            st.json({
                "chunk_size": settings.ingestion.chunk_size,
                "chunk_overlap": settings.ingestion.chunk_overlap,
                "splitter": settings.ingestion.splitter,
                "batch_size": settings.ingestion.batch_size,
            })
        
        st.markdown("---")
        st.success("✅ 系统配置已加载，Dashboard 运行正常！")
        
    except FileNotFoundError:
        st.error("❌ 配置文件未找到！请先运行 setup 或创建 config/settings.yaml")
    except SettingsError as e:
        st.error(f"❌ 配置错误: {e}")
    except Exception as e:
        st.error(f"❌ 加载配置失败: {e}")


def main():
    """Main dashboard application."""
    with st.sidebar:
        st.title("📚 Navigation")
        page = st.radio(
            "选择页面",
            ["🏠 系统总览", "📄 数据浏览", "📥 Ingestion 管理", "🔍 Query 追踪", "📈 评估面板"],
            index=0,
        )
    
    if page == "🏠 系统总览":
        show_overview()
    elif page == "📄 数据浏览":
        st.title("📄 数据浏览器")
        st.info("🚧 功能开发中...")
    elif page == "📥 Ingestion 管理":
        st.title("📥 Ingestion 管理")
        st.info("🚧 功能开发中...")
    elif page == "🔍 Query 追踪":
        st.title("🔍 Query 追踪")
        st.info("🚧 功能开发中...")
    elif page == "📈 评估面板":
        st.title("📈 评估面板")
        st.info("🚧 功能开发中...")


if __name__ == "__main__":
    main()
