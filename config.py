import streamlit as st

PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
INDEX_NAME = st.secrets.get("INDEX_NAME", "erp-embeddings")
DIMENSION = int(st.secrets.get("DIMENSION", 384))
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
