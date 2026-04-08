import os

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "hf")  # "hf" | "openai"
STRATEGY: str = os.getenv("STRATEGY", "semantic")      # "semantic" | "keyword" | "hybrid"

HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
HF_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL: str = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/pipeline/feature-extraction"

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
EMBEDDING_TIMEOUT_S: float = float(os.getenv("EMBEDDING_TIMEOUT_S", "5.0"))
