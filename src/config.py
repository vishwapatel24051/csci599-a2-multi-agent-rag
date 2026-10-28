"""Central configuration, loaded from environment variables (and .env locally).

Every model ID lives here and in .env.example, never inline in agent code, because
provider catalogs rotate faster than this assignment's lifetime.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider 1: OpenAI (also embeddings) ---
    openai_api_key: SecretStr | None = None
    openai_fast_model: str = "gpt-6-luna"
    openai_strong_model: str = "gpt-6.1-sol"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- LLM provider 2: NVIDIA NIM (OpenAI-compatible endpoint) ---
    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_strong_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_fast_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    nvidia_rpm_limit: int = 40  # free-tier account limit; the router throttles below this

    # --- Vector store: Pinecone serverless ---
    pinecone_api_key: SecretStr | None = None
    pinecone_index: str = "agent-engineering-corpus"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"  # the only region on the Starter (free) plan
    pinecone_namespace: str = "corpus"

    # --- Real-time knowledge: Tavily Remote MCP ---
    tavily_api_key: SecretStr | None = None
    tavily_mcp_base_url: str = "https://mcp.tavily.com/mcp/"

    # --- RAG ---
    retrieval_k: int = 8  # candidates fetched from Pinecone
    context_k: int = 5  # chunks that make it into the prompt

    # --- Orchestration ---
    max_hops: int = 6  # supervisor → worker round trips per turn
    worker_timeout_s: float = 120.0  # one agent's whole run
    # One LLM call. Of 1,004 successful calls in evaluation the slowest took 51 s (NVIDIA; OpenAI max 13 s), so
    # 60 s never cut off a real answer but halves the wait before failing over when NVIDIA stalls.
    llm_call_timeout_s: float = 60.0
    routing_policy: Literal["economy", "balanced", "quality"] = "balanced"

    # --- Conversation state ---
    state_backend: Literal["memory", "sqlite", "firestore"] = "sqlite"
    sqlite_checkpoint_path: str = str(ROOT / "data" / "checkpoints.sqlite")
    firestore_project_id: str | None = None  # uses the project's "(default)" Firestore database
    firestore_collection: str = "langgraph_checkpoints"

    # --- Data agent ---
    sqlite_db_path: str = str(ROOT / "data" / "agentlab.db")

    # --- Deployment (used by scripts / docs, not at runtime) ---
    gcp_project_id: str | None = None
    gcp_region: str = "us-east4"
    cloud_run_service: str = "multi-agent-rag"

    @property
    def tavily_mcp_url(self) -> str:
        key = self.tavily_api_key.get_secret_value() if self.tavily_api_key else ""
        return f"{self.tavily_mcp_base_url}?tavilyApiKey={key}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
