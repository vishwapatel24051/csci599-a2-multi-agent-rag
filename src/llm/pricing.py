"""Price table: the single source of truth for cost accounting and the Data agent's llm_models table.

OpenAI: standard (non-batch) rates from https://developers.openai.com/api/docs/pricing, checked 2026-10-26.
NVIDIA: hosted NIM endpoints on build.nvidia.com are free on the developer tier (trial terms,
rate-limited to ~40 requests/min per account), so their marginal cost is $0.
"""

from dataclasses import dataclass

PRICES_CHECKED_ON = "2026-10-26"


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    model_id: str
    tier: str  # "fast" | "strong" | "flagship" | "embedding"
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cached_input_usd_per_mtok: float | None
    context_window: int | None
    source: str


OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
NVIDIA_CATALOG_URL = "https://build.nvidia.com/nvidia"

PRICES: dict[str, ModelPrice] = {p.model_id: p for p in [
    ModelPrice("openai", "gpt-6-luna", "fast", 0.10, 0.50, 0.01, 1_050_000, OPENAI_PRICING_URL),
    ModelPrice("openai", "gpt-6.1-terra", "mid", 1.00, 5.00, 0.05, 1_050_000, OPENAI_PRICING_URL),
    ModelPrice("openai", "gpt-6.1-sol", "strong", 2.00, 10.00, 0.10, 1_050_000, OPENAI_PRICING_URL),
    ModelPrice("openai", "gpt-6-astra", "flagship", 10.00, 50.00, 1.00, 1_050_000, OPENAI_PRICING_URL),
    ModelPrice("openai", "text-embedding-3-small", "embedding", 0.02, 0.0, None, 8_191, OPENAI_PRICING_URL),
    ModelPrice("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", "strong", 0.0, 0.0, None, 1_000_000, NVIDIA_CATALOG_URL),
    ModelPrice("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b", "fast", 0.0, 0.0, None, None, NVIDIA_CATALOG_URL),
]}


def cost_usd(model_id: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
    """Cost of one call; unknown models count as $0 but are flagged by the caller's trace."""
    p = PRICES.get(model_id)
    if p is None:
        return 0.0
    cached_rate = p.cached_input_usd_per_mtok if p.cached_input_usd_per_mtok is not None else p.input_usd_per_mtok
    uncached = max(input_tokens - cached_input_tokens, 0)
    return (uncached * p.input_usd_per_mtok + cached_input_tokens * cached_rate + output_tokens * p.output_usd_per_mtok) / 1e6
