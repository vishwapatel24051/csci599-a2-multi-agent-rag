"""Chat-model factory for the two generation providers.

The router never names a model directly; it picks a *slot* ("nvidia:strong"), and the slot
resolves to whatever model ID the environment configures. Swapping a model is a .env change.
"""

from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings, get_settings
from src.llm.pricing import PRICES

Provider = Literal["openai", "nvidia"]
Tier = Literal["fast", "strong"]


@dataclass(frozen=True)
class ModelSlot:
    key: str  # e.g. "nvidia:strong"
    provider: Provider
    tier: Tier
    model_id: str

    @property
    def context_window(self) -> int | None:
        p = PRICES.get(self.model_id)
        return p.context_window if p else None


def model_slots(s: Settings | None = None) -> dict[str, ModelSlot]:
    s = s or get_settings()
    slots = [
        ModelSlot("openai:fast", "openai", "fast", s.openai_fast_model),
        ModelSlot("openai:strong", "openai", "strong", s.openai_strong_model),
        ModelSlot("nvidia:fast", "nvidia", "fast", s.nvidia_fast_model),
        ModelSlot("nvidia:strong", "nvidia", "strong", s.nvidia_strong_model),
    ]
    return {slot.key: slot for slot in slots}


def make_chat_model(slot: ModelSlot, *, max_tokens: int = 2048, timeout_s: float = 90,
                    reasoning: bool = False) -> BaseChatModel:
    """Build a LangChain chat model for a slot. Retries are left to the router (it falls back across providers).

    `reasoning` toggles Nemotron's thinking mode. Measured on the Phase 0 benchmark: with thinking on, Lightning
    spent 559 of 639 output tokens on hidden reasoning (6.3 s); off, 0.9 s and 0.23 s to first streamed token.
    """
    s = get_settings()
    if slot.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=slot.model_id,
            api_key=s.openai_api_key.get_secret_value(),
            max_tokens=max_tokens,
            timeout=timeout_s,
            max_retries=1,
            stream_usage=True,  # token counts on streamed responses → cost accounting
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(
        model=slot.model_id,
        api_key=s.nvidia_api_key.get_secret_value(),
        base_url=s.nvidia_base_url,
        max_completion_tokens=max_tokens,  # ChatNVIDIA silently defaults to 1024
        model_kwargs={"chat_template_kwargs": {"enable_thinking": reasoning}},
    )
