"""LLM 适配器 - 桥接 ReuleauxCoder-ezcode 与 dogcode-memory 的 LLM 接口。

该模块提供轻量封装，将 reuleauxcoder.services.llm.client.LLM
包装成 dogcode-memory 可识别的 LLMClient / EmbeddingProvider。

"""

from __future__ import annotations

from typing import Any, Protocol


class _LLMResponseLike(Protocol):
    """只要返回对象有 .content 属性即可。"""

    content: str


class _ReuleauxLLM(Protocol):
    """ReuleauxCoder LLM 的精简协议，用于类型标注。"""

    model: str

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token: Any | None = None,
        **kwargs: Any,
    ) -> _LLMResponseLike:
        ...


class ReuleauxLLMAdapter:
    """
    ReuleauxCoder LLM 适配器。

    将 reuleauxcoder 的 `LLM.chat(messages=..., tools=...)` 接口
    包装为 dogcode-memory `LLMClient` 协议兼容的形态。

    Usage:
        from reuleauxcoder.services.llm.client import LLM
        from dogcode_memory.adapter import ReuleauxLLMAdapter

        rc_llm = LLM(model="gpt-4o", api_key="...")
        adapter = ReuleauxLLMAdapter(rc_llm)

        pipeline = MemoryPipeline.create(
            storage_dir="~/.dogcode/memories",
            llm=adapter,                # 提取 + 去重决策
        )
    """

    def __init__(
        self,
        llm: _ReuleauxLLM,
        *,
        model_override: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        """
        初始化适配器。

        Args:
            llm: reuleauxcoder 的 LLM 实例
            model_override: 可选的模型覆盖，用于提取/去重时切换至轻量模型
            max_tokens: 最大输出 token 数
            temperature: 温度参数
        """
        self._llm = llm
        self._model_override = model_override
        self._max_tokens = max_tokens
        self._temperature = temperature

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> _LLMResponseLike:
        """
        发送聊天请求，返回包含 .content 属性的响应对象。

        符合 dogcode-memory 的 LLMClient 协议。

        Args:
            messages: 消息字典列表，格式 [{"role": "...", "content": "..."}, ...]
            **kwargs: 透传参数（被忽略，保持协议兼容）

        Returns:
            响应对象，可通过 .content 获取文本内容
        """
        # reuleauxcoder 的 LLM.chat 接受 list[dict]，与 messages 兼容
        # 移除 dogcode-memory 可能传入但 reuleauxcoder 不认识的参数
        rc_kwargs: dict[str, Any] = {}

        # 若设置了模型覆盖，临时切换（仅在 llm 支持 reconfigure 时）
        original_model: str | None = None
        if self._model_override and hasattr(self._llm, "model"):
            original_model = self._llm.model
            if hasattr(self._llm, "reconfigure"):
                self._llm.reconfigure(
                    model=self._model_override,
                    api_key=getattr(self._llm, "api_key", ""),
                    base_url=getattr(self._llm, "base_url", None),
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            else:
                # 不支持热切换，回退到原模型
                pass

        try:
            response = self._llm.chat(
                messages=messages,  # type: ignore[arg-type]
                **rc_kwargs,
            )
        finally:
            # 恢复原始模型
            if original_model and hasattr(self._llm, "reconfigure"):
                self._llm.reconfigure(
                    model=original_model,
                    api_key=getattr(self._llm, "api_key", ""),
                    base_url=getattr(self._llm, "base_url", None),
                    temperature=getattr(self._llm, "temperature", 0.0),
                    max_tokens=getattr(self._llm, "max_tokens", 4096),
                )

        return response

    @property
    def underlying(self) -> _ReuleauxLLM:
        """获取被包装的原始 LLM 实例。"""
        return self._llm


class ReuleauxEmbeddingAdapter:
    """
    ReuleauxCoder Embedding 适配器。

    如果 reuleauxcoder 配置了支持 Embedding 的 Provider，
    可以通过该适配器将其包装为 dogcode-memory 的 EmbeddingProvider。

    Usage:
        from reuleauxcoder.services.llm.client import LLM
        from dogcode_memory.adapter import ReuleauxEmbeddingAdapter

        rc_llm = LLM(model="text-embedding-3-small", api_key="...")
        embed_adapter = ReuleauxEmbeddingAdapter(rc_llm)

        pipeline = MemoryPipeline.create(
            storage_dir="~/.dogcode/memories",
            embedding_provider=embed_adapter,
        )
    """

    def __init__(
        self,
        llm: _ReuleauxLLM,
        *,
        model_override: str | None = None,
    ):
        """
        初始化 Embedding 适配器。

        Args:
            llm: reuleauxcoder 的 LLM 实例
            model_override: 可选的 Embedding 模型覆盖
        """
        self._llm = llm
        self._model_override = model_override

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量文本向量化。

        符合 dogcode-memory 的 EmbeddingProvider 协议。

        Args:
            texts: 待向量化的文本列表

        Returns:
            向量列表，每个向量是 float 列表
        """
        # 检查底层是否直接支持 embed 方法（OpenAI client 有 embeddings.create）
        client = getattr(self._llm, "client", None)
        if client is None:
            raise RuntimeError(
                "ReuleauxEmbeddingAdapter: underlying LLM has no .client attribute. "
                "Ensure the LLM is initialized with a provider that supports embeddings."
            )

        if not hasattr(client, "embeddings"):
            raise RuntimeError(
                "ReuleauxEmbeddingAdapter: underlying client does not support embeddings API. "
                "Provider may need to be an OpenAI-compatible endpoint with /v1/embeddings."
            )

        model = self._model_override or getattr(self._llm, "model", "text-embedding-3-small")
        results: list[list[float]] = []

        for text in texts:
            try:
                resp = client.embeddings.create(
                    model=model,
                    input=text,
                )
                # OpenAI embeddings response: data[0].embedding
                vector = resp.data[0].embedding
                results.append(vector)
            except Exception as e:
                raise RuntimeError(f"Embedding failed for text (len={len(text)}): {e}") from e

        return results

    @property
    def underlying(self) -> _ReuleauxLLM:
        """获取被包装的原始 LLM 实例。"""
        return self._llm


def adapt_llm(
    llm: Any,
    *,
    extraction_model: str | None = None,
    embedding_model: str | None = None,
) -> tuple[Any, Any]:
    """
    一键适配工厂函数。

    同时生成 LLM 适配器和 Embedding 适配器，便于直接传入 MemoryPipeline.create()。

    Args:
        llm: reuleauxcoder 的 LLM 实例
        extraction_model: 记忆提取时使用的轻量模型（如 "gpt-4o-mini"）
        embedding_model: Embedding 模型（如 "text-embedding-3-small"）

    Returns:
        (llm_adapter, embedding_adapter) 元组，可直接解包传入 MemoryPipeline.create()

    Example:
        from reuleauxcoder.services.llm.client import LLM
        from dogcode_memory.adapter import adapt_llm
        from dogcode_memory import MemoryPipeline

        rc_llm = LLM(model="gpt-4o", api_key="...")
        llm_adapter, embed_adapter = adapt_llm(
            rc_llm,
            extraction_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        )

        pipeline = MemoryPipeline.create(
            storage_dir="~/.dogcode/memories",
            llm=llm_adapter,
            embedding_provider=embed_adapter,
        )
    """
    llm_adapter = ReuleauxLLMAdapter(
        llm,
        model_override=extraction_model,
    )
    embedding_adapter = ReuleauxEmbeddingAdapter(
        llm,
        model_override=embedding_model,
    ) if embedding_model else None
    return llm_adapter, embedding_adapter
