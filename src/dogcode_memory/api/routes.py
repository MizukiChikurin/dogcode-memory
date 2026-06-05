"""HTTP API 路由 - 记忆模块 REST 接口。

提供前端可直接调用的记忆管理 API，独立于服务端集成路径。

路由列表：
    GET  /memories/search?q=...&limit=...&type=...
    GET  /memories/{uri}
    DELETE /memories/{uri}
    GET  /memories/stats
    GET  /injected-meta
    POST /maintenance
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable

from dogcode_memory.pipeline import MemoryPipeline

logger = logging.getLogger(__name__)

# 请求体大小限制：10MB
_MAX_CONTENT_LENGTH = 10 * 1024 * 1024


class MemoryAPIHandler(BaseHTTPRequestHandler):
    """记忆模块 HTTP 请求处理器。

    安全特性：
    - 可选的 Bearer Token 认证
    - localhost-only 模式
    - 请求体大小限制
    - 异常信息脱敏
    """

    def __init__(
        self,
        pipeline: MemoryPipeline,
        *args: Any,
        api_token: str | None = None,
        allow_localhost_only: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ):
        self.pipeline = pipeline
        self._api_token = api_token
        self._allow_localhost_only = allow_localhost_only
        self._debug = debug
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """使用结构化日志替代默认日志。"""
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        """发送 JSON 响应。"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))

    def _send_error(self, message: str, status: int = 400) -> None:
        """发送错误响应。

        生产模式下不暴露详细异常信息，避免泄露内部路径或实现细节。
        """
        if self._debug or status < 500:
            # 客户端错误或调试模式下，返回原始信息
            payload = {"error": message}
        else:
            # 服务器内部错误，生产模式下脱敏
            payload = {"error": "Internal server error", "code": status}
            logger.error("API 内部错误 (status=%d): %s", status, message)
        self._send_json(payload, status=status)

    def _check_auth(self) -> bool:
        """检查请求认证。

        - 若配置了 api_token，检查 Authorization: Bearer <token>
        - 若配置了 allow_localhost_only，拒绝非本地请求
        """
        if self._allow_localhost_only:
            client_addr = self.client_address[0]
            # 允许 IPv4/IPv6 本地地址
            if client_addr not in ("127.0.0.1", "::1", "localhost"):
                logger.warning("拒绝非本地请求: %s %s", client_addr, self.path)
                self._send_error("Forbidden: localhost only", 403)
                return False

        if self._api_token:
            auth_header = self.headers.get("Authorization", "")
            expected = f"Bearer {self._api_token}"
            if auth_header != expected:
                logger.warning("认证失败: %s", self.path)
                self._send_error("Unauthorized", 401)
                return False

        return True

    def _check_content_length(self) -> bool:
        """检查请求体大小是否超限。"""
        length_str = self.headers.get("Content-Length", "0")
        try:
            length = int(length_str)
        except ValueError:
            length = 0
        if length > _MAX_CONTENT_LENGTH:
            logger.warning("请求体过大: %d bytes", length)
            self._send_error(f"Content too large (max {_MAX_CONTENT_LENGTH} bytes)", 413)
            return False
        return True

    def _parse_query(self) -> dict[str, list[str]]:
        """解析 URL 查询参数。"""
        parsed = urllib.parse.urlparse(self.path)
        return urllib.parse.parse_qs(parsed.query)

    def _parse_path(self) -> tuple[str, list[str]]:
        """解析路径，返回 (path_without_query, path_parts)。"""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        parts = path.strip("/").split("/")
        return path, parts

    def _safe_handle(self, handler: Callable[[], None]) -> None:
        """包装处理器，统一处理异常和认证。"""
        if not self._check_auth():
            return
        if not self._check_content_length():
            return
        try:
            handler()
        except ValueError as e:
            # URI 校验失败等客户端错误
            logger.warning("客户端错误: %s", e)
            self._send_error(str(e), 400)
        except Exception as e:
            logger.exception("API 处理异常: %s", self.path)
            self._send_error(str(e), 500)

    def do_GET(self) -> None:
        """处理 GET 请求。"""
        path, parts = self._parse_path()
        query = self._parse_query()

        def _handler() -> None:
            if path == "/memories/search" or path == "/memories/search/":
                self._handle_search(query)
            elif path.startswith("/memories/") and len(parts) >= 3:
                uri = "/".join(parts[1:])
                self._handle_read(uri)
            elif path == "/memories/stats" or path == "/memories/stats/":
                self._handle_stats()
            elif path == "/injected-meta" or path == "/injected-meta/":
                self._handle_injected_meta()
            elif path == "/health" or path == "/health/":
                self._handle_health()
            else:
                self._send_error("Not found", 404)

        self._safe_handle(_handler)

    def do_DELETE(self) -> None:
        """处理 DELETE 请求。"""
        path, parts = self._parse_path()

        def _handler() -> None:
            if path.startswith("/memories/") and len(parts) >= 3:
                uri = "/".join(parts[1:])
                self._handle_delete(uri)
            else:
                self._send_error("Not found", 404)

        self._safe_handle(_handler)

    def do_POST(self) -> None:
        """处理 POST 请求。"""
        path, _ = self._parse_path()

        def _handler() -> None:
            if path == "/maintenance" or path == "/maintenance/":
                self._handle_maintenance()
            else:
                self._send_error("Not found", 404)

        self._safe_handle(_handler)

    def do_OPTIONS(self) -> None:
        """处理 CORS 预检请求。"""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, DELETE, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    # ───────────────────────────────────────────────
    # 具体处理器
    # ───────────────────────────────────────────────

    def _handle_search(self, query: dict[str, list[str]]) -> None:
        """搜索记忆。"""
        q = query.get("q", [""])[0]
        try:
            limit = int(query.get("limit", ["10"])[0])
        except ValueError:
            self._send_error("Invalid query parameter 'limit'", 400)
            return

        if not q:
            self._send_error("Missing query parameter 'q'", 400)
            return

        # 限制返回数量，防止过大请求
        limit = min(max(limit, 1), 100)

        try:
            results = self.pipeline._retriever.retrieve(
                query=q,
                limit=limit,
                token_budget=2000,
            )
        except Exception:
            # 回退到 FTS 搜索
            results = []
            try:
                fts_results = self.pipeline._index.search(q, limit=limit)
                from dogcode_memory.format import deserialize_memory
                for uri, score in fts_results:
                    content = self.pipeline._store.read(uri)
                    abstract = uri
                    if content:
                        try:
                            memory = deserialize_memory(content)
                            abstract = memory.abstract or uri
                        except Exception:
                            pass
                    results.append(type("R", (), {
                        "uri": uri, "abstract": abstract, "overview": "",
                        "content": content or "", "level": 0, "score": score,
                        "updated_at": "",
                    })())
            except Exception:
                pass

        # 转换为前端友好的格式
        items = []
        for r in results:
            items.append({
                "uri": r.uri,
                "abstract": r.abstract,
                "overview": r.overview,
                "content": r.content,
                "level": r.level,
                "score": r.score,
                "updated_at": getattr(r, "updated_at", ""),
            })

        self._send_json({
            "query": q,
            "count": len(items),
            "results": items,
        })

    def _handle_read(self, uri: str) -> None:
        """读取单条记忆。"""
        content = self.pipeline._store.read(uri)
        if not content:
            self._send_error(f"Memory not found: {uri}", 404)
            return

        # 尝试解析元数据
        try:
            from dogcode_memory.format import deserialize_memory
            memory = deserialize_memory(content)
            self._send_json({
                "uri": uri,
                "type": memory.type,
                "space": memory.space,
                "abstract": memory.abstract,
                "content": memory.content,
                "fields": memory.fields,
                "created_at": memory.created_at,
                "updated_at": memory.updated_at,
                "active_count": memory.active_count,
            })
        except Exception:
            # 解析失败返回原始内容
            self._send_json({
                "uri": uri,
                "raw_content": content,
            })

    def _handle_delete(self, uri: str) -> None:
        """删除记忆。"""
        content = self.pipeline._store.read(uri)
        if not content:
            self._send_error(f"Memory not found: {uri}", 404)
            return

        self.pipeline._store.delete(uri)
        self.pipeline._index.remove(uri)
        self._send_json({"deleted": True, "uri": uri})

    def _handle_stats(self) -> None:
        """获取统计信息。"""
        stats = self.pipeline.get_stats()
        self._send_json(stats)

    def _handle_injected_meta(self) -> None:
        """获取最近一次注入记忆的元数据。"""
        meta = self.pipeline.get_injected_memory_meta()
        self._send_json(meta or {"last_injection": None})

    def _handle_maintenance(self) -> None:
        """触发维护任务。"""
        result = self.pipeline.run_maintenance()
        self._send_json({"maintenance": result})

    def _handle_health(self) -> None:
        """健康检查。"""
        self._send_json({
            "status": "ok",
            "memory_enabled": self.pipeline._enabled,
        })
