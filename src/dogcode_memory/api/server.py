"""HTTP 服务封装 - 基于标准库 http.server 的独立记忆服务。

使用方式：
    # Python 代码启动
    from dogcode_memory.api.server import MemoryAPIServer
    server = MemoryAPIServer(pipeline, host="localhost", port=8081)
    server.start()

    # 命令行启动
    python -m dogcode_memory.api.server --port 8081 --storage-dir ~/.dogcode/memories
"""

from __future__ import annotations

import argparse
import threading
from http.server import HTTPServer, ThreadingHTTPServer
from typing import Any

from dogcode_memory.api.routes import MemoryAPIHandler
from dogcode_memory.pipeline import MemoryPipeline


class MemoryAPIServer:
    """记忆模块 HTTP API 服务。"""

    def __init__(
        self,
        pipeline: MemoryPipeline,
        host: str = "localhost",
        port: int = 8081,
        *,
        api_token: str | None = None,
        allow_localhost_only: bool = False,
        debug: bool = False,
    ):
        """
        初始化 API 服务。

        Args:
            pipeline: 记忆管线实例
            host: 监听地址
            port: 监听端口
            api_token: 可选的 API Token，设置后请求需携带 Authorization: Bearer <token>
            allow_localhost_only: 是否仅允许本地访问
            debug: 是否启用调试模式（暴露详细错误信息）
        """
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self._api_token = api_token
        self._allow_localhost_only = allow_localhost_only
        self._debug = debug
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _make_handler(self) -> type[MemoryAPIHandler]:
        """创建绑定 pipeline 的 Handler 类。"""
        pipeline = self.pipeline
        api_token = self._api_token
        allow_localhost_only = self._allow_localhost_only
        debug = self._debug

        class _Handler(MemoryAPIHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(
                    pipeline,
                    *args,
                    api_token=api_token,
                    allow_localhost_only=allow_localhost_only,
                    debug=debug,
                    **kwargs,
                )

        return _Handler

    def start(self, blocking: bool = False) -> None:
        """
        启动服务。

        Args:
            blocking: 是否阻塞当前线程
        """
        handler_class = self._make_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler_class)
        # 获取实际绑定的端口（port=0 时由系统分配）
        _, self.port = self._server.server_address

        if blocking:
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """停止服务。"""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        """服务是否正在运行。"""
        return self._server is not None

    def url(self) -> str:
        """获取服务 URL。"""
        return f"http://{self.host}:{self.port}"


# ───────────────────────────────────────────────
# 命令行入口
# ───────────────────────────────────────────────

def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="DogCode Memory HTTP API Server")
    parser.add_argument("--host", default="localhost", help="监听地址")
    parser.add_argument("--port", type=int, default=8081, help="监听端口")
    parser.add_argument(
        "--storage-dir",
        default="~/.dogcode/memories",
        help="记忆存储目录",
    )
    parser.add_argument(
        "--enabled",
        action="store_true",
        default=True,
        help="启用记忆系统",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="API Token，设置后请求需携带 Authorization: Bearer <token>",
    )
    parser.add_argument(
        "--localhost-only",
        action="store_true",
        default=False,
        help="仅允许本地访问",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="启用调试模式（暴露详细错误信息）",
    )
    args = parser.parse_args()

    import os

    storage_dir = os.path.expanduser(args.storage_dir)
    from dogcode_memory.config import MemoryConfig

    config = MemoryConfig(enabled=args.enabled)
    pipeline = MemoryPipeline.create(storage_dir=storage_dir, config=config)

    server = MemoryAPIServer(
        pipeline,
        host=args.host,
        port=args.port,
        api_token=args.api_token,
        allow_localhost_only=args.localhost_only,
        debug=args.debug,
    )
    print(f"Starting DogCode Memory API server at {server.url()}")
    print(f"Storage: {storage_dir}")
    if args.api_token:
        print("API Token: enabled")
    if args.localhost_only:
        print("Localhost only: enabled")
    print("Press Ctrl+C to stop")

    try:
        server.start(blocking=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == "__main__":
    main()
