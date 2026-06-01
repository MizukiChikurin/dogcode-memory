"""异步操作 - 后台任务调度器。

基于 concurrent.futures.ThreadPoolExecutor 的后台任务池，
零外部依赖，用于将 LLM 记忆提取等耗时操作异步化。

线程安全：
- 所有任务提交和状态查询受 _lock 保护
- ThreadPoolExecutor 内部自动序列化任务队列
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import uuid
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class AsyncTask(Generic[T]):
    """单个异步任务封装，包含任务元数据和结果。"""

    def __init__(
        self,
        task_id: str,
        task_type: str,
        payload: dict[str, Any],
    ):
        """
        初始化异步任务。

        Args:
            task_id: 唯一任务标识符
            task_type: 任务类型，如 "memory_extraction", "maintenance"
            payload: 任务输入数据
        """
        self.task_id = task_id
        self.task_type = task_type
        self.payload = payload
        self.created_at = time.time()
        self.completed_at: float | None = None
        self.result: T | None = None
        self.error: Exception | None = None
        self._future: concurrent.futures.Future[T] | None = None

    def is_done(self) -> bool:
        """任务是否已完成（成功或失败）。"""
        return self._future is not None and self._future.done()

    def is_running(self) -> bool:
        """任务是否正在执行。"""
        return self._future is not None and self._future.running()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "is_done": self.is_done(),
            "is_running": self.is_running(),
            "has_error": self.error is not None,
            "error_message": str(self.error) if self.error else None,
        }


class AsyncTaskPool:
    """后台任务池，基于 ThreadPoolExecutor。"""

    def __init__(
        self,
        max_workers: int = 2,
        thread_name_prefix: str = "dogcode-memory-",
    ):
        """
        初始化任务池。

        Args:
            max_workers: 最大工作线程数，默认 2
            thread_name_prefix: 线程名前缀
        """
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._tasks: dict[str, AsyncTask[Any]] = {}
        self._lock = threading.Lock()
        self._shutdown = False

    def submit(
        self,
        task_type: str,
        fn: Callable[..., T],
        payload: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncTask[T]:
        """
        提交一个后台任务。

        Args:
            task_type: 任务类型
            fn: 执行函数
            payload: 任务输入数据（用于记录和查询）
            *args, **kwargs: 传递给 fn 的参数

        Returns:
            AsyncTask 实例
        """
        if self._shutdown:
            raise RuntimeError("AsyncTaskPool has been shut down")

        task_id = _generate_task_id(task_type)
        task = AsyncTask[T](
            task_id=task_id,
            task_type=task_type,
            payload=payload or {},
        )

        future = self._executor.submit(self._run_task, task, fn, args, kwargs)
        task._future = future

        with self._lock:
            self._tasks[task_id] = task

        return task

    def _run_task(
        self,
        task: AsyncTask[T],
        fn: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        """包装执行函数，捕获结果和异常。"""
        try:
            result = fn(*args, **kwargs)
            task.result = result
            task.completed_at = time.time()
            return result
        except Exception as e:
            task.error = e
            task.completed_at = time.time()
            raise

    def get_task(self, task_id: str) -> AsyncTask[Any] | None:
        """获取指定任务。"""
        with self._lock:
            return self._tasks.get(task_id)

    def get_pending(self) -> list[str]:
        """获取正在执行（未完成）的任务 ID 列表。"""
        with self._lock:
            return [
                tid for tid, task in self._tasks.items()
                if not task.is_done()
            ]

    def get_completed(self) -> list[str]:
        """获取已完成的任务 ID 列表。"""
        with self._lock:
            return [
                tid for tid, task in self._tasks.items()
                if task.is_done()
            ]

    def get_all_tasks(self) -> list[dict[str, Any]]:
        """获取所有任务的摘要信息。"""
        with self._lock:
            return [task.to_dict() for task in self._tasks.values()]

    def wait_for(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> AsyncTask[Any] | None:
        """
        等待指定任务完成。

        Args:
            task_id: 任务 ID
            timeout: 等待超时（秒），None 表示无限等待

        Returns:
            任务实例，任务不存在返回 None
        """
        task = self.get_task(task_id)
        if task is None or task._future is None:
            return task

        try:
            task._future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            pass
        except Exception:
            # 任务执行中抛出的异常已在 _run_task 中记录
            pass

        return task

    def wait_all(
        self,
        timeout: float | None = None,
    ) -> None:
        """等待所有已提交的任务完成。"""
        with self._lock:
            futures = [
                task._future for task in self._tasks.values()
                if task._future is not None
            ]

        if not futures:
            return

        concurrent.futures.wait(futures, timeout=timeout)

    def cleanup_completed(self, max_age_seconds: float = 3600.0) -> int:
        """
        清理已完成的旧任务。

        Args:
            max_age_seconds: 已完成任务保留的最长时间（秒）

        Returns:
            清理的任务数量
        """
        now = time.time()
        to_remove: list[str] = []

        with self._lock:
            for tid, task in self._tasks.items():
                if task.is_done() and task.completed_at:
                    if now - task.completed_at > max_age_seconds:
                        to_remove.append(tid)

            for tid in to_remove:
                del self._tasks[tid]

        return len(to_remove)

    def shutdown(self, wait: bool = True) -> None:
        """
        关闭任务池，释放资源。

        Args:
            wait: 是否等待所有任务完成
        """
        self._shutdown = True
        self._executor.shutdown(wait=wait)

    @property
    def is_shutdown(self) -> bool:
        """任务池是否已关闭。"""
        return self._shutdown


def _generate_task_id(task_type: str) -> str:
    """生成唯一任务 ID。"""
    return f"{task_type}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


class ExtractionTask:
    """记忆提取任务封装，提供便捷的创建和查询接口。"""

    def __init__(self, pool: AsyncTaskPool):
        """
        初始化提取任务管理器。

        Args:
            pool: 异步任务池
        """
        self._pool = pool

    def submit(
        self,
        session_id: str,
        extractor_fn: Callable[[], dict[str, Any]],
    ) -> AsyncTask[dict[str, Any]]:
        """
        提交记忆提取任务。

        Args:
            session_id: 会话 ID
            extractor_fn: 提取函数，返回统计字典

        Returns:
            异步任务实例
        """
        return self._pool.submit(
            task_type="memory_extraction",
            fn=extractor_fn,
            payload={"session_id": session_id},
        )

    def get_stats_for_session(self, session_id: str) -> list[dict[str, Any]]:
        """获取指定会话的所有提取任务统计。"""
        all_tasks = self._pool.get_all_tasks()
        return [
            t for t in all_tasks
            if t.get("payload", {}).get("session_id") == session_id
        ]
