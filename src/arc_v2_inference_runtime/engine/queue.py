from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


class CancelledByClient(Exception):
    pass


@dataclass
class _Job:
    id: str
    fn: Callable[[threading.Event], Iterator[Any]]
    out_queue: asyncio.Queue  # pyright: ignore[reportMissingTypeArgument]
    cancel_event: threading.Event = field(default_factory=threading.Event)
    loop: asyncio.AbstractEventLoop | None = None


_SENTINEL_DONE = object()


class RequestQueue:
    """FIFO queue of generation jobs, executed one at a time on a worker thread."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None  # pyright: ignore[reportMissingTypeArgument]
        self._jobs: dict[str, _Job] = {}

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None

    async def submit(
        self,
        fn: Callable[[threading.Event], Iterator[Any]],
        timeout_s: float | None = None,
    ) -> AsyncIterator[Any]:
        """Submit a job; fn receives a threading.Event to poll for cancellation
        and must be an iterator/generator yielding output chunks (e.g. tokens).
        Yields chunks as they arrive. Raises asyncio.TimeoutError if timeout_s
        elapses with the job still running (the underlying job is cancelled).
        """
        job = _Job(
            id=str(uuid.uuid4()),
            fn=fn,
            out_queue=asyncio.Queue(),
            loop=asyncio.get_running_loop(),
        )
        self._jobs[job.id] = job
        await self._queue.put(job)

        try:
            while True:
                get_coro = job.out_queue.get()
                item = (
                    await asyncio.wait_for(get_coro, timeout=timeout_s)
                    if timeout_s
                    else await get_coro
                )
                if item is _SENTINEL_DONE:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        except asyncio.TimeoutError:
            job.cancel_event.set()
            raise
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected / consumer stopped iterating — stop the worker early.
            job.cancel_event.set()
            raise
        finally:
            self._jobs.pop(job.id, None)

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            await self._run_job(job)

    async def _run_job(self, job: _Job) -> None:
        loop = job.loop or asyncio.get_running_loop()

        def _thread_target() -> None:
            try:
                for chunk in job.fn(job.cancel_event):
                    if job.cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(job.out_queue.put_nowait, chunk)
                loop.call_soon_threadsafe(job.out_queue.put_nowait, _SENTINEL_DONE)
            except Exception as exc:  # surface to the awaiting consumer
                loop.call_soon_threadsafe(job.out_queue.put_nowait, exc)

        thread = threading.Thread(target=_thread_target, daemon=True)
        thread.start()
        # Wait for the thread to finish before pulling the next job off the
        # queue — this is what makes the queue serial. The consumer in
        # submit() drains out_queue concurrently via the event loop.
        await asyncio.to_thread(thread.join)
