from __future__ import annotations
import asyncio

import uvicorn
from arc_service.service import Service

from arc_v2_inference_runtime.api import app
from arc_v2_inference_runtime.constants import ARC_RUNTIME_PORT
from arc_v2_inference_runtime.engine.config import EngineConfig
from arc_v2_inference_runtime.engine.engine import Engine


class InferenceRuntime(Service):
    def __init__(self) -> None:
        super().__init__(
            name="ARC Inference Runtime Server",
            version="0.1.0",
            description="Inference server backend for ARC",
        )

        self._server: uvicorn.Server | None = None
        self._engine: Engine | None = None
        self._ready = False
        self._server_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        assert self.ctx is not None
        self.ctx.logger.info("Loading runtime...")

        config = EngineConfig.from_env()
        self.ctx.logger.debug(f"Engine-config: {config}")
        self.ctx.logger.info("Config ready")

        self._engine = Engine(config)
        await self._engine.start()
        self.ctx.logger.info("Engine Ready")

        app.state.engine = self._engine

        self.ctx.logger.info(
            f"Model loaded: {self._engine.model_name} (n_gpu_layers={config.n_gpu_layers}, n_ctx={config.n_ctx})"
        )

        self.ctx.logger.info("Runtime ready")

        server_config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=ARC_RUNTIME_PORT.value,
            log_level="info",
        )

        self._server = uvicorn.Server(server_config)
        await self._server.serve()

    async def ready(self):
        ok = (
            self._engine is not None
            and self._server is not None
            and self._server.started
        )
        return ok, None if ok else "runtime not ready yet"

    async def healthy(self):
        # server actually started and is alive
        if self._server is None:
            return False, "server not created"
        return (
            self._server.started,
            None if self._server.started else "server not started yet",
        )

    async def stop(self) -> None:
        server = self._server
        server_task = self._server_task

        try:
            if server is not None:
                server.should_exit = True

            if server_task is not None:
                try:
                    await asyncio.shield(server_task)
                except asyncio.CancelledError:
                    if not server_task.cancelled():
                        raise

                    if self.ctx is not None:
                        self.ctx.logger.warning(
                            "Uvicorn task was cancelled during shutdown"
                        )
        finally:
            try:
                if self._engine is not None:
                    await self._engine.unload()
            finally:
                self._engine = None
                self._server = None
                self._server_task = None
