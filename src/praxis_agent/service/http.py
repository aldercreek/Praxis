"""HTTP commands and one ordered journal reader per SSE connection."""

from __future__ import annotations

import asyncio
import hmac
import json

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from praxis_agent.sessions import SessionNotFoundError, SessionStoreError

from .errors import ServiceError, required_text

PROTOCOL_VERSION = 1


def create_app(manager, *, token, instance_id, heartbeat=15.0, poll_interval=0.1):
    store = manager.store

    async def body(request):
        if int(request.headers.get("content-length", "0")) > 2_000_000:
            raise ServiceError("too_large", "Request is too large", 413)
        raw = await request.body()
        if len(raw) > 2_000_000:
            raise ServiceError("too_large", "Request is too large", 413)
        try:
            data = json.loads(raw)
        except ValueError:
            raise ServiceError("invalid_json", "Expected JSON object")
        if not isinstance(data, dict):
            raise ServiceError("invalid_json", "Expected JSON object")
        return data

    async def health(request):
        return JSONResponse(
            dict(
                service="praxis",
                instance_id=instance_id,
                protocol_version=PROTOCOL_VERSION,
                ready=True,
                **manager.configuration(),
            )
        )

    async def sessions(request):
        if request.method == "POST":
            return JSONResponse(
                await asyncio.to_thread(manager.create, await body(request))
            )
        return JSONResponse(
            await asyncio.to_thread(manager.list, request.query_params.get("workspace"))
        )

    async def mcp_servers(request):
        return JSONResponse(manager.mcp_status())

    async def snapshot(request):
        return JSONResponse(
            await asyncio.to_thread(store.snapshot, request.path_params["session_id"])
        )

    async def delete_session(request):
        try:
            return JSONResponse(
                await asyncio.to_thread(manager.delete, request.path_params["session_id"])
            )
        except ServiceError:
            raise
        except Exception as error:
            raise ServiceError("session_delete_failed", "Could not finish deleting this session. Retry deletion.", 500) from error

    async def inputs(request):
        data = await body(request)
        runtime = await asyncio.to_thread(
            manager.runtime, request.path_params["session_id"]
        )
        return JSONResponse(
            await asyncio.to_thread(
                runtime.submit,
                required_text(data, "command_id"),
                required_text(data, "text"),
            )
        )

    async def resume(request):
        data = await body(request)
        runtime = await asyncio.to_thread(
            manager.runtime, request.path_params["session_id"]
        )
        return JSONResponse(
            await asyncio.to_thread(runtime.resume, required_text(data, "command_id"))
        )

    async def delete_input(request):
        runtime = await asyncio.to_thread(
            manager.runtime, request.path_params["session_id"]
        )
        return JSONResponse(
            await asyncio.to_thread(
                runtime.delete_input, request.path_params["command_id"]
            )
        )

    async def cancel(request):
        runtime = await asyncio.to_thread(
            manager.runtime, request.path_params["session_id"]
        )
        return JSONResponse(
            await asyncio.to_thread(runtime.cancel, request.path_params["turn_id"])
        )

    async def decide(request):
        data = await body(request)
        return JSONResponse(
            await asyncio.to_thread(
                manager.decide, request.path_params["approval_id"], data.get("approved")
            )
        )

    async def commands(request):
        runtime = await asyncio.to_thread(
            manager.runtime, request.path_params["session_id"]
        )
        with runtime.lock:
            runtime.ensure_open()
            command_sessions = runtime.agent.command_sessions if runtime.agent else None
        if request.method == "POST":
            if command_sessions is None:
                raise ServiceError("command_not_found", "Command does not exist", 404)
            await asyncio.to_thread(
                command_sessions.stop, request.path_params["command_id"]
            )
            return JSONResponse({"stopped": True})
        return JSONResponse(
            {
                "commands": command_sessions.summaries()
                if command_sessions
                else []
            }
        )

    async def configure(request):
        data = await body(request)
        return JSONResponse(await asyncio.to_thread(manager.configure, data))

    async def events(request):
        session_id = request.path_params["session_id"]
        await asyncio.to_thread(store.require_available, session_id)
        try:
            after = int(
                request.query_params.get(
                    "after", request.headers.get("last-event-id", "0")
                )
            )
            if after < 0:
                raise ValueError()
        except ValueError:
            raise ServiceError("invalid_cursor", "after must be a nonnegative integer")
        with store.db.connection() as connection:
            latest = connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        if after > latest:
            raise ServiceError(
                "invalid_cursor",
                "Cursor is ahead of this session; load a snapshot",
                409,
            )

        async def stream():
            cursor = after
            loop = asyncio.get_running_loop()
            last_heartbeat = loop.time()
            # Readiness is transport control, not a persisted business event.
            yield "event: ready\ndata: {}\n\n"
            while not manager.closed:
                try:
                    await asyncio.to_thread(store.require_available, session_id)
                except ServiceError as error:
                    if error.status not in (404, 410):
                        raise
                    # A terminal transport control survives removal of the event journal.
                    yield f"event: deleted\ndata: {json.dumps({'session_id': session_id})}\n\n"
                    return
                batch = await asyncio.to_thread(store.events, session_id, cursor)
                for frame in batch:
                    yield f"id: {frame['sequence']}\nevent: {frame['event']}\ndata: {json.dumps(frame, ensure_ascii=False)}\n\n"
                    cursor = frame["sequence"]
                if loop.time() - last_heartbeat >= heartbeat:
                    yield ": heartbeat\n\n"
                    last_heartbeat = loop.time()
                if not batch:
                    # Bounded polling cannot lose a notification and does not block
                    # Agent execution or hold a database transaction during I/O.
                    await asyncio.sleep(poll_interval)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def service_error(request, error):
        return JSONResponse(
            {"error": {"code": error.code, "message": str(error)}},
            status_code=error.status,
        )

    async def store_error(request, error):
        status = 404 if isinstance(error, SessionNotFoundError) else 500
        return JSONResponse(
            {
                "error": {
                    "code": "session_not_found" if status == 404 else "storage_error",
                    "message": str(error),
                }
            },
            status_code=status,
        )

    routes = [
        Route("/api/health", health),
        Route("/api/mcp/servers", mcp_servers),
        Route("/api/settings", configure, methods=["PUT"]),
        Route("/api/sessions", sessions, methods=["GET", "POST"]),
        Route("/api/sessions/{session_id}", delete_session, methods=["DELETE"]),
        Route("/api/sessions/{session_id}/snapshot", snapshot),
        Route("/api/sessions/{session_id}/inputs", inputs, methods=["POST"]),
        Route("/api/sessions/{session_id}/inputs/resume", resume, methods=["POST"]),
        Route(
            "/api/sessions/{session_id}/inputs/{command_id}",
            delete_input,
            methods=["DELETE"],
        ),
        Route(
            "/api/sessions/{session_id}/turns/{turn_id}/cancel",
            cancel,
            methods=["POST"],
        ),
        Route("/api/approvals/{approval_id}/decision", decide, methods=["POST"]),
        Route("/api/sessions/{session_id}/events", events),
        Route("/api/sessions/{session_id}/commands", commands),
        Route(
            "/api/sessions/{session_id}/commands/{command_id}/stop",
            commands,
            methods=["POST"],
        ),
    ]
    app = Starlette(
        routes=routes,
        exception_handlers={
            ServiceError: service_error,
            SessionStoreError: store_error,
        },
    )

    class Authentication:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                headers = dict(scope.get("headers", []))
                if not hmac.compare_digest(
                    headers.get(b"authorization", b""), f"Bearer {token}".encode()
                ):
                    response = JSONResponse(
                        {
                            "error": {
                                "code": "unauthorized",
                                "message": "Authentication required",
                            }
                        },
                        status_code=401,
                    )
                    return await response(scope, receive, send)
                if b"origin" in headers:
                    response = JSONResponse(
                        {
                            "error": {
                                "code": "origin_denied",
                                "message": "Browser origin not allowed",
                            }
                        },
                        status_code=403,
                    )
                    return await response(scope, receive, send)
            return await self.app(scope, receive, send)

    app.add_middleware(Authentication)
    return app
