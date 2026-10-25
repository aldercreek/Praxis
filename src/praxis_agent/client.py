"""Native client transport. Closing this object never stops server execution."""

from __future__ import annotations

import asyncio
import json
from threading import Event, Lock, Thread
from uuid import uuid4

import httpx

from praxis_agent.service.discovery import ensure_service


class APIError(RuntimeError):
    def __init__(self, response):
        try:
            error = response.json()["error"]
        except (ValueError, KeyError):
            error = {"code": "http_error", "message": f"HTTP {response.status_code}"}
        super().__init__(error["message"])
        self.code, self.status = error["code"], response.status_code


class ServiceClient:
    def __init__(self, directory=None, *, endpoint=None):
        self.directory = directory
        self.endpoint = endpoint or ensure_service(directory)
        self.subscriptions = []
        self.discovery_lock = Lock()

    def rediscover(self):
        with self.discovery_lock:
            self.endpoint = ensure_service(self.directory)

    def request(self, method, path, data=None):
        for attempt in range(4):
            try:
                with httpx.Client(trust_env=False, timeout=30) as client:
                    response = client.request(
                        method,
                        self.endpoint["url"] + path,
                        json=data,
                        headers={"Authorization": f"Bearer {self.endpoint['token']}"},
                    )
                if response.status_code == 401 and attempt < 3:
                    self.rediscover()
                    continue
                if response.is_error:
                    raise APIError(response)
                return response.json()
            except httpx.TransportError:
                if attempt == 3:
                    raise
                if attempt == 2:
                    self.rediscover()
                else:
                    Event().wait(0.5 * 2**attempt)

    def create(self, workspace, *, model=None):
        return self.request(
            "POST",
            "/api/sessions",
            dict(command_id=uuid4().hex, workspace=str(workspace), model=model),
        )["session"]

    def subscribe(self, session_id, callback, *, after=0, on_error=None, on_ready=None):
        subscription = Subscription(
            self, session_id, callback, after, on_error, on_ready
        )
        self.subscriptions.append(subscription)
        subscription.start()
        return subscription

    def close(self):
        for subscription in self.subscriptions:
            subscription.close()
        self.subscriptions.clear()


class Subscription:
    def __init__(self, client, session_id, callback, after, on_error, on_ready):
        self.client, self.session_id, self.callback, self.on_error = (
            client,
            session_id,
            callback,
            on_error,
        )
        self.applied = after
        self.on_ready = on_ready
        self.ready = Event()
        self.closed = Event()
        self.thread = Thread(target=self._run, name="praxis-events", daemon=True)
        self.loop = self.task = None

    def start(self):
        self.thread.start()
        if not self.ready.wait(20):
            self.close()
            raise TimeoutError("SSE subscription did not become ready")

    def acknowledge(self, sequence):
        self.applied = max(self.applied, sequence)

    def _run(self):
        try:
            asyncio.run(self._listen())
        except asyncio.CancelledError:
            pass

    async def _listen(self):
        self.loop, self.task = asyncio.get_running_loop(), asyncio.current_task()
        failures = 0
        while not self.closed.is_set():
            endpoint = self.client.endpoint
            try:
                async with httpx.AsyncClient(
                    trust_env=False, timeout=httpx.Timeout(45, connect=3)
                ) as client:
                    async with client.stream(
                        "GET",
                        endpoint["url"] + f"/api/sessions/{self.session_id}/events",
                        params={"after": self.applied},
                        headers={"Authorization": f"Bearer {endpoint['token']}"},
                    ) as response:
                        if response.is_error:
                            await response.aread()
                            raise APIError(response)
                        data, event_type = [], ""
                        async for line in response.aiter_lines():
                            if self.closed.is_set():
                                return
                            if line.startswith("event:"):
                                event_type = line[6:].strip()
                            elif line.startswith("data:"):
                                data.append(line[5:].lstrip())
                            elif line == "":
                                if event_type == "ready":
                                    if self.on_ready:
                                        self.on_ready()
                                    self.ready.set()
                                elif data:
                                    frame = json.loads("\n".join(data))
                                    if frame["sequence"] > self.applied:
                                        self.callback(frame)
                                data, event_type = [], ""
                                failures = 0
                raise ConnectionError("SSE stream ended")
            except (httpx.HTTPError, ConnectionError, ValueError, APIError) as error:
                if self.on_error:
                    self.on_error(str(error))
                if isinstance(error, APIError) and error.status not in (401, 503):
                    return
                if failures >= 3:
                    try:
                        await asyncio.to_thread(self.client.rediscover)
                    except RuntimeError as failure:
                        if self.on_error:
                            self.on_error(str(failure))
                    failures = 0
                await asyncio.sleep(min(0.5 * 2**failures, 2))
                failures += 1

    def close(self):
        self.closed.set()
        if self.loop and self.task and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.task.cancel)
        if self.thread.is_alive():
            self.thread.join(timeout=3)
