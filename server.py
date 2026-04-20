import json
import os

from aiohttp import WSMsgType, web

import logger as log
from session_handler import SessionHandler


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"service": "genesys-audiohook-bridge", "status": "running"})


async def ws_handler(request: web.Request) -> web.StreamResponse:
    headers = request.headers
    session_id = headers.get("audiohook-session-id")
    org_id = headers.get("audiohook-organization-id")
    correlation_id = headers.get("audiohook-correlation-id")
    api_key = headers.get("x-api-key")

    log.info(
        "Incoming AudioHook upgrade request",
        {
            "sessionId": session_id,
            "orgId": org_id,
            "correlationId": correlation_id,
            "url": str(request.rel_url),
        },
    )

    if not session_id:
        log.warn("Missing audiohook-session-id, rejecting")
        return web.Response(status=400, text="Missing audiohook-session-id")

    expected_key = os.environ.get("AUDIOHOOK_API_KEY")
    if expected_key and api_key != expected_key:
        log.warn("Invalid x-api-key, rejecting", {"sessionId": session_id})
        return web.Response(status=401, text="Unauthorized")

    ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
    await ws.prepare(request)

    log.info("AudioHook client connected", {"sessionId": session_id, "orgId": org_id})
    handler = SessionHandler(
        ws=ws,
        session_id=session_id,
        org_id=org_id,
        correlation_id=correlation_id,
    )

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                await handler.handle_binary_message(msg.data)
            elif msg.type == WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                    await handler.handle_text_message(payload)
                except Exception as err:
                    log.warn(
                        "Failed to parse text message",
                        {"sessionId": session_id, "error": str(err)},
                    )
            elif msg.type == WSMsgType.ERROR:
                log.error(
                    "AudioHook WebSocket error",
                    {"sessionId": session_id, "error": str(ws.exception())},
                )
                break
    finally:
        log.info("AudioHook client disconnected", {"sessionId": session_id})
        await handler.cleanup()

    return ws


async def start_server(port: int) -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/{tail:.*}", ws_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    return runner


async def stop_server(runner: web.AppRunner) -> None:
    await runner.cleanup()
