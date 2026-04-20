import asyncio
import os
import signal
import sys

from dotenv import load_dotenv
from pyngrok import ngrok

import logger as log
from server import start_server, stop_server


PORT = int(os.environ.get("PORT", "3000"))


def _missing_required_env() -> list[str]:
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "NGROK_AUTHTOKEN",
    ]
    missing: list[str] = []
    for key in required:
        val = os.environ.get(key, "")
        if not val or val.startswith("your_"):
            missing.append(key)
    return missing


async def main() -> None:
    load_dotenv()

    missing = _missing_required_env()
    if missing:
        print("\n[ERROR] Missing required .env values:")
        for key in missing:
            print(f"  {key}")
        sys.exit(1)

    runner = await start_server(PORT)
    log.info("AudioHook server listening", {"port": PORT})

    log.info("Starting ngrok tunnel...")
    ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])

    static_domain = os.environ.get("NGROK_STATIC_URL", "").strip()
    try:
        if static_domain:
            listener = ngrok.connect(addr=PORT, proto="http", domain=static_domain)
        else:
            listener = ngrok.connect(addr=PORT, proto="http")
    except Exception as err:
        log.error("ngrok failed", {"error": str(err)})
        await stop_server(runner)
        sys.exit(1)

    public_http_url = listener.public_url
    public_wss_url = public_http_url.replace("https://", "wss://").replace("http://", "ws://")

    log.info("ngrok tunnel up", {"publicHttpUrl": public_http_url, "publicWssUrl": public_wss_url})

    print("\n=======================================================")
    print("  Genesys AudioHook <-> Azure OpenAI Realtime Bridge")
    print("=======================================================")
    print(f"  Local port : {PORT}")
    print(f"  Public WSS : {public_wss_url}")
    print(f"  Public HTTP: {public_http_url}")
    print(f"  Deployment : {os.environ.get('AZURE_OPENAI_DEPLOYMENT')}")
    if static_domain:
        print("  Static URL : YES (URL will not change on restart)")
    print("=======================================================")
    print("\nGenesys AudioConnector Base URI:")
    print(f"  {public_wss_url}")
    print("\nPress Ctrl+C to stop.\n")

    stop_event = asyncio.Event()

    def _handle_stop(*_args: object) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    if os.name == "nt":
        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)
    else:
        loop.add_signal_handler(signal.SIGINT, _handle_stop)
        loop.add_signal_handler(signal.SIGTERM, _handle_stop)

    await stop_event.wait()

    log.info("Shutting down...")
    try:
        ngrok.disconnect(public_http_url)
    except Exception:
        pass
    try:
        ngrok.kill()
    except Exception:
        pass
    await stop_server(runner)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as err:
        log.error("Fatal", {"error": str(err)})
        sys.exit(1)
