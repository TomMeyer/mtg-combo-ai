import asyncio

from hypercorn.asyncio import serve

from mtg_ai_rag_webserver.main import app
from mtg_ai_rag_webserver.server_settings import server_settings


def run():
    config = server_settings.to_hypercorn_config()
    asyncio.run(serve(app, config))


if __name__ == "__main__":
    run()
