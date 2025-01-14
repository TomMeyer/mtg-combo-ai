from pathlib import Path

import trio
from hypercorn import Config
from hypercorn.trio import serve

from mtg_ai_webserver.main import app

script_path = Path(__file__).parent

def run():
    config = Config.from_toml(script_path.joinpath("config.toml"))
    trio.run(serve, app, config)


if __name__ == "__main__":
    run()
