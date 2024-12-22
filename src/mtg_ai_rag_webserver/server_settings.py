import logging
import os
from ssl import VerifyFlags, VerifyMode
from typing import Any, Optional

from hypercorn.config import BYTES, OCTETS, SECONDS, Config
from hypercorn.logging import Logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class MTGAIRagServerSettings(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, env_prefix="MTG_AI_MODEL")

    bind: list[str] = ["0.0.0.0:8889"]
    insecure_bind: list[str] = []
    quic_bind: list[str] = []
    quic_addresses: list[tuple] = []
    log: Optional[Logger] = None
    root_path: str = ""

    access_log_format: str = (
        '%(h)s %(l)s %(l)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
    )
    accesslog: Optional[logging.Logger | str] = None
    alpn_protocols: list[str] = ["h2", "http/1.1"]
    alt_svc_headers: list[str] = []
    application_path: str
    backlog: int = 100
    ca_certs: Optional[str] = None
    certfile: Optional[str] = None
    ciphers: str = "ECDHE+AESGCM"
    debug: bool = False
    dogstatsd_tags: str = ""
    errorlog: Optional[logging.Logger | str] = "-"
    graceful_timeout: float = 3 * SECONDS
    read_timeout: Optional[int] = None
    group: Optional[int] = None
    h11_max_incomplete_size: int = 16 * 1024 * BYTES
    h11_pass_raw_headers: bool = False
    h2_max_concurrent_streams: int = 100
    h2_max_header_list_size: int = 2**16
    h2_max_inbound_frame_size: int = 2**14 * OCTETS
    include_date_header: bool = True
    include_server_header: bool = True
    keep_alive_timeout: float = 5 * SECONDS
    keep_alive_max_requests: int = 1000
    keyfile: Optional[str] = None
    keyfile_password: Optional[str] = None
    logconfig: Optional[str] = None
    logconfig_dict: Optional[dict] = None
    logger_class: type[Any] = Logger
    loglevel: str = "INFO"
    max_app_queue_size: int = 10
    max_requests: Optional[int] = None
    max_requests_jitter: int = 0
    pid_path: Optional[str] = None
    server_names: list[str] = []
    shutdown_timeout: float = 60 * SECONDS
    ssl_handshake_timeout: float = 60 * SECONDS
    startup_timeout: float = 60 * SECONDS
    statsd_host: Optional[str] = None
    statsd_prefix: str = ""
    umask: Optional[int] = None
    use_reloader: bool = False
    user: Optional[int] = None
    verify_flags: Optional[VerifyFlags] = None
    verify_mode: Optional[VerifyMode] = None
    websocket_max_message_size: int = 16 * 1024 * 1024 * BYTES
    websocket_ping_interval: Optional[float] = None
    worker_class: str = "asyncio"
    workers: int = 1
    wsgi_max_body_size: int = 16 * 1024 * 1024 * BYTES

    def to_hypercorn_config(self) -> Config:
        return Config.from_mapping(self.model_dump())


server_settings = MTGAIRagServerSettings(
    application_path=os.path.dirname(os.path.realpath(__file__))
)
