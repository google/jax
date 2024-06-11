from typing import Any


def register_handler(
    c_api: Any,
    target_name: str,
    handler: Any,
    xla_platform_name: str,
    api_version: int = ...,
    traits: int = ...,
) -> None: ...
