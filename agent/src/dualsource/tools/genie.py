"""Genie removed in DualSource portfolio build."""
from __future__ import annotations
from typing import Any


class GenieClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Genie is not available in DualSource portfolio build")
