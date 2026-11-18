"""License token authentication middleware -- delegates to w2v_common."""

from __future__ import annotations

from typing import Any

import w2v_common.auth.middleware as _mw_mod


def __getattr__(name: str) -> Any:
    """Delegate attribute lookups to w2v_common.auth.middleware."""
    try:
        return getattr(_mw_mod, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
