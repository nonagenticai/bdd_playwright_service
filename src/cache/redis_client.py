"""Redis client for caching operations -- delegates to w2v_common.

This module re-exports the shared Redis client so that existing imports
(``from src.cache.redis_client import ...``) continue to work without
modification.
"""

from w2v_common.cache.redis_client import (
    RedisClient as RedisClient,
)
from w2v_common.cache.redis_client import (
    close_redis as close_redis,
)
from w2v_common.cache.redis_client import (
    get_redis_client as get_redis_client,
)
from w2v_common.cache.redis_client import (
    init_redis as init_redis,
)

__all__ = ["RedisClient", "close_redis", "get_redis_client", "init_redis"]
