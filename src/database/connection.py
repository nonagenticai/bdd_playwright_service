"""Database connection module -- delegates to w2v_common.

Re-exports the shared database connection functions so that existing imports
(``from src.database.connection import ...``) continue to work.
"""

from w2v_common.database.connection import (
    close_engine as close_engine,
)
from w2v_common.database.connection import (
    get_engine as get_engine,
)
from w2v_common.database.connection import (
    health_check as health_check,
)

__all__ = ["close_engine", "get_engine", "health_check"]
