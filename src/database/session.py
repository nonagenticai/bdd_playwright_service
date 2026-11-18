"""Database session management -- delegates to w2v_common.

Re-exports the shared session functions so that existing imports
(``from src.database.session import ...``) continue to work.
"""

from w2v_common.database.session import (
    close_session_maker as close_session_maker,
)
from w2v_common.database.session import (
    get_db as get_db,
)
from w2v_common.database.session import (
    get_db_session as get_db_session,
)
from w2v_common.database.session import (
    get_session_maker as get_session_maker,
)

__all__ = ["close_session_maker", "get_db", "get_db_session", "get_session_maker"]
