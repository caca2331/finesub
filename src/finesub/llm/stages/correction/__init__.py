"""The correction window loop, split by what each part is responsible for.

``run`` plans and finalizes a task; ``serial`` and ``parallel`` are the two
drivers over its windows; ``attempts`` is the per-window retry/split loop they
share; ``query_round`` is round 1; ``context`` holds the state a run's windows
share; ``commit`` owns what survives a rerun; ``metadata`` describes calls for
the artifacts.

Test doubles go on the module that does the name lookup, not here: a patch of
``finesub.llm.stages.correction.RoleClient`` would rebind nothing (see
``finesub.llm.stages.correction.run``).
"""

from .attempts import correction_role_for_profile  # noqa: F401
from .commit import WINDOW_CACHE_FILENAME  # noqa: F401
from .context import CorrectionRun  # noqa: F401
from .metadata import window_to_metadata  # noqa: F401
from .query_round import QueryRoundProduct, run_window_query_round  # noqa: F401
from .run import execute_correction_windows  # noqa: F401
