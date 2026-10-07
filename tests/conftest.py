import sys
import threading
from collections.abc import Iterator

import pytest

# Module singletons that importing ``main`` sets: (module, global, attribute of ``main``).
MAIN_SINGLETONS = (
    ("saved_config", "singleton", "configs"),
    ("gpt_client", "llms_client", "gpt_client"),
    ("updater_service", "updater_service", "updater_service"),
    ("interactions_model", "singleton", "interactions_model"),
)


@pytest.fixture(autouse=True, scope="module")
def release_backend_singletons() -> Iterator[None]:
    """After each test module, stop the interactions polling thread and release ``main``'s services.

    The interactions model polls a queue on a non-daemon thread that never ends on its own, so
    the test process couldn't exit; importing ``main`` starts one even when the import fails
    later. ``main`` also builds the backend's services from its test's mocks, and later test
    modules would get them from the factories instead of their own.

    Yields:
        None: Control to the test module.
    """
    yield
    interactions = getattr(sys.modules.get("interactions_model"), "singleton", None)
    if interactions is not None:
        interactions.poll_queue = False
        thread = getattr(interactions, "t", None)
        if isinstance(thread, threading.Thread):
            thread.join(timeout=5)
    main = sys.modules.get("main")
    if main is None:
        return
    for module_name, global_name, attribute in MAIN_SINGLETONS:
        module = sys.modules[module_name]
        if getattr(module, global_name) is getattr(main, attribute):
            setattr(module, global_name, None)
