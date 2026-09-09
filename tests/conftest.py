import pytest

from classroom_monitor.alerts import AlertEngine, AlertStore
from classroom_monitor.config import DebounceConfig


@pytest.fixture
def store():
    s = AlertStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def clock():
    class Clock:
        t = 1_000.0

        def __call__(self):
            return self.t

        def advance(self, s):
            self.t += s

    return Clock()


@pytest.fixture
def engine(store, clock):
    return AlertEngine(store, DebounceConfig(per_track_cooldown_s=120, per_room_cooldown_s=30), clock=clock)
