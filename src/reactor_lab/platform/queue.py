"""Redis-backed Dramatiq actor; messages contain only authoritative run IDs."""

from __future__ import annotations

import dramatiq
from dramatiq.brokers.redis import RedisBroker

from .settings import get_settings

# Constructing a RedisBroker does not connect; connections are opened on enqueue/work.
broker = RedisBroker(url=get_settings().redis_url)
dramatiq.set_broker(broker)


@dramatiq.actor(queue_name="reactor-runs")
def execute_run(run_id: str) -> None:
    from .worker import execute_run_by_id

    execute_run_by_id(run_id)
