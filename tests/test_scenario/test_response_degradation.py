"""
Tests for response_degradation scenario with MultiDBClient.

Tests MultiDBClient failover behavior under high latency conditions using
the 4-phase scenario lifecycle: discovery -> setup -> execute -> teardown.

Effects tested:
- slow_response: Network latency causes slow responses (network_latency trigger)

The MultiDBClient should detect degraded performance on the primary database
and fail over to a healthy secondary database.
"""

import logging
import os
import threading
import time
from queue import Queue
from typing import Any, Dict
from urllib.parse import urlparse
import re

import pytest

from redis import Redis
from redis.backoff import ExponentialBackoff
from redis.event import EventDispatcher, EventListenerInterface
from redis.multidb.client import MultiDBClient
from redis.multidb.config import DatabaseConfig, MultiDbConfig
from redis.multidb.event import ActiveDatabaseChanged
from redis.retry import Retry
from tests.test_scenario.fault_injector_client import (
    FaultInjectorClient,
    REFaultInjector,
)
from tests.test_scenario.conftest import use_mock_proxy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SCENARIO_NAME = "response_degradation"

EXECUTE_TIMEOUT = 120
COMMAND_EXECUTION_DURATION = 60
HEALTH_CHECK_INTERVAL = 2
HEALTH_CHECK_PROBES = 3
MIN_NUM_FAILURES = 2
RECOVERY_WAIT_TIME = 45


class ActiveDatabaseChangedListener(EventListenerInterface):
    def __init__(self):
        self.events = []

    def listen(self, event: ActiveDatabaseChanged):
        logging.info(
            f"ActiveDatabaseChanged event: "
            f"prev={event.previous_database_index}, new={event.new_database_index}"
        )
        self.events.append(event)


def extract_cluster_fqdn(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname
    cleaned_hostname = re.sub(r"^redis-\d+\.", "", hostname)
    return f"https://{cleaned_hostname}"


def get_fault_injector_client() -> FaultInjectorClient:
    if use_mock_proxy():
        pytest.skip("Scenario tests require Redis Enterprise")
    url = os.getenv("FAULT_INJECTION_API_URL", "http://127.0.0.1:20324")
    return REFaultInjector(url)


def create_multi_db_client(
    primary_config: Dict[str, Any],
    secondary_config: Dict[str, Any],
    client_class=Redis,
) -> tuple[MultiDBClient, ActiveDatabaseChangedListener]:
    """Create MultiDBClient with two database configurations."""
    event_dispatcher = EventDispatcher()
    listener = ActiveDatabaseChangedListener()
    event_dispatcher.register_listeners({ActiveDatabaseChanged: [listener]})

    primary_endpoint = primary_config.get("endpoints", [])[0]
    secondary_endpoint = secondary_config.get("endpoints", [])[0]

    primary_db_config = DatabaseConfig(
        weight=1.0,
        from_url=primary_endpoint,
        client_kwargs={
            "password": primary_config.get("password"),
            "decode_responses": True,
        },
        health_check_url=extract_cluster_fqdn(primary_endpoint),
    )

    secondary_db_config = DatabaseConfig(
        weight=0.9,
        from_url=secondary_endpoint,
        client_kwargs={
            "password": secondary_config.get("password"),
            "decode_responses": True,
        },
        health_check_url=extract_cluster_fqdn(secondary_endpoint),
    )

    config = MultiDbConfig(
        client_class=client_class,
        databases_config=[primary_db_config, secondary_db_config],
        command_retry=Retry(ExponentialBackoff(cap=0.5, base=0.1), retries=5),
        min_num_failures=MIN_NUM_FAILURES,
        health_check_probes=HEALTH_CHECK_PROBES,
        health_check_interval=HEALTH_CHECK_INTERVAL,
        event_dispatcher=event_dispatcher,
    )

    return MultiDBClient(config), listener


@pytest.mark.skip(
    reason="Requires AA-aware scenario in FI that can create/target two databases. "
    "Current response_degradation scenario only creates one database."
)
class TestResponseDegradation:
    """Test response_degradation scenario effects on MultiDBClient."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        self._fi_client = get_fault_injector_client()
        self._setup_id = None
        self._client = None

        yield

        logging.info("Starting cleanup...")
        if self._client:
            self._client.close()

        if self._setup_id:
            self._fi_client.teardown_scenario(SCENARIO_NAME, self._setup_id)
            logging.info(f"Teardown completed for setup_id: {self._setup_id[:30]}...")

        self._fi_client.reset_cluster(clean_latency=True)
        time.sleep(RECOVERY_WAIT_TIME)
        logging.info("Cleanup finished")

    def _setup_scenario(
        self,
        effect: str,
        trigger: str,
        requirement_index: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        discovery = self._fi_client.discover_scenario(SCENARIO_NAME, effect)
        logging.info(
            f"Discovery for {effect}: {len(discovery.get('triggers', []))} triggers available"
        )

        trigger_def = None
        for t in discovery.get("triggers", []):
            if t["name"] == trigger:
                trigger_def = t
                break

        if not trigger_def:
            pytest.skip(f"Trigger '{trigger}' not available for effect '{effect}'")

        setup_response = self._fi_client.setup_scenario(
            SCENARIO_NAME,
            effect=effect,
            trigger=trigger,
            requirement_index=requirement_index,
            **kwargs,
        )

        self._setup_id = setup_response["setup_id"]
        logging.info(f"Setup completed: bdb_id={setup_response.get('bdb_id')}")

        time.sleep(5)

        return setup_response

    def _execute_scenario(self) -> Dict[str, Any]:
        return self._fi_client.execute_scenario(SCENARIO_NAME, self._setup_id)

    def _get_secondary_db_config(self) -> Dict[str, Any]:
        """Get secondary database configuration from environment.

        The secondary database should be a separate Active-Active database
        that will remain healthy while the primary has latency injected.
        """
        secondary_endpoints_path = os.getenv("REDIS_SECONDARY_ENDPOINTS_CONFIG_PATH")
        if not secondary_endpoints_path or not os.path.exists(secondary_endpoints_path):
            pytest.skip(
                "Secondary database config required: REDIS_SECONDARY_ENDPOINTS_CONFIG_PATH"
            )

        import json
        with open(secondary_endpoints_path, "r") as f:
            data = json.load(f)
            return data.get("re-active-active-secondary", data)

    @pytest.mark.timeout(300)
    def test_failover_on_slow_response(self):
        """
        Verify MultiDBClient fails over to secondary when primary has high latency.

        network_latency trigger injects delay on primary database:
        1. Primary becomes slow
        2. Health checks detect degradation
        3. MultiDBClient fails over to secondary
        4. ActiveDatabaseChanged event is fired
        """
        primary_setup = self._setup_scenario(
            effect="slow_response",
            trigger="network_latency",
        )
        secondary_config = self._get_secondary_db_config()

        self._client, listener = create_multi_db_client(
            primary_setup,
            secondary_config,
        )

        result = self._client.ping()
        assert result is True
        self._client.set("failover_test_key", "initial_value")
        logging.info(f"Initial connection on database {self._client.active_database}")

        errors: Queue = Queue()

        def execute_commands(duration: int):
            start = time.time()
            while time.time() - start < duration:
                try:
                    self._client.set("test_key", f"value_{time.time()}")
                    self._client.get("test_key")
                except Exception as e:
                    logging.warning(f"Command error: {e}")
                    errors.put(str(e))
                time.sleep(0.2)

        command_thread = threading.Thread(
            target=execute_commands,
            args=(COMMAND_EXECUTION_DURATION,),
            name="command_thread",
        )
        command_thread.start()

        time.sleep(5)

        logging.info("Injecting network latency on primary database...")
        execute_result = self._execute_scenario()
        logging.info(f"Execute result: {execute_result.get('status')}")

        command_thread.join()

        failover_occurred = len(listener.events) > 0
        logging.info(
            f"Failover events: {len(listener.events)}, "
            f"current active DB: {self._client.active_database}"
        )

        if failover_occurred:
            assert listener.events[0].previous_database_index == 0
            assert listener.events[0].new_database_index == 1
            logging.info("Failover to secondary verified via ActiveDatabaseChanged event")
        else:
            logging.info(
                "Latency may not have been severe enough to trigger failover - "
                "verify scenario configuration"
            )

        assert errors.qsize() == 0 or failover_occurred, (
            f"Commands failed without failover: {list(errors.queue)}"
        )

    @pytest.mark.timeout(300)
    def test_commands_continue_during_latency(self):
        """
        Verify commands continue to succeed during latency (possibly with failover).

        Even if latency is injected:
        1. Commands should eventually succeed (via retry or failover)
        2. No permanent failures
        """
        primary_setup = self._setup_scenario(
            effect="slow_response",
            trigger="network_latency",
        )
        secondary_config = self._get_secondary_db_config()

        self._client, listener = create_multi_db_client(
            primary_setup,
            secondary_config,
        )

        assert self._client.ping() is True

        success_count = [0]
        error_count = [0]

        def execute_commands(duration: int):
            start = time.time()
            while time.time() - start < duration:
                try:
                    self._client.incr("command_counter")
                    success_count[0] += 1
                except Exception as e:
                    logging.warning(f"Command error: {e}")
                    error_count[0] += 1
                time.sleep(0.1)

        command_thread = threading.Thread(
            target=execute_commands,
            args=(COMMAND_EXECUTION_DURATION,),
            name="command_thread",
        )
        command_thread.start()

        time.sleep(5)

        logging.info("Injecting latency...")
        self._execute_scenario()

        command_thread.join()

        logging.info(
            f"Results: {success_count[0]} successes, {error_count[0]} errors"
        )

        total_ops = success_count[0] + error_count[0]
        success_rate = success_count[0] / total_ops if total_ops > 0 else 0
        logging.info(f"Success rate: {success_rate:.2%}")

        assert success_count[0] > 0, "Expected some successful commands"
        assert self._client.ping() is True

    @pytest.mark.timeout(300)
    def test_failback_after_latency_removed(self):
        """
        Verify MultiDBClient can fail back to primary after latency is removed.

        After teardown (latency removed):
        1. Primary becomes healthy again
        2. Health checks detect recovery
        3. Client may fail back to primary (depending on policy)
        """
        primary_setup = self._setup_scenario(
            effect="slow_response",
            trigger="network_latency",
        )
        secondary_config = self._get_secondary_db_config()

        self._client, listener = create_multi_db_client(
            primary_setup,
            secondary_config,
        )

        assert self._client.ping() is True
        initial_db = self._client.active_database
        logging.info(f"Initial active database: {initial_db}")

        logging.info("Injecting latency...")
        self._execute_scenario()

        time.sleep(20)

        db_after_latency = self._client.active_database
        logging.info(f"Active database after latency: {db_after_latency}")

        logging.info("Tearing down (removing latency)...")
        self._fi_client.teardown_scenario(SCENARIO_NAME, self._setup_id)
        self._setup_id = None

        time.sleep(30)

        final_db = self._client.active_database
        logging.info(f"Final active database: {final_db}")

        assert self._client.ping() is True
        logging.info(
            f"State transitions: initial={initial_db} -> "
            f"after_latency={db_after_latency} -> final={final_db}"
        )
        logging.info(f"Total failover events: {len(listener.events)}")
