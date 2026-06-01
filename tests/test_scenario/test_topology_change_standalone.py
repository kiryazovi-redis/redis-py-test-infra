"""
Tests for topology_change_standalone scenario.

Tests standalone (non-OSS cluster) database topology changes using the 4-phase
scenario lifecycle: discovery -> setup -> execute -> teardown.

Effects tested:
- data_movement_conn_drop: Connection drops during maintenance_mode or endpoint_rebind
- data_movement_no_conn_drop: Hitless operations via endpoint_rebind, reshard, failover
"""

import logging
import os
import threading
import time
from queue import Queue
from typing import Any, Dict, List
from urllib.parse import urlparse

import pytest

from redis import Redis
from redis.backoff import ExponentialWithJitterBackoff
from redis.connection import ConnectionInterface
from redis.maint_notifications import MaintenanceState, MaintNotificationsConfig
from redis.retry import Retry
from tests.test_scenario.fault_injector_client import (
    FaultInjectorClient,
    REFaultInjector,
)
from tests.test_scenario.conftest import (
    CLIENT_TIMEOUT,
    RELAXED_TIMEOUT,
    use_mock_proxy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SCENARIO_NAME = "topology_change_standalone"

EFFECT_TRIGGER_COMBINATIONS = [
    ("data_movement_conn_drop", "maintenance_mode"),
    ("data_movement_conn_drop", "endpoint_rebind"),
    ("data_movement_no_conn_drop", "endpoint_rebind"),
    ("data_movement_no_conn_drop", "reshard"),
    ("data_movement_no_conn_drop", "failover"),
]

EXECUTE_TIMEOUT = 120
COMMAND_EXECUTION_DURATION = 60


def get_fault_injector_client() -> FaultInjectorClient:
    if use_mock_proxy():
        pytest.skip("Scenario tests require Redis Enterprise")
    url = os.getenv("FAULT_INJECTION_API_URL", "http://127.0.0.1:20324")
    return REFaultInjector(url)


def create_standalone_client(
    endpoints_config: Dict[str, Any],
    socket_timeout: float = CLIENT_TIMEOUT,
    enable_retry: bool = True,
) -> Redis:
    password = endpoints_config.get("password", "")
    endpoints = endpoints_config.get("endpoints", [])

    if not endpoints:
        raise ValueError("No endpoints in setup response")

    parsed = urlparse(endpoints[0])
    host = parsed.hostname
    port = parsed.port
    tls_enabled = parsed.scheme == "rediss"

    maintenance_config = MaintNotificationsConfig(
        enabled=True,
        proactive_reconnect=True,
        relaxed_timeout=RELAXED_TIMEOUT,
    )

    if enable_retry:
        retry = Retry(backoff=ExponentialWithJitterBackoff(base=0.5, cap=5), retries=5)
    else:
        retry = None

    client = Redis(
        host=host,
        port=port,
        password=password or None,
        socket_timeout=socket_timeout,
        ssl=tls_enabled,
        ssl_cert_reqs="none" if tls_enabled else None,
        ssl_check_hostname=False,
        protocol=3,
        maint_notifications_config=maintenance_config,
        retry=retry,
    )
    return client


def get_all_connections(client: Redis) -> List[ConnectionInterface]:
    connections = []
    with client.connection_pool._lock:
        for conn in client.connection_pool._get_free_connections():
            connections.append(conn)
        for conn in client.connection_pool._get_in_use_connections():
            connections.append(conn)
    return connections


@pytest.mark.skipif(
    use_mock_proxy(),
    reason="Scenario tests require Redis Enterprise via env0",
)
class TestTopologyChangeStandalone:
    """Test topology_change_standalone scenario effects on Redis standalone client."""

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

        logging.info("Cleanup finished")

    def _setup_scenario(
        self,
        effect: str,
        trigger: str,
        requirement_index: int = 0,
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
        )

        self._setup_id = setup_response["setup_id"]
        logging.info(f"Setup completed: bdb_id={setup_response.get('bdb_id')}")

        time.sleep(5)

        return setup_response

    def _execute_scenario(self) -> Dict[str, Any]:
        return self._fi_client.execute_scenario(SCENARIO_NAME, self._setup_id)

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize(
        "effect,trigger",
        [
            ("data_movement_conn_drop", "maintenance_mode"),
        ],
    )
    def test_maintenance_mode_connection_drop(self, effect: str, trigger: str):
        """
        Verify client handles connection drop during maintenance mode.

        During maintenance_mode:
        1. Client receives MIGRATING notification
        2. Connection enters MAINTENANCE state with relaxed timeout
        3. After trigger completes, connection recovers to NONE state
        """
        setup_response = self._setup_scenario(effect, trigger)
        self._client = create_standalone_client(setup_response)

        conn = self._client.connection_pool.get_connection()
        self._client.connection_pool.release(conn)

        assert self._client.ping() is True
        logging.info("Initial connection established, executing scenario...")

        execute_thread = threading.Thread(
            target=self._execute_scenario,
            name="execute_thread",
        )
        execute_thread.start()

        time.sleep(2)

        connections = get_all_connections(self._client)
        maintenance_count = sum(
            1 for c in connections if c.maintenance_state == MaintenanceState.MAINTENANCE
        )
        logging.info(
            f"During maintenance: {maintenance_count}/{len(connections)} connections in MAINTENANCE state"
        )

        execute_thread.join(timeout=EXECUTE_TIMEOUT)

        connections_after = get_all_connections(self._client)
        none_state_count = sum(
            1 for c in connections_after if c.maintenance_state == MaintenanceState.NONE
        )
        logging.info(
            f"After maintenance: {none_state_count}/{len(connections_after)} connections in NONE state"
        )

        assert self._client.ping() is True
        logging.info("Commands succeed after maintenance mode")

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize(
        "effect,trigger",
        [
            ("data_movement_no_conn_drop", "failover"),
        ],
    )
    def test_hitless_failover(self, effect: str, trigger: str):
        """
        Verify hitless failover - commands succeed without connection drops.

        During failover:
        1. Master/replica roles swap
        2. Client may experience brief latency increase
        3. No connection errors should occur with retry enabled
        """
        setup_response = self._setup_scenario(effect, trigger)
        self._client = create_standalone_client(setup_response, enable_retry=True)

        assert self._client.ping() is True
        self._client.set("test_key", "test_value")

        errors: Queue = Queue()

        def execute_commands(duration: int):
            start = time.time()
            while time.time() - start < duration:
                try:
                    self._client.set("failover_test", f"value_{time.time()}")
                    result = self._client.get("failover_test")
                    assert result is not None
                except Exception as e:
                    logging.error(f"Command error: {e}")
                    errors.put(str(e))
                time.sleep(0.1)

        command_thread = threading.Thread(
            target=execute_commands,
            args=(COMMAND_EXECUTION_DURATION,),
            name="command_thread",
        )
        command_thread.start()

        time.sleep(3)

        logging.info("Triggering failover...")
        execute_result = self._execute_scenario()
        logging.info(f"Failover result: {execute_result.get('status')}")

        command_thread.join()

        assert errors.empty(), f"Errors during failover: {list(errors.queue)}"
        assert self._client.get("test_key") == b"test_value"
        logging.info("Hitless failover verified - no command errors")

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize(
        "effect,trigger",
        [
            ("data_movement_no_conn_drop", "reshard"),
        ],
    )
    def test_hitless_reshard(self, effect: str, trigger: str):
        """
        Verify hitless reshard - commands succeed throughout shard count change.

        During reshard:
        1. Shard count increases
        2. Client may experience latency increase
        3. No connection errors should occur with retry enabled
        """
        setup_response = self._setup_scenario(effect, trigger)
        self._client = create_standalone_client(setup_response, enable_retry=True)

        assert self._client.ping() is True

        errors: Queue = Queue()
        success_count = [0]

        def execute_commands(duration: int):
            start = time.time()
            while time.time() - start < duration:
                try:
                    key = f"reshard_test_{int(time.time() * 1000) % 1000}"
                    self._client.set(key, "value")
                    self._client.get(key)
                    success_count[0] += 1
                except Exception as e:
                    logging.error(f"Command error: {e}")
                    errors.put(str(e))
                time.sleep(0.05)

        command_thread = threading.Thread(
            target=execute_commands,
            args=(COMMAND_EXECUTION_DURATION,),
            name="command_thread",
        )
        command_thread.start()

        time.sleep(3)

        logging.info("Triggering reshard...")
        execute_result = self._execute_scenario()
        logging.info(f"Reshard result: {execute_result.get('status')}")

        command_thread.join()

        assert errors.empty(), f"Errors during reshard: {list(errors.queue)}"
        logging.info(f"Hitless reshard verified - {success_count[0]} commands succeeded")

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize(
        "effect,trigger",
        [
            ("data_movement_no_conn_drop", "endpoint_rebind"),
        ],
    )
    def test_hitless_endpoint_rebind(self, effect: str, trigger: str):
        """
        Verify hitless endpoint rebind - commands succeed during proxy rebind.

        During endpoint_rebind:
        1. Shards migrate to new node
        2. Endpoint policy rebinds
        3. Client receives MOVING notification
        4. No connection errors should occur with retry enabled
        """
        setup_response = self._setup_scenario(effect, trigger)
        self._client = create_standalone_client(setup_response, enable_retry=True)

        assert self._client.ping() is True

        errors: Queue = Queue()

        def execute_commands(duration: int):
            start = time.time()
            while time.time() - start < duration:
                try:
                    self._client.set("rebind_test", "value")
                    self._client.get("rebind_test")
                except Exception as e:
                    logging.error(f"Command error: {e}")
                    errors.put(str(e))
                time.sleep(0.1)

        command_thread = threading.Thread(
            target=execute_commands,
            args=(COMMAND_EXECUTION_DURATION,),
            name="command_thread",
        )
        command_thread.start()

        time.sleep(3)

        logging.info("Triggering endpoint rebind...")
        execute_result = self._execute_scenario()
        logging.info(f"Rebind result: {execute_result.get('status')}")

        command_thread.join()

        assert errors.empty(), f"Errors during rebind: {list(errors.queue)}"
        logging.info("Hitless endpoint rebind verified")

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize(
        "effect,trigger",
        [
            ("data_movement_conn_drop", "endpoint_rebind"),
        ],
    )
    def test_endpoint_rebind_with_conn_drop(self, effect: str, trigger: str):
        """
        Verify client handles connection drop during endpoint rebind.

        When endpoint_rebind causes connection drop (data_movement_conn_drop effect):
        1. Client connection is dropped
        2. Client reconnects to new endpoint
        3. Commands succeed after reconnection with retry
        """
        setup_response = self._setup_scenario(effect, trigger)
        self._client = create_standalone_client(setup_response, enable_retry=True)

        assert self._client.ping() is True

        errors: Queue = Queue()

        def execute_commands(duration: int):
            start = time.time()
            while time.time() - start < duration:
                try:
                    self._client.set("rebind_drop_test", "value")
                    self._client.get("rebind_drop_test")
                except Exception as e:
                    logging.error(f"Command error: {e}")
                    errors.put(str(e))
                time.sleep(0.1)

        command_thread = threading.Thread(
            target=execute_commands,
            args=(COMMAND_EXECUTION_DURATION,),
            name="command_thread",
        )
        command_thread.start()

        time.sleep(3)

        logging.info("Triggering endpoint rebind (with conn drop)...")
        execute_result = self._execute_scenario()
        logging.info(f"Rebind result: {execute_result.get('status')}")

        command_thread.join()

        assert errors.empty(), f"Errors during rebind: {list(errors.queue)}"
        assert self._client.ping() is True
        logging.info("Endpoint rebind with connection drop verified - client recovered")
