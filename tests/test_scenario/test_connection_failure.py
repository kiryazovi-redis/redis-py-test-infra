"""
Tests for connection_failure scenario.

Tests connection failure handling using the 4-phase scenario lifecycle:
discovery -> setup -> execute -> teardown.

Effects tested:
- connection_dropped: Existing connections are dropped (proxy_failure with restart)
- connection_refused: New connections are refused (proxy_failure with stop)
- connection_timeout: Connections time out (database_restart)
"""

import logging
import os
import threading
import time
from queue import Queue
from typing import Any, Dict
from urllib.parse import urlparse

import pytest

from redis import Redis
from redis.backoff import ExponentialWithJitterBackoff
from redis.exceptions import ConnectionError, TimeoutError
from redis.maint_notifications import MaintNotificationsConfig
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

SCENARIO_NAME = "connection_failure"

EXECUTE_TIMEOUT = 120
COMMAND_EXECUTION_DURATION = 30
RECOVERY_WAIT_TIME = 45


def get_fault_injector_client() -> FaultInjectorClient:
    if use_mock_proxy():
        pytest.skip("Scenario tests require Redis Enterprise")
    url = os.getenv("FAULT_INJECTION_API_URL", "http://127.0.0.1:20324")
    return REFaultInjector(url)


def create_redis_client(
    endpoints_config: Dict[str, Any],
    socket_timeout: float = CLIENT_TIMEOUT,
    enable_retry: bool = True,
    retry_count: int = 5,
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
        retry = Retry(
            backoff=ExponentialWithJitterBackoff(base=0.5, cap=5),
            retries=retry_count,
        )
    else:
        retry = None

    client = Redis(
        host=host,
        port=port,
        password=password or None,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_timeout,
        ssl=tls_enabled,
        ssl_cert_reqs="none" if tls_enabled else None,
        ssl_check_hostname=False,
        protocol=3,
        maint_notifications_config=maintenance_config,
        retry=retry,
    )
    return client


@pytest.mark.skipif(
    use_mock_proxy(),
    reason="Scenario tests require Redis Enterprise via env0",
)
class TestConnectionFailure:
    """Test connection_failure scenario effects on Redis client."""

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

        time.sleep(RECOVERY_WAIT_TIME)
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
    def test_connection_dropped_with_retry_standalone(self):
        """
        Verify standalone client retries and recovers after connection drop.

        proxy_failure (restart) causes:
        1. Existing connections to be dropped
        2. Client should retry and reconnect
        3. Commands should eventually succeed
        """
        setup_response = self._setup_scenario(
            effect="connection_dropped",
            trigger="proxy_failure",
        )
        self._client = create_redis_client(setup_response, enable_retry=True)

        assert self._client.ping() is True
        self._client.set("drop_test_key", "initial_value")
        logging.info("Initial connection established")

        errors: Queue = Queue()
        success_after_failure = [False]

        def execute_commands(duration: int):
            start = time.time()
            failure_detected = False
            while time.time() - start < duration:
                try:
                    self._client.set("drop_test", f"value_{time.time()}")
                    self._client.get("drop_test")
                    if failure_detected:
                        success_after_failure[0] = True
                        logging.info("Commands succeeding after connection drop")
                except (ConnectionError, TimeoutError) as e:
                    failure_detected = True
                    logging.info(f"Connection failure detected: {type(e).__name__}")
                except Exception as e:
                    errors.put(f"Unexpected error: {e}")
                time.sleep(0.2)

        command_thread = threading.Thread(
            target=execute_commands,
            args=(COMMAND_EXECUTION_DURATION,),
            name="command_thread",
        )
        command_thread.start()

        time.sleep(3)

        logging.info("Triggering connection drop (proxy restart)...")
        execute_result = self._execute_scenario()
        logging.info(f"Execute result: {execute_result.get('status')}")

        command_thread.join()

        assert errors.empty(), f"Unexpected errors: {list(errors.queue)}"
        assert success_after_failure[0], "Client should recover and succeed after connection drop"
        assert self._client.ping() is True
        logging.info("Connection drop recovery verified")

    @pytest.mark.timeout(300)
    def test_connection_refused_behavior_standalone(self):
        """
        Verify client behavior when connections are refused (proxy stopped).

        proxy_failure (stop) causes:
        1. All new connection attempts to be refused
        2. Client should raise ConnectionError
        3. After teardown (proxy restart), client should reconnect
        """
        setup_response = self._setup_scenario(
            effect="connection_refused",
            trigger="proxy_failure",
        )
        self._client = create_redis_client(setup_response, enable_retry=False)

        assert self._client.ping() is True
        logging.info("Initial connection established")

        logging.info("Triggering connection refused (proxy stop)...")
        execute_result = self._execute_scenario()
        logging.info(f"Execute result: {execute_result.get('status')}")

        time.sleep(2)

        connection_error_raised = False
        for _ in range(3):
            try:
                self._client.close()
                self._client = create_redis_client(
                    setup_response,
                    enable_retry=False,
                    socket_timeout=3,
                )
                self._client.ping()
            except (ConnectionError, TimeoutError, OSError):
                connection_error_raised = True
                break
            time.sleep(1)

        assert connection_error_raised, "Expected ConnectionError when proxy is stopped"
        logging.info("Connection refused verified - ConnectionError raised as expected")

    @pytest.mark.timeout(300)
    def test_connection_timeout_handling_standalone(self):
        """
        Verify client handles timeout caused by database_restart.

        database_restart causes:
        1. All database shards to be killed
        2. Connections to timeout
        3. After restart, client should reconnect
        """
        setup_response = self._setup_scenario(
            effect="connection_timeout",
            trigger="database_restart",
        )
        self._client = create_redis_client(
            setup_response,
            enable_retry=True,
            retry_count=10,
            socket_timeout=5,
        )

        assert self._client.ping() is True
        logging.info("Initial connection established")

        errors: Queue = Queue()
        timeout_detected = [False]
        recovered = [False]

        def execute_commands(duration: int):
            start = time.time()
            while time.time() - start < duration:
                try:
                    self._client.ping()
                    if timeout_detected[0]:
                        recovered[0] = True
                        logging.info("Recovered from timeout")
                except TimeoutError:
                    timeout_detected[0] = True
                    logging.info("Timeout detected during database restart")
                except ConnectionError as e:
                    if timeout_detected[0]:
                        logging.info(f"Connection error during recovery: {e}")
                    else:
                        timeout_detected[0] = True
                except Exception as e:
                    errors.put(f"Unexpected error: {e}")
                time.sleep(0.5)

        command_thread = threading.Thread(
            target=execute_commands,
            args=(60,),
            name="command_thread",
        )
        command_thread.start()

        time.sleep(3)

        logging.info("Triggering database restart...")
        execute_result = self._execute_scenario()
        logging.info(f"Execute result: {execute_result.get('status')}")

        command_thread.join()

        assert errors.empty(), f"Unexpected errors: {list(errors.queue)}"
        assert timeout_detected[0], "Timeout should be detected during database restart"
        assert recovered[0], "Client should recover after database restart"
        logging.info(
            f"Timeout handling verified - timeout_detected={timeout_detected[0]}, recovered={recovered[0]}"
        )

    @pytest.mark.timeout(300)
    def test_concurrent_commands_during_connection_failure(self):
        """
        Verify multiple threads can execute commands through connection failure.

        Multiple threads executing commands should:
        1. All eventually succeed with retry
        2. No unexpected errors
        """
        setup_response = self._setup_scenario(
            effect="connection_dropped",
            trigger="proxy_failure",
        )
        self._client = create_redis_client(
            setup_response,
            enable_retry=True,
            retry_count=10,
        )

        assert self._client.ping() is True

        errors: Queue = Queue()
        thread_count = 5

        def execute_commands(thread_id: int, duration: int):
            start = time.time()
            success_count = 0
            while time.time() - start < duration:
                try:
                    key = f"thread_{thread_id}_key"
                    self._client.set(key, f"value_{time.time()}")
                    self._client.get(key)
                    success_count += 1
                except (ConnectionError, TimeoutError):
                    pass
                except Exception as e:
                    errors.put(f"Thread {thread_id}: {e}")
                time.sleep(0.1)
            logging.info(f"Thread {thread_id}: {success_count} successful operations")

        threads = []
        for i in range(thread_count):
            t = threading.Thread(
                target=execute_commands,
                args=(i, COMMAND_EXECUTION_DURATION),
                name=f"command_thread_{i}",
            )
            threads.append(t)
            t.start()

        time.sleep(3)

        logging.info("Triggering connection drop during concurrent commands...")
        execute_result = self._execute_scenario()
        logging.info(f"Execute result: {execute_result.get('status')}")

        for t in threads:
            t.join()

        assert errors.empty(), f"Errors during concurrent execution: {list(errors.queue)}"
        assert self._client.ping() is True
        logging.info("Concurrent command execution through failure verified")
