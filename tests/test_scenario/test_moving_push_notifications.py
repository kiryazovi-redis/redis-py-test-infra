"""Tests for Redis Enterprise moving push notifications with real cluster operations."""

import json
import os
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import pytest

from redis import Redis
from redis.maintenance_events import MaintenanceEventsConfig, MaintenanceState
from tests.test_scenario.fault_injector_client import (ActionRequest,
                                                       ActionType,
                                                       FaultInjectorClient)


class TestMovingPushNotifications:
    """Test Redis Enterprise moving push notifications with real cluster
    operations."""

    @pytest.fixture(autouse=True)
    def setup(self, request):
        """Setup test environment."""
        self.endpoint_name = request.config.getoption("--endpoint-name")
        if not self.endpoint_name:
            pytest.skip("Endpoint name not provided via --endpoint-name")

        # Load endpoints configuration
        try:
            self.endpoint_config = self._get_full_endpoint_config(self.endpoint_name)
        except (FileNotFoundError, ValueError) as e:
            pytest.skip(f"Cannot load endpoint configuration: {e}")

        # Setup fault injector client for rladmin commands
        fault_injector_url = self._get_fault_injector_url()
        self.fault_injector = FaultInjectorClient(fault_injector_url)

        # Setup Redis client with maintenance events enabled
        self.redis_client = self._create_redis_client_with_maintenance_events()

        yield

        if hasattr(self, "redis_client"):
            self.redis_client.close()

    def _get_full_endpoint_config(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Get the full endpoint configuration from endpoints.json.

        Args:
            endpoint_name: Name of the endpoint to load

        Returns:
            Dict containing the full endpoint configuration
        """
        endpoints_config_path = os.getenv("REDIS_ENDPOINTS_CONFIG_PATH")
        if not (endpoints_config_path and os.path.exists(endpoints_config_path)):
            raise FileNotFoundError(
                f"Endpoints config file not found: {endpoints_config_path}"
            )

        try:
            with open(endpoints_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if endpoint_name not in data:
                    raise ValueError(
                        f"Endpoint '{endpoint_name}' not found in configuration"
                    )
                return data[endpoint_name]
        except Exception as e:
            raise ValueError(
                f"Failed to load endpoint '{endpoint_name}' from config file: "
                f"{endpoints_config_path}"
            ) from e

    def _get_fault_injector_url(self) -> str:
        """
        Get fault injector URL from environment variable with default.

        Returns:
            str: Fault injector URL
        """
        return os.getenv("FAULT_INJECTION_API_URL", "http://127.0.0.1:20324")

    def _create_redis_client_with_maintenance_events(self) -> Redis:
        """Create Redis client with maintenance events enabled."""

        # Get credentials from the configuration
        username = self.endpoint_config.get("username")
        password = self.endpoint_config.get("password")

        # Parse host and port from endpoints URL
        endpoints = self.endpoint_config.get("endpoints", [])
        if not endpoints:
            raise ValueError("No endpoints found in configuration")

        parsed = urlparse(endpoints[0])
        host = parsed.hostname
        port = parsed.port

        if not host:
            raise ValueError(f"Could not parse host from endpoint URL: {endpoints[0]}")

        print(f"Connecting to Redis Enterprise: {host}:{port} with user: {username}")

        # Configure maintenance events
        maintenance_config = MaintenanceEventsConfig(
            enabled=True, proactive_reconnect=True, relax_timeout=30
        )

        # Create Redis client with maintenance events config
        # This will automatically create the MaintenanceEventPoolHandler
        client = Redis(
            host=host,
            port=port,
            username=username,
            password=password,
            protocol=3,  # RESP3 required for push notifications
            maintenance_events_config=maintenance_config,
        )
        print("Redis client created with maintenance events enabled")
        print(f"Protocol: {client.connection_pool.get_protocol()}")
        maintenance_handler_exists = client.maintenance_events_pool_handler is not None
        print(f"Maintenance events pool handler: {maintenance_handler_exists}")

        return client

    def _check_for_push_notifications(self):
        """Actively check for pending push notifications on all connections."""
        pool = self.redis_client.connection_pool

        # Check if there are any connections that can be read from
        connections_to_check = []

        # Get a connection to check for push notifications
        try:
            test_conn = pool.get_connection()
            connections_to_check.append(test_conn)
        except Exception as e:
            print(f"Could not get connection for push check: {e}")
            return

        try:
            for conn in connections_to_check:
                print(f"Checking connection {conn} for push notifications...")

                # Check if there's data available to read
                checks = 0
                # Limit checks to avoid infinite loop
                while conn.can_read(timeout=0.1) and checks < 5:
                    checks += 1
                    try:
                        # reading is important, it triggers the push notification
                        push_response = conn.read_response(push_request=True)
                        print(
                            f"Read response: {push_response}. The result doesn't "
                            f"concern us, it just triggers the push notification"
                        )
                    except Exception as e:
                        print(f"Error reading push notification: {e}")
                        break

        finally:
            # Release the connection back to the pool
            for conn in connections_to_check:
                try:
                    pool.release(conn)
                except Exception as e:
                    print(f"Error releasing connection: {e}")

    def _monitor_maintenance_state_during_operation(
        self, action_id: str, operation_name: str, timeout: int = 120
    ):
        """Monitor maintenance state changes while a Redis Enterprise operation
        is running."""

        start_time = time.time()
        check_interval = 3  # Check more frequently during operations
        maintenance_states_found = set()
        print(
            f"Starting maintenance state monitoring for {operation_name} "
            f"operation {action_id}"
        )

        while time.time() - start_time < timeout:
            try:
                self._check_for_push_notifications()
                pool = self.redis_client.connection_pool

                # Check available connections
                if hasattr(pool, "_available_connections"):
                    for conn in pool._available_connections:
                        state = getattr(
                            conn, "maintenance_state", MaintenanceState.NONE
                        )
                        if state != MaintenanceState.NONE:
                            maintenance_states_found.add(state)

                print(f"Maintenance states found: {maintenance_states_found}")

                # Check operation status first
                status_result = self.fault_injector.get_action_status(action_id)
                operation_status = status_result.get("status", "unknown")

                completed_statuses = [
                    "failed",
                    "finished",
                ]
                if operation_status in completed_statuses:
                    print(
                        f"Operation {action_id} completed with status: "
                        f"{operation_status}"
                    )
                    break

                time.sleep(check_interval)

            except Exception as e:
                print(f"Error checking operation status: {e}")
                time.sleep(check_interval)
        else:
            print(f"Timeout waiting for {operation_name} operation {action_id}")

        print(
            f"Maintenance states detected during {operation_name}: "
            f"{maintenance_states_found}"
        )
        return maintenance_states_found

    def _get_cluster_nodes_info(self) -> Dict[str, Any]:
        """Get cluster nodes information from Redis Enterprise."""
        try:
            # Use rladmin status to get node information
            action = ActionRequest(
                action_type=ActionType.EXECUTE_RLADMIN_COMMAND,
                parameters={"rladmin_command": "status", "bdb_id": self.endpoint_config.get("bdb_id")},
            )
            result = self.fault_injector.trigger_action(action)
            start_time = time.time()

            while time.time() - start_time < 30:
                time.sleep(5)
                something = self.fault_injector.get_action_status(result.get("action_id"))
                status = something.get("status")

                print(f"Cluster nodes info: {status}")
                completed_statuses = [
                    "failed",
                    "finished",
                ]
                if status in completed_statuses:
                    print(f"Cluster nodes info: {something}")
                    break
            return something
        except Exception as e:
            pytest.fail(f"Failed to get cluster nodes info: {e}")

    def _find_target_node_and_empty_node(self) -> Tuple[str, str]:
        """Find the node with master shards and the node with no shards.
        
        Returns:
            tuple: (target_node, empty_node) where target_node has master shards 
                   and empty_node has no shards
        """
        cluster_info = self._get_cluster_nodes_info()
        output = cluster_info.get("output", {}).get("output", "")
        
        if not output:
            raise ValueError("No cluster status output found")
        
        # Parse the sections to find nodes with master shards and nodes with no shards
        lines = output.split('\n')
        shards_section_started = False
        nodes_section_started = False
        
        # Get all node IDs from CLUSTER NODES section
        all_nodes = set()
        nodes_with_shards = set()
        master_nodes = set()
        
        for line in lines:
            line = line.strip()
            
            # Start of CLUSTER NODES section
            if line.startswith("CLUSTER NODES:"):
                nodes_section_started = True
                continue
            elif line.startswith("DATABASES:"):
                nodes_section_started = False
                continue
            elif nodes_section_started and line and not line.startswith("NODE:ID"):
                # Parse node line: node:1  master 10.0.101.206 ... (ignore the role)
                parts = line.split()
                if len(parts) >= 1:
                    node_id = parts[0].replace('*', '')  # Remove * prefix if present
                    all_nodes.add(node_id)
            
            # Start of SHARDS section - only care about shard roles here
            if line.startswith("SHARDS:"):
                shards_section_started = True
                continue
            elif shards_section_started and line.startswith("DB:ID"):
                continue
            elif shards_section_started and line and not line.startswith("ENDPOINTS:"):
                # Parse shard line: db:1  m-standard  redis:1  node:2  master  0-8191  1.4MB  OK
                parts = line.split()
                if len(parts) >= 5:
                    node_id = parts[3]  # node:2
                    shard_role = parts[4]  # master/slave - this is what matters
                    
                    nodes_with_shards.add(node_id)
                    if shard_role == "master":
                        master_nodes.add(node_id)
            elif line.startswith("ENDPOINTS:") or not line:
                shards_section_started = False
        
        # Find empty node (node with no shards)
        empty_nodes = all_nodes - nodes_with_shards
        
        print(f"All nodes: {all_nodes}")
        print(f"Nodes with shards: {nodes_with_shards}")
        print(f"Master nodes: {master_nodes}")
        print(f"Empty nodes: {empty_nodes}")
        
        if not empty_nodes:
            raise ValueError("No empty nodes (nodes without shards) found")
        
        if not master_nodes:
            raise ValueError("No nodes with master shards found")
        
        # Return the first available empty node and master node (numeric part only)
        empty_node = next(iter(empty_nodes)).split(':')[1]  # node:1 -> 1
        target_node = next(iter(master_nodes)).split(':')[1]  # node:2 -> 2
        
        return target_node, empty_node

    def _find_endpoint_for_bind(self) -> str:
        """Find the endpoint ID from cluster status.
        
        Returns:
            str: The endpoint ID (e.g., "1:1")
        """
        cluster_info = self._get_cluster_nodes_info()
        output = cluster_info.get("output", {}).get("output", "")
        
        if not output:
            raise ValueError("No cluster status output found")
        
        # Parse the ENDPOINTS section to find endpoint ID
        lines = output.split('\n')
        endpoints_section_started = False
        
        for line in lines:
            line = line.strip()
            
            # Start of ENDPOINTS section
            if line.startswith("ENDPOINTS:"):
                endpoints_section_started = True
                continue
            elif line.startswith("SHARDS:"):
                endpoints_section_started = False
                break
            elif endpoints_section_started and line and not line.startswith("DB:ID"):
                # Parse endpoint line: db:1  m-standard  endpoint:1:1  node:2  single  No
                parts = line.split()
                if len(parts) >= 3:
                    endpoint_full = parts[2]  # endpoint:1:1
                    if endpoint_full.startswith("endpoint:"):
                        endpoint_id = endpoint_full.replace("endpoint:", "")  # 1:1
                        return endpoint_id
        
        raise ValueError("No endpoint ID found in cluster status")

    def _execute_rladmin_migrate(self, target_node: str, empty_node: str):
        """Execute rladmin migrate command and wait for completion."""
        command = f"migrate node {target_node} all_shards target_node {empty_node}"

        # Get bdb_id from endpoint configuration
        bdb_id = self.endpoint_config.get("bdb_id")

        print(f"Executing rladmin command: {command} with bdb_id: {bdb_id}")

        try:
            # Correct parameter format for fault injector
            parameters = {
                "bdb_id": bdb_id,
                "rladmin_command": command,  # Just the command without "rladmin" prefix
            }

            print(f"Using rladmin_command parameter: {parameters}")

            action = ActionRequest(
                action_type=ActionType.EXECUTE_RLADMIN_COMMAND, parameters=parameters
            )

            result = self.fault_injector.trigger_action(action)
            print(
                f"Migrate command {command} with parameters {parameters} trigger result: {result}"
            )

            action_id = result.get("action_id")

            if action_id:
                print(f"Migrate command triggered with action_id: {action_id}")

                states = self._monitor_maintenance_state_during_operation(
                    action_id, "migrate", timeout=120
                )
            else:
                print(f"Warning: No action_id found in response: {result}")
            return states
        except Exception as e:
            pytest.fail(f"Failed to execute rladmin migrate: {e}")

    def _execute_rladmin_bind_endpoint(self, endpoint_id: str):
        """Execute rladmin bind endpoint command and wait for completion."""
        command = f"bind endpoint {endpoint_id} policy single"

        bdb_id = self.endpoint_config.get("bdb_id")

        print(f"Executing rladmin command: {command} with bdb_id: {bdb_id}")

        try:
            parameters = {
                "rladmin_command": command,  # Just the command without "rladmin" prefix
                "bdb_id": bdb_id,
            }

            action = ActionRequest(
                action_type=ActionType.EXECUTE_RLADMIN_COMMAND, parameters=parameters
            )

            result = self.fault_injector.trigger_action(action)
            print(
                f"Migrate command {command} with parameters {parameters} trigger result: {result}"
            )

            action_id = result.get("action_id") 

            if action_id:
                print(f"Bind command triggered with action_id: {action_id}")
                states = self._monitor_maintenance_state_during_operation(
                    action_id, "bind", timeout=60
                )
            else:
                print(f"Warning: No action_id found in response: {result}")
            return states
        except Exception as e:
            pytest.fail(f"Failed to execute rladmin bind endpoint: {e}")

    @pytest.mark.timeout(300)  # 5 minutes timeout for this test
    def test_receive_moving_push_notification(self):
        """
        Test the push notifications are received when executing cluster operations.

        """

        try:
            target_node, empty_node = self._find_target_node_and_empty_node()
            print(f"Using target_node: {target_node}, empty_node: {empty_node}")
        except Exception as e:
            pytest.fail(f"Failed to find target and empty nodes: {e}")
        total_result = []

        print("Executing rladmin migrate command...")
        try:
            migrate_result = self._execute_rladmin_migrate(target_node, empty_node)
            print(f"Migrate command result: {migrate_result}")
            total_result.append(migrate_result)
        except Exception as e:
            pytest.fail(f"Failed to execute rladmin migrate: {e}")

        try:
            endpoint_id = self._find_endpoint_for_bind()
            print(f"Using endpoint: {endpoint_id}")
        except Exception as e:
            pytest.fail(f"Failed to find endpoint for bind operation: {e}")

        print("Executing rladmin bind endpoint command...")
        try:
            bind_result = self._execute_rladmin_bind_endpoint(endpoint_id)
            print(f"Bind command result: {bind_result}")
            total_result.append(bind_result)
        except Exception as e:
            pytest.fail(f"Failed to execute rladmin bind endpoint: {e}")
        
        all_states = set()
        for states_set in total_result:
            all_states.update(states_set)
        
        assert MaintenanceState.MOVING in all_states
        assert MaintenanceState.MIGRATING in all_states

        print("Cleaning up... Technically we don't need to do this due to dynamic node selection, but it's good practice")
        try:
            self._execute_rladmin_migrate(empty_node, target_node)
        except Exception as e:
            pytest.fail(f"Failed to execute rladmin migrate: {e}")

        try:
            self._execute_rladmin_bind_endpoint(endpoint_id)
        except Exception as e:
            pytest.fail(f"Failed to execute rladmin bind endpoint: {e}")