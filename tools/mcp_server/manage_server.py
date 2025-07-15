#!/usr/bin/env python3
"""
Manage the MCP server - start, stop, and check status.
"""

import subprocess
import sys
import signal
import time
from pathlib import Path

def get_server_pid():
    """Get the PID of the running MCP server."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if "tools/mcp_server/main.py" in line and "grep" not in line:
                parts = line.split()
                return int(parts[1])
        
        return None
    except:
        return None

def start_server():
    """Start the MCP server in the background."""
    pid = get_server_pid()
    if pid:
        print(f"✅ MCP server is already running (PID: {pid})")
        return True
    
    print("🚀 Starting MCP server in background...")
    
    try:
        process = subprocess.Popen(
            [
                "/Users/ivaylo.kiryazov/redis-py-test-infra/.venv/bin/python",
                "tools/mcp_server/main.py",
                "--debug"
            ],
            cwd="/Users/ivaylo.kiryazov/redis-py-test-infra",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if it's running
        pid = get_server_pid()
        if pid:
            print(f"✅ MCP server started successfully (PID: {pid})")
            return True
        else:
            print("❌ Failed to start MCP server")
            return False
    
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def stop_server():
    """Stop the MCP server."""
    pid = get_server_pid()
    if not pid:
        print("ℹ️  MCP server is not running")
        return True
    
    print(f"🛑 Stopping MCP server (PID: {pid})")
    
    try:
        # Try graceful shutdown first
        subprocess.run(["kill", str(pid)], check=True)
        
        # Wait a bit for graceful shutdown
        time.sleep(2)
        
        # Check if still running
        if get_server_pid():
            print("🔨 Force killing server...")
            subprocess.run(["kill", "-9", str(pid)], check=True)
            time.sleep(1)
        
        if not get_server_pid():
            print("✅ MCP server stopped successfully")
            return True
        else:
            print("❌ Failed to stop MCP server")
            return False
    
    except Exception as e:
        print(f"❌ Error stopping server: {e}")
        return False

def server_status():
    """Check the status of the MCP server."""
    pid = get_server_pid()
    if pid:
        print(f"✅ MCP server is running (PID: {pid})")
        return True
    else:
        print("❌ MCP server is not running")
        return False

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python manage_server.py [start|stop|status|restart]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "start":
        success = start_server()
    elif command == "stop":
        success = stop_server()
    elif command == "status":
        success = server_status()
    elif command == "restart":
        print("🔄 Restarting MCP server...")
        stop_server()
        time.sleep(1)
        success = start_server()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python manage_server.py [start|stop|status|restart]")
        sys.exit(1)
    
    if success:
        print("\n📋 Current status:")
        server_status()
        
        if command in ["start", "restart"]:
            print("\n🎯 Next steps:")
            print("1. The MCP server is now running in the background")
            print("2. In Cursor, the tools should now be available")
            print("3. Check Cursor's MCP connection status")
            print("4. If still not working, try restarting Cursor")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 