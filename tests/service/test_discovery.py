import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from support.service import wait_until

from praxis_agent.service.discovery import ServiceLock, discover, ensure_service


def test_os_lock_blocks_other_process_before_database_open(tmp_path):
    lock = ServiceLock(tmp_path)
    assert lock.acquire()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "praxis_agent.service", "--data-dir", str(tmp_path)],
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "state.sqlite3").exists()
        assert not (tmp_path / "service.json").exists()
    finally:
        lock.close()
    second = ServiceLock(tmp_path)
    assert second.acquire()
    second.close()


def test_simultaneous_launchers_discover_one_detached_service_and_restart(tmp_path):
    record = None
    try:
        with ThreadPoolExecutor(4) as pool:
            records = list(
                pool.map(lambda _: ensure_service(tmp_path, timeout=25), range(4))
            )
        record = records[0]
        assert {item["instance_id"] for item in records} == {record["instance_id"]}
        assert {item["pid"] for item in records} == {record["pid"]}
        assert discover(tmp_path)["instance_id"] == record["instance_id"]
        if os.name == "nt":
            # Probe the actual service interpreter, including the virtualenv
            # redirector, rather than just checking Popen's creation flags.
            probe = subprocess.run(
                [sys.executable, "-c", """
import ctypes
import sys
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
user = ctypes.WinDLL('user32', use_last_error=True)
kernel.GetConsoleWindow.restype = ctypes.c_void_p
user.IsWindowVisible.argtypes = [ctypes.c_void_p]
kernel.FreeConsole()
attached = kernel.AttachConsole(int(sys.argv[1]))
error = ctypes.get_last_error()
if attached:
    window = kernel.GetConsoleWindow()
    visible = bool(window and user.IsWindowVisible(window))
    kernel.FreeConsole()
    assert not visible, 'Service created a visible console window'
else:
    assert error == 6, error  # ERROR_INVALID_HANDLE: no console to attach to.
""", str(record["pid"])],
                capture_output=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            assert probe.returncode == 0, probe.stderr.decode(errors="replace")
        # This PID comes only from the service launched in this test directory.
        os.kill(record["pid"], signal.SIGTERM)
        wait_until(lambda: discover(tmp_path) is None, timeout=10)
        previous = record["instance_id"]
        record = ensure_service(tmp_path, timeout=25)
        assert record["instance_id"] != previous
        assert record["url"].startswith("http://127.0.0.1:")
    finally:
        if record:
            current = discover(tmp_path)
            if current and current["instance_id"] == record["instance_id"]:
                os.kill(record["pid"], signal.SIGTERM)
