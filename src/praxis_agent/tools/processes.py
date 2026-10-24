"""Lifetime of a command and its descendants (not an execution sandbox)."""
from __future__ import annotations

import os
import signal
import subprocess
from threading import Lock


class CommandProcess:
    def __init__(self, argv: list[str], **kwargs: object) -> None:
        self.job = None
        self.process = None
        self._lock = Lock()
        self._close_lock = Lock()
        self._closed = False
        if os.name == "nt":
            self._start_windows(argv, kwargs)
        else:
            self.process = subprocess.Popen(argv, start_new_session=True, **kwargs)

    def _start_windows(self, argv: list[str], kwargs: dict[str, object]) -> None:
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t),
                        ("maximum", ctypes.c_size_t), ("active", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", ctypes.c_ulonglong * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel = kernel
        self.job = kernel.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            # Assign before running any user code, so even immediately spawned children
            # belong to the job. Popen closes the primary thread handle on Windows;
            # NtResumeProcess resumes the suspended process via its retained handle.
            self.process = subprocess.Popen(
                argv, creationflags=subprocess.CREATE_NO_WINDOW | 0x4, **kwargs,
            )
            if not kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            resume = ctypes.WinDLL("ntdll").NtResumeProcess
            resume.argtypes = [wintypes.HANDLE]
            resume.restype = ctypes.c_long
            status = resume(int(self.process._handle))
            if status != 0:
                raise OSError(f"could not resume command process: NTSTATUS {status}")
        except BaseException:
            if self.process is not None:
                self.process.kill()
            self.close()
            raise

    def request_stop(self) -> None:
        """Signal the process tree without waiting on the UI/cancellation thread."""
        with self._lock:
            if self._closed:
                return
            if os.name == "nt":
                if self.job:
                    if not self.kernel.TerminateJobObject(self.job, 1):
                        import ctypes
                        raise ctypes.WinError(ctypes.get_last_error())
            elif self.process is not None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self.request_stop()
            with self._lock:
                if os.name == "nt" and self.job:
                    self.kernel.CloseHandle(self.job)
                    self.job = None
            if self.process is not None:
                self.process.wait(timeout=10)
            with self._lock:
                self._closed = True
