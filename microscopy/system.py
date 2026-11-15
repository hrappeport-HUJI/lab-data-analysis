"""Small operating-system helpers that work on macOS, Linux and Windows."""
import ctypes
import os
import sys


def physical_ram_bytes():
    """Total physical memory of this computer, in bytes."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullTotalPhys)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def lower_priority(nice=10):
    """Run this process at lower CPU priority so the computer stays usable. Best effort."""
    try:
        if sys.platform == "win32":
            below_normal = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # Without these, ctypes truncates the 64-bit pseudo-handle to int and the call fails.
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel32.SetPriorityClass.restype = ctypes.c_int
            if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.nice(int(nice))
    except (OSError, AttributeError):
        pass
