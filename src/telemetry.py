"""Read Windows system memory without loading torch."""
import ctypes
import os


def memory_snapshot():
    if os.name != 'nt':
        return {}
    from ctypes import wintypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD),
                    ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong),
                    ('total_page', ctypes.c_ulonglong), ('available_page', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
                    ('available_extended', ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    result = {}
    if kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        result.update(system_ram_total_mib=round(status.total_physical / 1024**2, 1),
                      system_ram_available_mib=round(status.available_physical / 1024**2, 1))
    return result
