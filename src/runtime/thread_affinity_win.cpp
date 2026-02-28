#include "runtime/thread_affinity_win.h"
#ifdef _WIN32
#include <windows.h>

void set_thread_affinity_win(int core_id)
{
    HANDLE th = GetCurrentThread();
    DWORD_PTR mask = (DWORD_PTR)1ULL << (core_id & 63);
    SetThreadAffinityMask(th, mask);
}
#endif