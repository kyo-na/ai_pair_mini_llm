#include <windows.h>

void set_thread_affinity(int core_id)
{
    HANDLE thread = GetCurrentThread();

    DWORD_PTR mask = 1ULL << core_id;

    SetThreadAffinityMask(thread, mask);
}