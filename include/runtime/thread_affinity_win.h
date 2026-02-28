#pragma once
#ifdef _WIN32
void set_thread_affinity_win(int core_id);
#endif