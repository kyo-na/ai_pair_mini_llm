#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <condition_variable>

class ThreadPool {
public:
    ThreadPool(size_t n);
    ~ThreadPool();

    void enqueue(std::function<void()> job);
    void wait();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex m_;
    std::condition_variable cv_;
    bool stop_;
    size_t active_;
};