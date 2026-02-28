#include "runtime/thread_pool.h"

ThreadPool::ThreadPool(size_t n)
: stop_(false), active_(0)
{
    for(size_t i=0;i<n;++i)
    {
        workers_.emplace_back([this](){
            while(true)
            {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(m_);
                    cv_.wait(lock,[this]{
                        return stop_ || !tasks_.empty();
                    });

                    if(stop_ && tasks_.empty())
                        return;

                    task = std::move(tasks_.front());
                    tasks_.pop();
                    active_++;
                }

                task();

                {
                    std::unique_lock<std::mutex> lock(m_);
                    active_--;
                    if(tasks_.empty() && active_==0)
                        cv_.notify_all();
                }
            }
        });
    }
}

void ThreadPool::enqueue(std::function<void()> job)
{
    {
        std::unique_lock<std::mutex> lock(m_);
        tasks_.push(job);
    }
    cv_.notify_one();
}

void ThreadPool::wait()
{
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock,[this]{
        return tasks_.empty() && active_==0;
    });
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(m_);
        stop_=true;
    }
    cv_.notify_all();

    for(auto& t:workers_)
        t.join();
}