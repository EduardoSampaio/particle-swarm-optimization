#pragma once

#include <time.h>

namespace util {

    struct gpu_timer {
#ifdef __CUDACC__
        gpu_timer() {
            cudaEventCreate(&m_start);
            cudaEventCreate(&m_end);
            cudaEventRecord(m_start, 0);
        }

        float milliseconds_elapsed() {
            float elapsed_time;
            cudaEventRecord(m_end, 0);
            cudaEventSynchronize(m_end);
            cudaEventElapsedTime(&elapsed_time, m_start, m_end);
            return elapsed_time;
        }

        float seconds_elapsed() {
            return milliseconds_elapsed() / 1000.0;
        }

    protected:
        cudaEvent_t m_start, m_end;
#endif // __CUDACC__
    };

    struct cpu_timer {
    public:
        cpu_timer() {
            t1 = ClockGetTime();
        }

        double nanoseconds_elapsed() {
            t2 = ClockGetTime();
            return diff(t1, t2).tv_sec * 1000000000LL + diff(t1, t2).tv_nsec;
        }

        double microseconds_elapsed() {
            return nanoseconds_elapsed() / 1000.0;
        }

        double milliseconds_elapsed() {
            return microseconds_elapsed() / 1000.0;
        }

        double seconds_elapsed() {
            return milliseconds_elapsed() / 1000.0;
        }

    protected:
        timespec diff(timespec start, timespec end) {
            timespec temp;
            if ((end.tv_nsec - start.tv_nsec) < 0) {
                temp.tv_sec = end.tv_sec - start.tv_sec - 1;
                temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
            }
            else {
                temp.tv_sec = end.tv_sec - start.tv_sec;
                temp.tv_nsec = end.tv_nsec - start.tv_nsec;
            }
            return temp;
        }

        // Nano second precision CPU clock
        timespec ClockGetTime() {
            timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            return ts; //(uint64_t)ts.tv_sec * 1000000000LL + (uint64_t)ts.tv_nsec;
        }

    protected:
        timespec t1, t2;
    };
} // namespace util