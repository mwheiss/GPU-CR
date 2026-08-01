#ifndef COMMON_H
#define COMMON_H

#include <unistd.h>   // for many thing
#include <stdlib.h>   // for standard library
#include <stdio.h>    // for file dump
#include <time.h>     // for timing
#include <dlfcn.h>    // for loading real shared library
#include <stdint.h>   // for uint64_t defn
#include <stdbool.h>  // for true false
#include <elf.h>      // for ELF Header
#include <sys/wait.h> // for waiting subprocess
#include <sys/stat.h> // for directory
#include <sys/mman.h>  // for mmap
#include <pthread.h>  // for mutex lock
#include <signal.h>   // for signal handling
#include <map>
#include <utility>
#include <atomic>

#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define ROUND_UP_2MB(x) (((x) + (2 * 1024 * 1024 - 1)) & ~(2 * 1024 * 1024 - 1))

// SHM_SIZE: Per-GPU checkpoint buffer on hugepages.
// Each GPU process allocates SHM_SIZE + STAGING_BUF_SIZE*STAGING_BUF_NUM.
// For TP=N, total hugepage needed = N * (SHM_SIZE + 2GB) + overhead.
// Override at compile time: cmake .. -DSHM_SIZE_GB=40
#ifndef SHM_SIZE_GB
#define SHM_SIZE_GB 25
#endif
#define SHM_SIZE ((unsigned long)SHM_SIZE_GB << 30)

#define MAX_FILE_NUM 4096
#define COPY_THRESHOLD (1UL << 29) // 0.5GB, when to copy from host_buf to shm
#define NUM_COPY_THREADS 4
#define CR_INIT_SIGNAL     SIGRTMAX
// Long-lived Python/vLLM services may replace SIGUSR1/SIGUSR2 after the
// preload constructor runs. Keep every internal GPU-CR control message on a
// dedicated realtime signal so the handler cannot silently disappear.
#define CR_CKPT_SIGNAL     (SIGRTMAX - 4)
#define CR_RESTORE_SIGNAL  (SIGRTMAX - 5)

// Multi-GPU: IPC teardown/rebuild signals (real-time signals)
// These replace the old NCCL suspend/resume signals.
#define CR_IPC_TEARDOWN_SIGNAL  (SIGRTMAX - 1)
#define CR_IPC_REBUILD_SIGNAL   (SIGRTMAX - 2)
#define CR_IPC_VALIDATE_SIGNAL  (SIGRTMAX - 3)

// Legacy aliases (for backward compatibility during transition)
#define CR_NCCL_SUSPEND_SIGNAL  CR_IPC_TEARDOWN_SIGNAL
#define CR_NCCL_RESUME_SIGNAL   CR_IPC_REBUILD_SIGNAL

// Maximum number of processes in multi-GPU checkpoint
#define MAX_MULTI_GPU_PROCS 32

#ifndef STAGING_BUF_MB
#define STAGING_BUF_MB 256
#endif
#define STAGING_BUF_SIZE ((unsigned long)STAGING_BUF_MB << 20)
#define STAGING_BUF_NUM 2

typedef void (*sighandler_t)(int);
typedef sighandler_t (*signal_func_t)(int, sighandler_t);

// Global memory tracking map: ptr -> size
extern std::map<void*, size_t> allocated_memory;

// Global memory type tracking: ptr -> type (0=runtime Malloc, 1=VMM)
extern std::map<void*, int> allocated_memory_type;

// Helper function declarations
void memcpy_multi(void* dest, void* src, size_t size);

// Heavy checkpoint work must never run in a POSIX signal handler.  The
// preload library routes control signals through a pipe to a dedicated
// thread.  Allocation hooks call this after fork so vLLM worker children get
// their own controller (threads do not survive fork()).
void gpu_cr_ensure_control_thread(void);

struct shared_mem_file {
    void* ptr;
    uint64_t start_offset;
    uint64_t size;
};

struct shared_mem_fs {
    uint64_t file_num;
    uint64_t current_offset;
    struct shared_mem_file files[MAX_FILE_NUM];
};

struct signal_controls {
    uint32_t signal;
};

#endif
