#include <atomic>
#include <chrono>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "common.h"
#include "comm/comm.h"
#include "backend/backend.h"
#include "GPUs/GPU.h"
#include "nccl_hooks.h"

// IPC hooks for cuMem IPC management (defined in ipc_hooks.cpp).
// On NVIDIA this header also declares ipc_disable_all_peer_access /
// ipc_reenable_all_peer_access, the canonical P2P teardown helpers.
#include "ipc_hooks.h"
// UDS fd exchange for cross-process CUDA handle transfer (defined in ipc_fd_exchange.cpp)
#include "ipc_fd_exchange.h"

// Buffer for saving IPC export GPU data between teardown and rebuild phases
static void*  g_ipc_export_data_buf  = nullptr;
static size_t g_ipc_export_data_size = 0;
static void*  g_local_alloc_data_buf  = nullptr;
static size_t g_local_alloc_data_size = 0;

std::mutex fs_mutex;
Comm *comm;
Backend *backend;
GPU *gpu;

void* staging_buf[STAGING_BUF_NUM];

// Optional two-tier checkpointing.  The synchronous CUDA path writes the
// allocation image into anonymous RAM, releases VRAM, and returns.  A worker
// then moves page-aligned chunks into the persistent backend.  Restore holds
// g_persistence_io_mutex so it observes a stable split: the persisted prefix
// comes from the file mapping and the remaining suffix comes from RAM.
static void* g_ram_checkpoint = nullptr;
static bool g_async_persist = false;
static std::string g_cache_policy = "keep";
static int g_cr_id = -1;
static std::mutex g_persistence_io_mutex;
static std::atomic<bool> g_persistence_cancel{false};
static std::atomic<bool> g_restore_waiting{false};
static std::atomic<bool> g_persistence_complete{false};
static std::atomic<uint64_t> g_persisted_offset{0};
static std::atomic<uint64_t> g_checkpoint_end{0};
static std::thread* g_persistence_thread = nullptr;
static std::mutex g_file_mapping_mutex;
static bool g_file_data_mapped = true;

bool CR_initialized = false;

static void handle_cr_signal(int signum);
static void handle_ipc_signal(int signum);

static std::mutex control_thread_mutex;
static pid_t control_thread_pid = -1;
static int control_pipe[2] = {-1, -1};

static void* control_thread_main(void*) {
    for (;;) {
        int signum = 0;
        ssize_t got = read(control_pipe[0], &signum, sizeof(signum));
        if (got == (ssize_t)sizeof(signum)) {
            if (signum == CR_INIT_SIGNAL || signum == CR_CKPT_SIGNAL ||
                signum == CR_RESTORE_SIGNAL) {
                handle_cr_signal(signum);
            } else if (signum == CR_IPC_TEARDOWN_SIGNAL ||
                       signum == CR_IPC_REBUILD_SIGNAL) {
                handle_ipc_signal(signum);
            } else if (signum == CR_IPC_VALIDATE_SIGNAL) {
                ipc_validate_all_mappings("ON-DEMAND");
                fflush(stderr);
            }
            continue;
        }
        if (got < 0 && errno == EINTR) continue;
        break;
    }
    return nullptr;
}

void gpu_cr_ensure_control_thread(void) {
    const pid_t self = getpid();
    std::lock_guard<std::mutex> guard(control_thread_mutex);
    if (control_thread_pid == self && control_pipe[1] >= 0) return;

    // A vLLM worker may be forked after the preload constructor ran.  Threads
    // do not survive fork, so discard the inherited pipe and create a worker-
    // local controller when an allocation hook first runs in the child.
    if (control_pipe[0] >= 0) close(control_pipe[0]);
    if (control_pipe[1] >= 0) close(control_pipe[1]);
    control_pipe[0] = control_pipe[1] = -1;

    if (pipe2(control_pipe, O_CLOEXEC) != 0) {
        perror("[vGPU] pipe2 control channel");
        return;
    }
    int flags = fcntl(control_pipe[1], F_GETFL, 0);
    if (flags >= 0) fcntl(control_pipe[1], F_SETFL, flags | O_NONBLOCK);

    pthread_t thread;
    int rc = pthread_create(&thread, nullptr, control_thread_main, nullptr);
    if (rc != 0) {
        errno = rc;
        perror("[vGPU] pthread_create control thread");
        close(control_pipe[0]);
        close(control_pipe[1]);
        control_pipe[0] = control_pipe[1] = -1;
        return;
    }
    pthread_detach(thread);
    control_thread_pid = self;
    fprintf(stderr, "[vGPU] Control thread ready for PID %d\n", self);
}

static void queue_control_signal(int signum) {
    int saved_errno = errno;
    int fd = control_pipe[1];
    if (fd >= 0) {
        // write(2) is async-signal-safe.  Control operations are serialized by
        // the manager, so a full pipe indicates a genuine protocol failure.
        (void)write(fd, &signum, sizeof(signum));
    }
    errno = saved_errno;
}

static void control_atfork_prepare(void) {
    control_thread_mutex.lock();
}

static void control_atfork_parent(void) {
    control_thread_mutex.unlock();
}

static void control_atfork_child(void) {
    if (control_pipe[0] >= 0) close(control_pipe[0]);
    if (control_pipe[1] >= 0) close(control_pipe[1]);
    control_pipe[0] = control_pipe[1] = -1;
    control_thread_pid = -1;
    control_thread_mutex.unlock();
}

// Helper function: multi-threaded memcpy
void memcpy_multi(void* dest, void* src, size_t size) {
    std::vector<std::thread> threads;
    size_t chunk_size = (size + NUM_COPY_THREADS - 1) / NUM_COPY_THREADS;
    for (int i = 0; i < NUM_COPY_THREADS; i++) {
        size_t offset = i * chunk_size;
        if (offset >= size) break;
        size_t this_chunk_size = std::min(chunk_size, size - offset);
        threads.emplace_back([=]() {
            memcpy((char*)dest + offset, (char*)src + offset, this_chunk_size);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}

static bool env_enabled(const char* name, bool default_value = false) {
    const char* value = std::getenv(name);
    if (!value || !*value) return default_value;
    return !strcmp(value, "1") || !strcasecmp(value, "true") ||
           !strcasecmp(value, "yes") || !strcasecmp(value, "on");
}

static void write_persistence_status(const char* state, uint64_t bytes,
                                     const char* detail = nullptr) {
    const char* root = std::getenv("EXPORT_FILE_PATH");
    if (!root || g_cr_id < 0) return;
    char path[768];
    char temporary[800];
    snprintf(path, sizeof(path), "%s/ckpt-%d.persist.json", root, g_cr_id);
    snprintf(temporary, sizeof(temporary), "%s.tmp.%d", path, getpid());
    int fd = open(temporary, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        fprintf(stderr, "[vGPU-persist] open status failed: %s\n", strerror(errno));
        return;
    }
    dprintf(fd,
            "{\"state\":\"%s\",\"bytes\":%llu,\"end\":%llu,"
            "\"pid\":%d,\"cache_policy\":\"%s\",\"detail\":\"%s\"}\n",
            state, (unsigned long long)bytes,
            (unsigned long long)g_checkpoint_end.load(std::memory_order_acquire),
            getpid(), g_cache_policy.c_str(), detail ? detail : "");
    close(fd);
    if (rename(temporary, path) != 0) {
        fprintf(stderr, "[vGPU-persist] rename status failed: %s\n", strerror(errno));
        unlink(temporary);
    }
}

static bool checkpoint_file_path(char* path, size_t size) {
    const char* root = std::getenv("EXPORT_FILE_PATH");
    if (!root || g_cr_id < 0) return false;
    snprintf(path, size, "%s/ckpt-%d.data", root, g_cr_id);
    return true;
}

static bool checkpoint_bulk_file_path(char* path, size_t size) {
    const char* root = std::getenv("EXPORT_FILE_PATH");
    if (!root || g_cr_id < 0) return false;
    snprintf(path, size, "%s/ckpt-%d.bulk.data", root, g_cr_id);
    return true;
}

static bool evict_persistent_file_cache(uint64_t end) {
    const uint64_t start = ROUND_UP_2MB(sizeof(shared_mem_fs));
    if (end <= start) return true;
    const size_t length = SHM_SIZE - start;
    char path[768];
    if (!checkpoint_file_path(path, sizeof(path))) return false;

    // Remove every PTE referencing the bulk file range, then reserve the same
    // virtual addresses with an inaccessible anonymous mapping.  This lets
    // POSIX_FADV_DONTNEED actually reclaim clean page-cache pages while keeping
    // the address stable for a later MAP_FIXED remap.  The first 2 MiB remains
    // file-backed because it contains GPU-CR's control metadata.
    {
        std::lock_guard<std::mutex> mapping_guard(g_file_mapping_mutex);
        char* address = static_cast<char*>(backend->get_tmp_buf()) + start;
        if (g_file_data_mapped && munmap(address, length) != 0) {
            fprintf(stderr, "[vGPU-persist] persistent data unmap failed: %s\n",
                    strerror(errno));
            return false;
        }
        void* reservation = mmap(address, length, PROT_NONE,
                                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                                     MAP_FIXED,
                                 -1, 0);
        if (reservation == MAP_FAILED || reservation != address) {
            fprintf(stderr, "[vGPU-persist] address reservation failed: %s\n",
                    strerror(errno));
            return false;
        }
        g_file_data_mapped = false;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[vGPU-persist] open for fadvise failed: %s\n",
                strerror(errno));
        return false;
    }
    int rc = posix_fadvise(fd, start, end - start, POSIX_FADV_DONTNEED);
    close(fd);
    if (rc != 0) {
        fprintf(stderr, "[vGPU-persist] POSIX_FADV_DONTNEED failed: %s\n",
                strerror(rc));
        return false;
    }
    fprintf(stderr,
            "[vGPU-persist] Unmapped and invalidated persistent file cache for %llu bytes\n",
            (unsigned long long)(end - start));
    return true;
}

static bool direct_write_all(int fd, const void* buffer, size_t amount,
                             uint64_t offset) {
    const char* source = static_cast<const char*>(buffer);
    size_t done = 0;
    while (done < amount) {
        ssize_t written = pwrite(fd, source + done, amount - done, offset + done);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            fprintf(stderr, "[vGPU-persist] direct pwrite failed at %llu: %s\n",
                    (unsigned long long)(offset + done), strerror(errno));
            return false;
        }
        done += written;
    }
    return true;
}

static bool direct_read_all(int fd, void* buffer, size_t amount,
                            uint64_t offset) {
    char* destination = static_cast<char*>(buffer);
    size_t done = 0;
    while (done < amount) {
        ssize_t got = pread(fd, destination + done, amount - done, offset + done);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            fprintf(stderr, "[vGPU-restore] direct pread failed at %llu: %s\n",
                    (unsigned long long)(offset + done), strerror(errno));
            return false;
        }
        done += got;
    }
    return true;
}

static void persistence_worker() {
    char* ram_image = static_cast<char*>(g_ram_checkpoint);
    const uint64_t end = g_checkpoint_end.load(std::memory_order_acquire);
    uint64_t offset = ROUND_UP_2MB(sizeof(shared_mem_fs));
    auto started = std::chrono::steady_clock::now();
    write_persistence_status("pending", offset);

    char path[768];
    if (!checkpoint_bulk_file_path(path, sizeof(path))) {
        write_persistence_status("failed", offset, "file_path_missing");
        return;
    }
    int direct_fd = open(path, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (direct_fd < 0) {
        fprintf(stderr, "[vGPU-persist] O_DIRECT open failed: %s\n", strerror(errno));
        write_persistence_status("failed", offset, "direct_open_failed");
        return;
    }
    if (ftruncate(direct_fd, end) != 0) {
        fprintf(stderr, "[vGPU-persist] bulk ftruncate failed: %s\n",
                strerror(errno));
        close(direct_fd);
        write_persistence_status("failed", offset, "bulk_truncate_failed");
        return;
    }
    int allocation_rc = posix_fallocate(direct_fd, 0, end);
    if (allocation_rc != 0) {
        fprintf(stderr, "[vGPU-persist] bulk preallocation failed: %s\n",
                strerror(allocation_rc));
        close(direct_fd);
        write_persistence_status("failed", offset, "bulk_preallocate_failed");
        return;
    }
    {
        std::lock_guard<std::mutex> guard(g_persistence_io_mutex);
        if (!evict_persistent_file_cache(end)) {
            close(direct_fd);
            write_persistence_status("failed", offset, "file_unmap_failed");
            return;
        }
    }

    while (offset < end) {
        if (g_persistence_cancel.load(std::memory_order_acquire)) {
            close(direct_fd);
            write_persistence_status("cancelled", offset);
            fprintf(stderr, "[vGPU-persist] Cancelled at %llu/%llu bytes\n",
                    (unsigned long long)offset, (unsigned long long)end);
            return;
        }
        while (g_restore_waiting.load(std::memory_order_acquire) &&
               !g_persistence_cancel.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        size_t amount = std::min((uint64_t)STAGING_BUF_SIZE, end - offset);
        {
            // Restore holds this mutex for its complete host-to-device copy.
            // A completed chunk is therefore switched from RAM to file
            // atomically from restore's point of view.
            std::lock_guard<std::mutex> guard(g_persistence_io_mutex);
            if (g_persistence_cancel.load(std::memory_order_acquire)) continue;
            if (!direct_write_all(direct_fd, ram_image + offset, amount, offset)) {
                close(direct_fd);
                write_persistence_status("failed", offset, "direct_write_failed");
                return;
            }
            uint64_t new_offset = offset + amount;
            g_persisted_offset.store(new_offset, std::memory_order_release);
            if (g_cache_policy != "keep" &&
                madvise(ram_image + offset, amount, MADV_DONTNEED) != 0) {
                    fprintf(stderr,
                            "[vGPU-persist] MADV_DONTNEED RAM chunk failed at %llu: %s\n",
                            (unsigned long long)offset, strerror(errno));
            }
            offset = new_offset;
        }
    }
    close(direct_fd);

    g_persistence_complete.store(true, std::memory_order_release);
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    fprintf(stderr,
            "[vGPU-persist] Complete: %llu bytes in %.3f seconds (%.2f GiB/s)\n",
            (unsigned long long)end, elapsed,
            (end / (1024.0 * 1024.0 * 1024.0)) / elapsed);

    // O_DIRECT bypasses page cache for the bulk image.  Re-issue the hint for
    // metadata/tail pages without blocking the already-completed suspend.
    bool cache_evicted = evict_persistent_file_cache(end);
    fprintf(stderr, "[vGPU-persist] Cache policy applied: %s (file_evicted=%s)\n",
            g_cache_policy.c_str(), cache_evicted ? "true" : "false");
    write_persistence_status("complete", end,
                             cache_evicted ? "file_cache_evicted" : "");
}

static void stop_persistence_worker() {
    if (!g_persistence_thread) return;
    g_persistence_cancel.store(true, std::memory_order_release);
    if (g_persistence_thread->joinable()) g_persistence_thread->join();
    delete g_persistence_thread;
    g_persistence_thread = nullptr;
    g_persistence_cancel.store(false, std::memory_order_release);
    g_restore_waiting.store(false, std::memory_order_release);
}

static void start_persistence_worker(uint64_t end) {
    g_checkpoint_end.store(end, std::memory_order_release);
    g_persisted_offset.store(ROUND_UP_2MB(sizeof(shared_mem_fs)),
                             std::memory_order_release);
    g_persistence_complete.store(false, std::memory_order_release);
    g_persistence_cancel.store(false, std::memory_order_release);
    write_persistence_status("pending", g_persisted_offset.load());
    g_persistence_thread = new std::thread(persistence_worker);
}

static bool ram_range_resident(uint64_t offset, size_t amount) {
    const long page_size = sysconf(_SC_PAGESIZE);
    size_t pages = (amount + page_size - 1) / page_size;
    std::vector<unsigned char> residency(pages);
    if (mincore(static_cast<char*>(g_ram_checkpoint) + offset, amount,
                residency.data()) != 0) {
        fprintf(stderr, "[vGPU-restore] mincore failed: %s\n", strerror(errno));
        return false;
    }
    for (unsigned char value : residency) {
        if (!(value & 1)) return false;
    }
    return true;
}

static void copy_checkpoint_chunk(void* destination, uint64_t offset,
                                  size_t amount, int direct_fd,
                                  uint64_t persisted_offset) {
    if (!g_async_persist) {
        memcpy_multi(destination,
                     static_cast<char*>(backend->get_tmp_buf()) + offset,
                     amount);
        return;
    }
    bool use_ram = false;
    bool must_use_ram = offset >= persisted_offset;
    bool prefer_resident_ram = g_cache_policy == "keep" &&
        ram_range_resident(offset, amount);
    use_ram = must_use_ram || prefer_resident_ram;
    if (use_ram) {
        memcpy_multi(destination, static_cast<char*>(g_ram_checkpoint) + offset,
                     amount);
    } else if (!direct_read_all(direct_fd, destination, amount, offset)) {
        exit(EXIT_FAILURE);
    }
}


double ckpt() {
    fprintf(stderr, "[vGPU-CKPT] ckpt() entered, PID=%d\n", getpid());
    fflush(stderr);
    
    double tot_size = 0;
    
    auto time_start = std::chrono::high_resolution_clock::now();
    long submit_time = 0, sync_time = 0, cpu_copy_time = 0, release_time = 0;

    if (g_async_persist) stop_persistence_worker();
    void* tmp_buf = backend->get_tmp_buf();
    void* checkpoint_data = g_async_persist ? g_ram_checkpoint : tmp_buf;
    fprintf(stderr, "[vGPU-CKPT] tmp_buf=%p\n", tmp_buf);
    fflush(stderr);
    shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;
    int current_buf = 0;
    size_t buf_offset = 0;
    size_t des_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));
    
    fs_mutex.lock();
    
    fs->file_num = 0;
    fs->current_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));

    GPUStream stream;
    GPUEvent event;
    if (gpu->createStream(&stream) != 0) {
        fprintf(stderr, "Error: Failed to create stream\\n");
        fs_mutex.unlock();
        exit(-1);
    }
    if (gpu->createEvent(&event) != 0) {
        fprintf(stderr, "Error: Failed to create event\\n");
        fs_mutex.unlock();
        exit(-1);
    }
    gpu->recordEvent(event, stream);
    
    fprintf(stderr, "[vGPU-CKPT] ckpt %ld ptrs\n", allocated_memory.size());
    fflush(stderr);

    int ptr_idx = 0;
    for (const auto& entry : allocated_memory) {
        fprintf(stderr, "[vGPU-CKPT] Processing ptr #%d: %p\n", ++ptr_idx, entry.first);
        fflush(stderr);
        void* d = entry.first;
        uint64_t size = ROUND_UP_2MB(entry.second);
        tot_size += size;

        // Record file info
        fs->files[fs->file_num].ptr = d;
        fs->files[fs->file_num].start_offset = fs->current_offset;
        fs->files[fs->file_num].size = size;
        fs->current_offset += size;
        if (fs->current_offset > SHM_SIZE) {
            fprintf(stderr, "[vGPU-CKPT] Error: Not enough space in shared memory\n");
            fs_mutex.unlock();
            exit(-1);
        }
        fs->file_num++;
        if (fs->file_num >= MAX_FILE_NUM) {
            fprintf(stderr, "[vGPU-CKPT] Error: Too many files in shared memory fs\n");
            fs_mutex.unlock();
            exit(-1);
        }
        
        // Copy data from GPU to staging buffer
        while(size > 0) {
            size_t cur_size = std::min(size, (size_t)STAGING_BUF_SIZE - buf_offset);
            void* start_addr = (char*)staging_buf[current_buf & 1] + buf_offset;
            
            auto tm1 = std::chrono::high_resolution_clock::now();
            int memcpy_rc = gpu->memcpyAsync(start_addr, d, cur_size,
                                             GPUMemcpyKind::DeviceToHost, stream);
            auto tm2 = std::chrono::high_resolution_clock::now();
            submit_time += std::chrono::duration_cast<std::chrono::microseconds>(tm2 - tm1).count();
            if (memcpy_rc != 0) {
                fprintf(stderr, "Error: memcpyAsync failed\\n");
                fs_mutex.unlock();
                exit(-1);
            }
            
            buf_offset += cur_size;
            d = (char*)d + cur_size;
            size -= cur_size;
            if(buf_offset >= STAGING_BUF_SIZE) {
                assert(buf_offset == STAGING_BUF_SIZE);
                if(current_buf > 0) {
                    auto t3 = std::chrono::high_resolution_clock::now();
                    gpu->synchronizeEvent(event);
                    auto t4 = std::chrono::high_resolution_clock::now();
                    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
                    
                    auto t5 = std::chrono::high_resolution_clock::now();
                    memcpy_multi((char*)checkpoint_data + des_offset, staging_buf[(current_buf - 1) & 1], STAGING_BUF_SIZE);
                    auto t6 = std::chrono::high_resolution_clock::now();
                    cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
                    
                    des_offset += STAGING_BUF_SIZE;
                }
                buf_offset = 0;
                current_buf++;
                gpu->recordEvent(event, stream);
            }
        }
    }
    if(current_buf > 0) {
        auto t3 = std::chrono::high_resolution_clock::now();
        gpu->synchronizeEvent(event);
        auto t4 = std::chrono::high_resolution_clock::now();
        sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        
        auto t5 = std::chrono::high_resolution_clock::now();
        memcpy_multi((char*)checkpoint_data + des_offset, staging_buf[(current_buf - 1) & 1], STAGING_BUF_SIZE);
        auto t6 = std::chrono::high_resolution_clock::now();
        cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
        
        des_offset += STAGING_BUF_SIZE;
    }
    
    auto t7 = std::chrono::high_resolution_clock::now();
    gpu->synchronizeStream(stream);
    auto t8 = std::chrono::high_resolution_clock::now();
    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();
    
    auto t9 = std::chrono::high_resolution_clock::now();
    memcpy_multi((char*)checkpoint_data + des_offset, staging_buf[current_buf & 1], buf_offset);
    auto t10 = std::chrono::high_resolution_clock::now();
    cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t10 - t9).count();
    
    assert(des_offset + buf_offset == fs->current_offset);
    gpu->destroyStream(stream);
    gpu->destroyEvent(event);

    // Release physical GPU memory after checkpoint (but keep virtual addresses)
    fprintf(stderr, "Releasing physical GPU memory for %ld pointers...\n", allocated_memory.size());
    auto t11 = std::chrono::high_resolution_clock::now();
    for (const auto& entry : allocated_memory) {
        void* ptr = entry.first;
        if (gpu->releasePhysicalMemory(ptr) != 0) {
            fprintf(stderr, "Error: Failed to release physical memory for ptr %p\n", ptr);
            fs_mutex.unlock();
            exit(-1);
        }
    }
    auto t12 = std::chrono::high_resolution_clock::now();
    release_time = std::chrono::duration_cast<std::chrono::microseconds>(t12 - t11).count();
    fprintf(stderr, "Physical GPU memory released, virtual addresses preserved\n");

    if (g_async_persist) {
        start_persistence_worker(fs->current_offset);
        fprintf(stderr,
                "[vGPU-CKPT] VRAM released; persistence continues in background "
                "from anonymous RAM (%llu bytes)\n",
                (unsigned long long)fs->current_offset);
    }
    

    fprintf(stderr, "=== Checkpoint Timing Breakdown ===\n");
    fprintf(stderr, "  GPU submit:       %6ld ms\n", submit_time / 1000);
    fprintf(stderr, "  GPU sync:         %6ld ms\n", sync_time / 1000);
    fprintf(stderr, "  CPU memcpy:       %6ld ms (%.2f GB/s)\n", 
            cpu_copy_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (cpu_copy_time / 1000000.0));
    fprintf(stderr, "  Release memory:   %6ld ms\n", release_time / 1000);
    long data_transfer_time = sync_time + cpu_copy_time;
    fprintf(stderr, "  Data transfer:    %6ld ms (%.2f GB/s)\n",
            data_transfer_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (data_transfer_time / 1000000.0));
    fprintf(stderr, "===================================\n");
    
    fs_mutex.unlock();
    return tot_size;
}

double restore_ptr_and_content() {
    double tot_size = 0;
    
    long remap_time = 0, submit_time = 0, cpu_copy_time = 0, sync_time = 0;
    
    void* tmp_buf = backend->get_tmp_buf();
    shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;

    // Freeze the RAM/file ownership boundary for the duration of restore.
    // At most one 256 MiB persistence chunk must finish before this lock is
    // acquired, so restore never waits for the complete disk image.
    std::unique_lock<std::mutex> persistence_guard(
        g_persistence_io_mutex, std::defer_lock);
    if (g_async_persist) {
        g_restore_waiting.store(true, std::memory_order_release);
        persistence_guard.lock();
        g_restore_waiting.store(false, std::memory_order_release);
    }
    uint64_t persisted_offset = g_async_persist
        ? g_persisted_offset.load(std::memory_order_acquire)
        : fs->current_offset;
    fprintf(stderr, "[vGPU-restore] source boundary: file=%llu RAM=%llu bytes\n",
            (unsigned long long)persisted_offset,
            (unsigned long long)(fs->current_offset - persisted_offset));

    uint64_t file_num = fs->file_num;
    fprintf(stderr, "[vGPU-restore] restore %lu ptrs\n", file_num);
    int direct_fd = -1;
    if (g_async_persist) {
        char path[768];
        if (!checkpoint_bulk_file_path(path, sizeof(path)) ||
            (direct_fd = open(path, O_RDONLY | O_DIRECT)) < 0) {
            fprintf(stderr, "[vGPU-restore] O_DIRECT open failed: %s\n",
                    strerror(errno));
            return -1;
        }
    }
    
    // Remap physical memory for all pointers before copying data
    fprintf(stderr, "[vGPU-restore] Remapping physical GPU memory for %lu pointers...\n", file_num);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < file_num; i++) {
        void* ptr = fs->files[i].ptr;
        uint64_t size = fs->files[i].size;
        if (gpu->remapPhysicalMemory(ptr, size) != 0) {
            fprintf(stderr, "Error: Failed to remap physical memory for ptr %p\n", ptr);
            exit(-1);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    remap_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    fprintf(stderr, "[vGPU-restore] Physical GPU memory remapped\n");
    
    GPUStream stream;
    GPUEvent event;
    if (gpu->createStream(&stream) != 0) {
        fprintf(stderr, "Error: Failed to create stream\\n");
        exit(-1);
    }
    if (gpu->createEvent(&event) != 0) {
        fprintf(stderr, "Error: Failed to create event\\n");
        exit(-1);
    }
    gpu->recordEvent(event, stream);

    int current_buf = 0;
    size_t buf_offset = 0;
    size_t src_offset = 0;
    
    for (uint64_t i = 0; i < file_num; i++) {
        void* requestedAddr = fs->files[i].ptr;
        uint64_t offset = fs->files[i].start_offset;
        uint64_t size = fs->files[i].size;
        tot_size += size;
        
        if(i == 0) {
            src_offset = fs->files[i].start_offset;
            size_t cpu_copy_size = std::min((size_t)(fs->current_offset - src_offset), (size_t)STAGING_BUF_SIZE);
            auto tc1 = std::chrono::high_resolution_clock::now();
            copy_checkpoint_chunk(staging_buf[current_buf & 1], src_offset,
                                  cpu_copy_size, direct_fd, persisted_offset);
            auto tc2 = std::chrono::high_resolution_clock::now();
            cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(tc2 - tc1).count();
            buf_offset = 0;
        }
        
        while(size > 0) {
            size_t this_copy_size = std::min(size, (size_t)STAGING_BUF_SIZE - buf_offset);
            assert(buf_offset == offset - src_offset);
            
            auto tm1 = std::chrono::high_resolution_clock::now();
            int memcpy_rc = gpu->memcpyAsync(
                requestedAddr,
                (char*)staging_buf[current_buf & 1] + (offset - src_offset),
                this_copy_size, GPUMemcpyKind::HostToDevice, stream);
            auto tm2 = std::chrono::high_resolution_clock::now();
            submit_time += std::chrono::duration_cast<std::chrono::microseconds>(tm2 - tm1).count();
            if (memcpy_rc != 0) {
                fprintf(stderr, "Error: memcpyAsync failed\\n");
                exit(-1);
            }
            
            buf_offset += this_copy_size;
            offset += this_copy_size;
            requestedAddr = (char*)requestedAddr + this_copy_size;
            size -= this_copy_size;
            
            if(buf_offset >= STAGING_BUF_SIZE) {
                assert(buf_offset == STAGING_BUF_SIZE);
                src_offset += STAGING_BUF_SIZE;
                size_t cpu_copy_size = std::min((size_t)(fs->current_offset - src_offset), (size_t)STAGING_BUF_SIZE);
                
                auto ts1 = std::chrono::high_resolution_clock::now();
                gpu->synchronizeEvent(event);
                auto ts2 = std::chrono::high_resolution_clock::now();
                sync_time += std::chrono::duration_cast<std::chrono::microseconds>(ts2 - ts1).count();
                
                auto tc3 = std::chrono::high_resolution_clock::now();
                copy_checkpoint_chunk(staging_buf[(current_buf + 1) & 1],
                                      src_offset, cpu_copy_size, direct_fd,
                                      persisted_offset);
                auto tc4 = std::chrono::high_resolution_clock::now();
                cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(tc4 - tc3).count();
                
                buf_offset = 0;
                current_buf++;
                gpu->recordEvent(event, stream);
            }
        }
    }
    
    auto ts3 = std::chrono::high_resolution_clock::now();
    gpu->synchronizeStream(stream);
    auto ts4 = std::chrono::high_resolution_clock::now();
    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(ts4 - ts3).count();
    
    gpu->destroyStream(stream);
    gpu->destroyEvent(event);
    if (direct_fd >= 0) close(direct_fd);
    
    fprintf(stderr, "=== Restore Timing Breakdown ===\n");
    fprintf(stderr, "  Remap memory:     %6ld ms\n", remap_time / 1000);
    fprintf(stderr, "  GPU submit:       %6ld ms\n", submit_time / 1000);
    fprintf(stderr, "  CPU memcpy:       %6ld ms (%.2f GB/s)\n",
            cpu_copy_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (cpu_copy_time / 1000000.0));
    fprintf(stderr, "  GPU sync:         %6ld ms\n", sync_time / 1000);
    long data_transfer_time = cpu_copy_time + sync_time;
    fprintf(stderr, "  Data transfer:    %6ld ms (%.2f GB/s)\n",
            data_transfer_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (data_transfer_time / 1000000.0));
    fprintf(stderr, "================================\n");
    
    return tot_size;
}

int get_id() {
    char id_name[512];
    const char* ctl_dir = std::getenv("EXPORT_FILE_PATH");
    if (!ctl_dir) ctl_dir = "/mnt/huge-ckpt";
    snprintf(id_name, sizeof(id_name), "%s/control", ctl_dir);
    int fd_id = open(id_name, O_CREAT | O_RDWR, 0755);
    if (fd_id < 0) {
        perror("open()");
        exit(EXIT_FAILURE);
    }
    // Set file size before mmap to avoid Bus error
    if (ftruncate(fd_id, HUGE_PAGE_SIZE) < 0) {
        perror("ftruncate()");
        exit(EXIT_FAILURE);
    }
    std::atomic<int>* id_ptr = (std::atomic<int>*)mmap(NULL, HUGE_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_id, 0);
    if (id_ptr == MAP_FAILED) {
        perror("mmap()");
        exit(EXIT_FAILURE);
    }
    int id = id_ptr->fetch_add(1);
    fprintf(stderr, "Process ID: %d, assigned CR ID: %d\n", getpid(), id);
    return id;
}


void init_CR() {
    if (CR_initialized) {
        fprintf(stderr, "[init_CR] CR already initialized\n");
        return;
    }

    fprintf(stderr, "[init_CR] Starting CR initialization...\n");
    int id = get_id();
    g_cr_id = id;
    comm = new ShareMemComm(getpid());
    comm->setup();
    backend = new ShareMem(id);
    backend->setup();
    g_async_persist = env_enabled("GPU_CR_ASYNC_PERSIST", false);
    const char* cache_policy = std::getenv("GPU_CR_CHECKPOINT_CACHE_POLICY");
    if (cache_policy && *cache_policy) g_cache_policy = cache_policy;
    if (g_cache_policy != "keep" && g_cache_policy != "cold" &&
        g_cache_policy != "pageout") {
        fprintf(stderr, "[init_CR] Error: invalid GPU_CR_CHECKPOINT_CACHE_POLICY=%s\n",
                g_cache_policy.c_str());
        exit(EXIT_FAILURE);
    }
    if (g_async_persist) {
        g_ram_checkpoint = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                                -1, 0);
        if (g_ram_checkpoint == MAP_FAILED) {
            perror("[init_CR] mmap RAM checkpoint tier");
            exit(EXIT_FAILURE);
        }
        (void)madvise(g_ram_checkpoint, SHM_SIZE, MADV_DONTDUMP);
        if (env_enabled("GPU_CR_RAM_PREFAULT", true)) {
            uint64_t needed = ROUND_UP_2MB(sizeof(shared_mem_fs));
            for (const auto& entry : allocated_memory) {
                needed += ROUND_UP_2MB(entry.second);
            }
            auto prefault_start = std::chrono::steady_clock::now();
            if (madvise(static_cast<char*>(g_ram_checkpoint), needed,
                        MADV_POPULATE_WRITE) != 0) {
                fprintf(stderr, "[init_CR] RAM tier prefault failed: %s\n",
                        strerror(errno));
            } else {
                auto prefault_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - prefault_start).count();
                fprintf(stderr,
                        "[init_CR] Prefaulted %llu RAM-tier bytes in %.3f seconds\n",
                        (unsigned long long)needed, prefault_seconds);
            }
        }
        fprintf(stderr,
                "[init_CR] Two-tier checkpointing enabled: RAM=%p size=%lu "
                "cache_policy=%s\n",
                g_ram_checkpoint, SHM_SIZE, g_cache_policy.c_str());
    }
    gpu = createGPU();  // createGPU() will detect the GPU vendor and return the appropriate GPU object
    fprintf(stderr, "[init_CR] GPU vendor: %s\n", gpu->getVendorName().c_str());
    fprintf(stderr, "[init_CR] Allocating staging buffer (%zu MB)...\n", 
            (STAGING_BUF_SIZE * STAGING_BUF_NUM) / (1024 * 1024));
    
    void* tmp_buf_host = backend->get_host_buffer();
    if (!tmp_buf_host) {
        fprintf(stderr, "[init_CR] Error: Backend host buffer is null\n");
        exit(EXIT_FAILURE);
    }
    
    // Try to register as pinned memory
    size_t total_size = STAGING_BUF_SIZE * STAGING_BUF_NUM;
    if (gpu->registerHostMemory(tmp_buf_host, total_size) == 0) {
        fprintf(stderr, "[init_CR] Successfully registered as pinned memory\n");
    } else {
        const char* allow_pageable = std::getenv("GPU_CR_ALLOW_PAGEABLE_STAGING");
        bool explicitly_allowed = allow_pageable &&
            (!strcmp(allow_pageable, "1") || !strcasecmp(allow_pageable, "true"));
        if (!explicitly_allowed) {
            fprintf(stderr,
                    "[init_CR] Error: CUDA staging memory is not pinned; refusing the "
                    "known-slow pageable path. Set GPU_CR_ALLOW_PAGEABLE_STAGING=1 "
                    "only for an explicit compatibility fallback.\n");
            exit(EXIT_FAILURE);
        }
        fprintf(stderr,
                "[init_CR] Warning: explicitly allowing unpinned staging; "
                "checkpoint performance may be severely degraded\n");
    }
    
    for (int i = 0; i < STAGING_BUF_NUM; i++) {
        staging_buf[i] = (char*)tmp_buf_host + i * STAGING_BUF_SIZE;
    }

    CR_initialized = true;
    fprintf(stderr, "[init_CR] Initialization complete, setting CR_initialized = true\n");
}

static void handle_cr_signal(int signum) {
    fprintf(stderr, "[vGPU] Received signal %d from process %d\n", signum, getpid());
    fflush(stderr);
    
    // Only handle our specific signals
    if (signum != CR_INIT_SIGNAL && signum != CR_CKPT_SIGNAL && signum != CR_RESTORE_SIGNAL) {
        fprintf(stderr, "[vGPU] Ignoring unknown signal %d (not a CR signal)\n", signum);
        return;
    }
    
    if(signum == CR_INIT_SIGNAL) {
        if (!CR_initialized) {
            fprintf(stderr, "[vGPU] Starting init_CR()...\n");
            init_CR();
            fprintf(stderr, "[vGPU] CR initialization complete\n");
        } else {
            fprintf(stderr, "[vGPU] CR already initialized, skipping\n");
        }
        comm->send_msg(FINISH_MSG);
        fprintf(stderr, "[vGPU] FINISH_MSG sent, returning from signal handler\n");
        fflush(stderr);
        return;
    }

    if(!CR_initialized) {
        fprintf(stderr, "CR not initialized, initializing now...\n");
        init_CR();
    }

    uint32_t msg = comm->recv_msg();
    if(msg == CKPT_MSG) {
        fprintf(stderr, "waiting for kernels to finish...\n");
        gpu->syncAllKernels();
        fprintf(stderr, "start ckpt...\n");
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = ckpt();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        fprintf(stderr, "ckpt size: %f GB, time: %ld ms, bw: %f GB/s\n",
               tot_size / 1024 / 1024 / 1024, duration.count(),
               tot_size / duration.count() * 1000 / 1024 / 1024 / 1024);

        // Disable P2P peer access before cuda-checkpoint freeze.
        // P2P access creates driver-level state that cuda-checkpoint cannot restore.
        // This must happen AFTER ckpt() (data is saved) and BEFORE cuda-checkpoint runs.
#if !defined(__HIP_PLATFORM_AMD__)
        fprintf(stderr, "[vGPU] Disabling P2P peer access for cuda-checkpoint...\n");
        ipc_disable_all_peer_access();
#endif
        // Note: External checkpoint (cuda-checkpoint for NVIDIA, CRIU for AMD)
        // is called from cr_client, not here
    } else if (msg == RESTORE_MSG) {
        // Note: cuda-checkpoint restore was already called by cr_client before this signal
        fprintf(stderr, "start restore...\n");
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = restore_ptr_and_content();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        fprintf(stderr, "restore size: %f GB, time: %ld ms, bw: %f GB/s\n",
               tot_size / 1024 / 1024 / 1024, duration.count(),
               tot_size / duration.count() * 1000 / 1024 / 1024 / 1024);

        // Re-enable P2P peer access after data restore
#if !defined(__HIP_PLATFORM_AMD__)
        fprintf(stderr, "[vGPU] Re-enabling P2P peer access after restore...\n");
        ipc_reenable_all_peer_access();
#endif
        fprintf(stderr, "finish restore\n");
    }
    comm->send_msg(FINISH_MSG);
}

// ---------------------------------------------------------------------------
// IPC teardown/rebuild signal handler (for multi-GPU checkpoint/restore)
// Replaces the old NCCL suspend/resume handler — no NCCL source mods needed.
// ---------------------------------------------------------------------------
static void handle_ipc_signal(int signum) {
    fprintf(stderr, "[vGPU-IPC] Received signal %d (PID=%d)\n", signum, getpid());
    fflush(stderr);

    if (!CR_initialized) {
        fprintf(stderr, "[vGPU-IPC] CR not initialized, initializing now...\n");
        init_CR();
    }

    uint32_t msg = comm->recv_msg();

    if (msg == IPC_TEARDOWN_MSG) {
        // === Checkpoint Phase 1: Teardown IPC state ===
        fprintf(stderr, "[vGPU-IPC] === IPC Teardown Phase === (imports=%d, exports=%d, events=%d)\n",
                ipc_get_import_count(), ipc_get_export_count(), ipc_get_event_count());
        fflush(stderr);

        auto t_phase_start = std::chrono::high_resolution_clock::now();

        // GPU sync
        fprintf(stderr, "[vGPU-IPC] Synchronizing GPU (waiting for in-flight kernels)...\n");
        gpu->syncAllKernels();
        auto t_sync = std::chrono::high_resolution_clock::now();
        auto sync_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_sync - t_phase_start).count();
        fprintf(stderr, "[vGPU-IPC] GPU synchronized (%ld ms)\n", sync_ms);

        auto t0 = std::chrono::high_resolution_clock::now();

        // Diagnostic: dump IPC state and nvidia fds BEFORE teardown
        ipc_dump_state();
        ipc_dump_nvidia_fds("BEFORE teardown");

        // Teardown all imported IPC mappings (cuMemUnmap + cuMemRelease)
        auto t_imports = std::chrono::high_resolution_clock::now();
        int torn = ipc_teardown_all_imports();
        auto t_imports_end = std::chrono::high_resolution_clock::now();
        auto imports_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_imports_end - t_imports).count();
        fprintf(stderr, "[vGPU-IPC] Torn down %d IPC imports (%ld ms)\n", torn, imports_ms);

        // Save export GPU data to host buffer, then fully teardown exports
        size_t export_data_needed = ipc_get_export_data_size();
        fprintf(stderr, "[vGPU-IPC] Export data size needed: %zu bytes\n", export_data_needed);

        if (export_data_needed > 0) {
            if (g_ipc_export_data_buf) {
                munmap(g_ipc_export_data_buf, g_ipc_export_data_size);
                g_ipc_export_data_buf = nullptr;
            }
            g_ipc_export_data_size = export_data_needed;
            g_ipc_export_data_buf = mmap(nullptr, g_ipc_export_data_size,
                                          PROT_READ | PROT_WRITE,
                                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (g_ipc_export_data_buf == MAP_FAILED) {
                fprintf(stderr, "[vGPU-IPC] ERROR: mmap for export data buffer failed\n");
                g_ipc_export_data_buf = nullptr;
                g_ipc_export_data_size = 0;
            }
        }

        auto t_exports = std::chrono::high_resolution_clock::now();
        int export_torn = 0;
        if (g_ipc_export_data_buf && g_ipc_export_data_size > 0) {
            export_torn = ipc_save_and_teardown_all_exports(
                g_ipc_export_data_buf, g_ipc_export_data_size);
            fprintf(stderr, "[vGPU-IPC] Export save+teardown: %d exports processed\n", export_torn);
        } else if (export_data_needed == 0) {
            fprintf(stderr, "[vGPU-IPC] No export data to save (0 mapped exports)\n");
        }
        auto t_exports_end = std::chrono::high_resolution_clock::now();
        auto exports_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_exports_end - t_exports).count();
        fprintf(stderr, "[vGPU-IPC] Export teardown total: %ld ms\n", exports_ms);

        // Teardown IPC events
        auto t_events = std::chrono::high_resolution_clock::now();
        int events_torn = ipc_teardown_all_events();
        auto t_events_end = std::chrono::high_resolution_clock::now();
        auto events_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_events_end - t_events).count();
        fprintf(stderr, "[vGPU-IPC] IPC events torn down: %d (%ld ms)\n", events_torn, events_ms);

        // Teardown non-exported cuMem allocs
        size_t local_alloc_needed = ipc_get_local_alloc_data_size();
        fprintf(stderr, "[vGPU-IPC] Local cuMem alloc data size: %zu bytes\n", local_alloc_needed);

        if (local_alloc_needed > 0) {
            if (g_local_alloc_data_buf) {
                munmap(g_local_alloc_data_buf, g_local_alloc_data_size);
                g_local_alloc_data_buf = nullptr;
            }
            g_local_alloc_data_size = local_alloc_needed;
            g_local_alloc_data_buf = mmap(nullptr, g_local_alloc_data_size,
                                          PROT_READ | PROT_WRITE,
                                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (g_local_alloc_data_buf == MAP_FAILED) {
                fprintf(stderr, "[vGPU-IPC] ERROR: mmap for local alloc buffer failed\n");
                g_local_alloc_data_buf = nullptr;
                g_local_alloc_data_size = 0;
            }
        }

        auto t_local = std::chrono::high_resolution_clock::now();
        if (g_local_alloc_data_buf && g_local_alloc_data_size > 0) {
            int local_torn = ipc_save_and_teardown_local_allocs(
                g_local_alloc_data_buf, g_local_alloc_data_size);
            fprintf(stderr, "[vGPU-IPC] Local alloc save+teardown: %d allocs processed\n", local_torn);
        } else if (local_alloc_needed == 0) {
            fprintf(stderr, "[vGPU-IPC] No local cuMem allocs to teardown\n");
        }
        auto t_local_end = std::chrono::high_resolution_clock::now();
        auto local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_local_end - t_local).count();
        fprintf(stderr, "[vGPU-IPC] Local alloc teardown total: %ld ms\n", local_ms);

        // Diagnostic: dump nvidia fds AFTER teardown
        ipc_dump_nvidia_fds("AFTER teardown");

        // Disable P2P peer access
#if !defined(__HIP_PLATFORM_AMD__)
        auto t_p2p = std::chrono::high_resolution_clock::now();
        ipc_disable_all_peer_access();
        auto t_p2p_end = std::chrono::high_resolution_clock::now();
        auto p2p_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_p2p_end - t_p2p).count();
        fprintf(stderr, "[vGPU-IPC] P2P peer access disabled (%ld ms)\n", p2p_ms);
#endif

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t_phase_start).count();
        fprintf(stderr, "[vGPU-IPC] IPC teardown completed in %ld ms (excl. GPU sync)\n", ms);
        fprintf(stderr, "[vGPU-IPC] === Teardown Timing Summary: GPU-sync=%ld, Imports=%ld, Exports=%ld, Events=%ld, LocalAllocs=%ld, Total=%ld ms ===\n",
                sync_ms, imports_ms, exports_ms, events_ms, local_ms, total_ms);

    } else if (msg == IPC_EXPORT_MSG) {
        // === Restore Phase 3a: Re-export and publish handle info ===
        fprintf(stderr, "[vGPU-IPC] === IPC Re-export Phase ===\n");
        fflush(stderr);

        auto t0 = std::chrono::high_resolution_clock::now();

        // Rebuild export allocations at original VAs, restore GPU data, re-export
        auto t_exports = std::chrono::high_resolution_clock::now();
        int rebuilt = 0;
        if (g_ipc_export_data_buf && g_ipc_export_data_size > 0) {
            rebuilt = ipc_rebuild_and_restore_all_exports(
                g_ipc_export_data_buf, g_ipc_export_data_size);
            fprintf(stderr, "[vGPU-IPC] Rebuilt+restored %d export allocations\n", rebuilt);

            munmap(g_ipc_export_data_buf, g_ipc_export_data_size);
            g_ipc_export_data_buf = nullptr;
            g_ipc_export_data_size = 0;
        } else {
            fprintf(stderr, "[vGPU-IPC] No export data to restore (buffer empty)\n");
        }
        auto t_exports_end = std::chrono::high_resolution_clock::now();
        auto exports_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_exports_end - t_exports).count();

        // Rebuild non-exported cuMem allocs
        auto t_local = std::chrono::high_resolution_clock::now();
        if (g_local_alloc_data_buf && g_local_alloc_data_size > 0) {
            int local_rebuilt = ipc_rebuild_local_allocs(
                g_local_alloc_data_buf, g_local_alloc_data_size);
            fprintf(stderr, "[vGPU-IPC] Rebuilt %d local cuMem allocs\n", local_rebuilt);

            munmap(g_local_alloc_data_buf, g_local_alloc_data_size);
            g_local_alloc_data_buf = nullptr;
            g_local_alloc_data_size = 0;
        }
        auto t_local_end = std::chrono::high_resolution_clock::now();
        auto local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_local_end - t_local).count();

        // Write export info to shared memory for peers to read
        auto t_shm = std::chrono::high_resolution_clock::now();
        void* tmp_buf = backend->get_tmp_buf();
        shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;
        IpcRebuildShmBlock* my_block = (IpcRebuildShmBlock*)((char*)tmp_buf +
            ROUND_UP_2MB(sizeof(shared_mem_fs)) - sizeof(IpcRebuildShmBlock));
        ipc_write_export_info_to_shm(my_block);
        auto t_shm_end = std::chrono::high_resolution_clock::now();
        auto shm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_shm_end - t_shm).count();

        // Start UDS fd server
        auto t_uds = std::chrono::high_resolution_clock::now();
        uds_fd_server_start();
        auto t_uds_end = std::chrono::high_resolution_clock::now();
        auto uds_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_uds_end - t_uds).count();

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        fprintf(stderr, "[vGPU-IPC] IPC re-export completed in %ld ms\n", ms);
        fprintf(stderr, "[vGPU-IPC] === Re-export Timing Summary: Exports=%ld, LocalAllocs=%ld, SHM-write=%ld, UDS-server=%ld, Total=%ld ms ===\n",
                exports_ms, local_ms, shm_ms, uds_ms, ms);

    } else if (msg == IPC_IMPORT_MSG) {
        // === Restore Phase 3b: Import from peers ===
        fprintf(stderr, "[vGPU-IPC] === IPC Re-import Phase ===\n");
        fflush(stderr);

        auto t0 = std::chrono::high_resolution_clock::now();

        void* tmp_buf = backend->get_tmp_buf();
        IpcRebuildShmBlock* peer_block = (IpcRebuildShmBlock*)((char*)tmp_buf +
            ROUND_UP_2MB(sizeof(shared_mem_fs)) - sizeof(IpcRebuildShmBlock) * 2);

        auto t_import = std::chrono::high_resolution_clock::now();
        if (peer_block->num_exports > 0) {
            int imported = ipc_import_from_shm_block(peer_block);
            fprintf(stderr, "[vGPU-IPC] Imported %d mappings from peers\n", imported);
        } else {
            fprintf(stderr, "[vGPU-IPC] No peer exports to import\n");
        }
        auto t_import_end = std::chrono::high_resolution_clock::now();
        auto import_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_import_end - t_import).count();

        // Stop UDS fd server
        auto t_uds_stop = std::chrono::high_resolution_clock::now();
        uds_fd_server_stop();
        auto t_uds_stop_end = std::chrono::high_resolution_clock::now();
        auto uds_stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_uds_stop_end - t_uds_stop).count();

        // Validate all IPC mappings after rebuild
        auto t_validate = std::chrono::high_resolution_clock::now();
        ipc_validate_all_mappings("AFTER import rebuild");
        auto t_validate_end = std::chrono::high_resolution_clock::now();
        auto validate_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_validate_end - t_validate).count();

        // Re-enable P2P peer access
#if !defined(__HIP_PLATFORM_AMD__)
        auto t_p2p = std::chrono::high_resolution_clock::now();
        ipc_reenable_all_peer_access();
        auto t_p2p_end = std::chrono::high_resolution_clock::now();
        auto p2p_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_p2p_end - t_p2p).count();
#else
        long p2p_ms = 0;
#endif

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        fprintf(stderr, "[vGPU-IPC] IPC re-import completed in %ld ms\n", ms);
        fprintf(stderr, "[vGPU-IPC] === Re-import Timing Summary: Import=%ld, UDS-stop=%ld, Validate=%ld, P2P=%ld, Total=%ld ms ===\n",
                import_ms, uds_stop_ms, validate_ms, p2p_ms, ms);

    } else {
        fprintf(stderr, "[vGPU-IPC] WARNING: unexpected message %u\n", msg);
    }

    comm->send_msg(FINISH_MSG);
    fflush(stderr);
}

// ---------------------------------------------------------------------------
// Library constructor: register all signal handlers
// ---------------------------------------------------------------------------
__attribute__((constructor)) void init() {
    fprintf(stderr, "[vGPU] Library loaded! Registering signal handlers...\n");
    fprintf(stderr, "[vGPU] Multi-GPU CR support enabled (IPC hook mode)\n");
    fflush(stderr);

    pthread_atfork(control_atfork_prepare, control_atfork_parent,
                   control_atfork_child);
    gpu_cr_ensure_control_thread();

    // Signal handlers only enqueue a small integer.  mmap, allocation, CUDA,
    // mutex, logging and coordinator communication all run on the controller.
    signal(CR_INIT_SIGNAL, queue_control_signal);
    signal(CR_CKPT_SIGNAL, queue_control_signal);
    signal(CR_RESTORE_SIGNAL, queue_control_signal);

    // Multi-GPU IPC teardown/rebuild signals (replaces NCCL suspend/resume)
    signal(CR_IPC_TEARDOWN_SIGNAL, queue_control_signal);
    signal(CR_IPC_REBUILD_SIGNAL, queue_control_signal);

    // Diagnostic: validate all IPC mappings on demand
    signal(CR_IPC_VALIDATE_SIGNAL, queue_control_signal);
}
