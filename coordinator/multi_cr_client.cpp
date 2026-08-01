/**
 * multi_cr_client.cpp - Multi-GPU Checkpoint/Restore Orchestrator
 *
 * This tool coordinates checkpoint/restore across multiple worker processes,
 * each owning one GPU.  It implements a phased protocol:
 *
 * === Checkpoint (-c) ===
 *   Phase 1  NCCL Suspend   — signal ALL workers to call ncclCommSuspend()
 *            (collective, all ranks must enter simultaneously via barrier)
 *   Phase 2  Data Dump      — signal ALL workers to dump GPU memory to host
 *            (each worker independent, using existing vGPU.so ckpt logic)
 *   Phase 3  CUDA Freeze    — call cuda-checkpoint --toggle for each worker
 *            (each worker independent)
 *
 * === Restore (-r) ===
 *   Phase 1  CUDA Unfreeze  — call cuda-checkpoint --toggle for each worker
 *   Phase 2  Data Restore   — signal ALL workers to restore GPU memory
 *   Phase 3  NCCL Resume    — signal ALL workers to call ncclCommResume()
 *            (collective, all ranks must enter simultaneously)
 *
 * === Init (-i) ===
 *   Signal ALL workers to initialize the CR subsystem (vGPU.so init_CR).
 *
 * Usage:
 *   multi_cr_client -c -p pid1,pid2,pid3,pid4          # checkpoint
 *   multi_cr_client -r -p pid1,pid2,pid3,pid4          # restore
 *   multi_cr_client -i -p pid1,pid2,pid3,pid4          # init
 *   multi_cr_client -c -p pid1,pid2 -b                 # buffer only (skip cuda-checkpoint)
 *   multi_cr_client -c -p pid1,pid2 -n                 # no NCCL (skip nccl suspend/resume)
 *
 * The -p flag accepts comma-separated PIDs.  The order does not matter.
 * For backward compatibility with single-GPU, a single PID works as well.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include <thread>

#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <limits.h>
#include <libgen.h>

#ifdef __HIP_PLATFORM_AMD__
// AMD platform
#else
#include <cuda.h>
#endif

#include "common.h"
#include "comm/comm.h"
#include "ipc_hooks.h"   // for IpcRebuildShmBlock, IpcExportShmEntry

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string get_cuda_checkpoint_path() {
    // Determine the project root from the executable location.
    // Executable is at <PROJECT_ROOT>/build/multi_cr_client.
    char exe_path[1024];
    ssize_t count = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (count == -1) {
        perror("readlink");
        return "";
    }
    exe_path[count] = '\0';

    // dirname may modify the buffer, so work with copies.
    char* build_dir = dirname(exe_path);            // e.g. /path/to/GPU-CR/build
    std::string project_root = std::string(build_dir) + "/..";

    // Resolve to a canonical absolute path (eliminates "..")
    char resolved[PATH_MAX];
    if (realpath(project_root.c_str(), resolved)) {
        project_root = resolved;
    }

    // The cuda-checkpoint binary is shipped pre-built directly under the
    // project root, NOT inside build/.
    const char* rel = "/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint";

    // Try several candidate locations:
    std::string candidates[] = {
        project_root + rel,          // <PROJECT_ROOT>/cuda-checkpoint/bin/...
        std::string(build_dir) + rel // <BUILD_DIR>/cuda-checkpoint/bin/... (legacy)
    };

    for (auto& path : candidates) {
        if (access(path.c_str(), F_OK) == 0) {
            // Ensure execute permission
            if (access(path.c_str(), X_OK) != 0) {
                chmod(path.c_str(), 0755);
            }
            fprintf(stderr, "[cuda-checkpoint] Found at: %s\n", path.c_str());
            return path;
        }
    }

    fprintf(stderr, "ERROR: cuda-checkpoint binary not found.\n"
                    "  Searched:\n");
    for (auto& path : candidates) {
        fprintf(stderr, "    %s\n", path.c_str());
    }
    fprintf(stderr, "  Please ensure cuda-checkpoint/ is in the project root.\n");
    return "";
}

/** Parse "pid1,pid2,pid3" into a vector of ints. */
static std::vector<int> parse_pids(const char* arg) {
    std::vector<int> pids;
    char* buf = strdup(arg);
    char* token = strtok(buf, ",");
    while (token) {
        int pid = atoi(token);
        if (pid > 0) pids.push_back(pid);
        token = strtok(nullptr, ",");
    }
    free(buf);
    return pids;
}

/** Elapsed time helper, returns seconds. */
static double elapsed_sec(std::chrono::high_resolution_clock::time_point t0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

static long control_timeout_ms() {
    const char* value = std::getenv("GPU_CR_CONTROL_TIMEOUT_MS");
    if (!value || !*value) value = std::getenv("GPU_CR_LOCK_TIMEOUT_MS");
    if (!value || !*value) return 600000;
    char* end = nullptr;
    long timeout = std::strtol(value, &end, 10);
    return end != value && timeout > 0 ? timeout : 600000;
}

// ---------------------------------------------------------------------------
// Per-process communication channels
// ---------------------------------------------------------------------------
struct WorkerCtx {
    int   pid;
    int   cr_id;    // CR ID (index into /mnt/huge-ckpt/{id})
    Comm* comm;
    void* shm_ptr;  // mmap'd pointer to worker's checkpoint shm (lazily opened)
};

static std::vector<WorkerCtx> workers;

static void setup_workers(const std::vector<int>& pids) {
    for (size_t i = 0; i < pids.size(); i++) {
        Comm* c = new ShareMemComm(pids[i]);
        c->setup();
        workers.push_back({pids[i], (int)i, c, nullptr});
    }
}

// ---------------------------------------------------------------------------
// Shared memory access helpers for IPC export/import exchange
// ---------------------------------------------------------------------------

/**
 * Open (mmap) a worker's checkpoint shared memory by CR ID.
 * Returns the mmap'd pointer, or nullptr on failure.
 */
static void* open_worker_shm(int cr_id) {
    char shm_path[512];
    const char* export_file_path = std::getenv("EXPORT_FILE_PATH");

    if (export_file_path) {
        sprintf(shm_path, "%s/ckpt-%d.data", export_file_path, cr_id);
    } else {
        sprintf(shm_path, "/mnt/huge-ckpt/%d", cr_id);
    }

    int fd = open(shm_path, O_RDWR, 0755);
    if (fd < 0) {
        fprintf(stderr, "[SHM] ERROR: Cannot open %s: %s\n", shm_path, strerror(errno));
        return nullptr;
    }

    void* ptr = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "[SHM] ERROR: mmap failed for %s: %s\n", shm_path, strerror(errno));
        return nullptr;
    }

    fprintf(stderr, "[SHM] Opened worker shm cr_id=%d at %p (%s)\n", cr_id, ptr, shm_path);
    return ptr;
}

/** Get the my_block location within a worker's shm (same layout as vGPU.cpp) */
static IpcRebuildShmBlock* get_my_block(void* shm_ptr) {
    return (IpcRebuildShmBlock*)((char*)shm_ptr +
        ROUND_UP_2MB(sizeof(shared_mem_fs)) - sizeof(IpcRebuildShmBlock));
}

/** Get the peer_block location within a worker's shm (same layout as vGPU.cpp) */
static IpcRebuildShmBlock* get_peer_block(void* shm_ptr) {
    return (IpcRebuildShmBlock*)((char*)shm_ptr +
        ROUND_UP_2MB(sizeof(shared_mem_fs)) - sizeof(IpcRebuildShmBlock) * 2);
}

/**
 * After Phase 3a (re-export), cross-copy export info between shm files.
 *
 * CR IDs are assigned by an atomic counter (get_id()) and may not match
 * the PID list order. So we work directly with shm files 0..N-1:
 *   1. Open all N shm files
 *   2. Read each shm's my_block (which contains owner_pid in entries)
 *   3. For each shm file, write all OTHER shm files' exports into its peer_block
 *
 * The owner_pid field in each export entry ensures pidfd_getfd targets
 * the correct process during import, regardless of shm-to-PID mapping.
 */
static void exchange_ipc_export_info() {
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t n = workers.size();

    // Open all N shm files (cr_id 0 through N-1)
    std::vector<void*> shm_ptrs(n, nullptr);
    for (size_t i = 0; i < n; i++) {
        shm_ptrs[i] = open_worker_shm((int)i);
        if (!shm_ptrs[i]) {
            fprintf(stderr, "[IPC-Exchange] FATAL: Cannot open shm cr_id=%zu\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Read each shm's my_block (export info written by the worker)
    std::vector<IpcRebuildShmBlock> export_blocks(n);
    for (size_t i = 0; i < n; i++) {
        IpcRebuildShmBlock* src = get_my_block(shm_ptrs[i]);
        memcpy(&export_blocks[i], src, sizeof(IpcRebuildShmBlock));
        int owner_pid = (export_blocks[i].num_exports > 0)
                        ? export_blocks[i].entries[0].owner_pid : -1;
        fprintf(stderr, "[IPC-Exchange] shm[%zu]: %d exports (owner_pid=%d)\n",
                i, export_blocks[i].num_exports, owner_pid);
    }

    // For each shm file, write all OTHER shm files' exports into its peer_block
    for (size_t i = 0; i < n; i++) {
        IpcRebuildShmBlock* dst = get_peer_block(shm_ptrs[i]);
        dst->num_exports = 0;

        for (size_t j = 0; j < n; j++) {
            if (j == i) continue;  // skip same shm file

            for (int k = 0; k < export_blocks[j].num_exports; k++) {
                if (dst->num_exports >= IPC_MAX_EXPORTS_PER_PROC) {
                    fprintf(stderr, "[IPC-Exchange] WARNING: peer_block overflow for shm[%zu]\n", i);
                    break;
                }
                dst->entries[dst->num_exports] = export_blocks[j].entries[k];
                dst->num_exports++;
            }
        }

        fprintf(stderr, "[IPC-Exchange] shm[%zu] peer_block: %d peer exports written\n",
                i, dst->num_exports);
    }

    // Cleanup: unmap the shm files we opened
    for (size_t i = 0; i < n; i++) {
        if (shm_ptrs[i]) {
            munmap(shm_ptrs[i], SHM_SIZE);
        }
    }

    printf("[IPC-Exchange] Export info exchanged across %zu shm files (%.3f s)\n",
           n, elapsed_sec(t0));
}

// ---------------------------------------------------------------------------
// Phase helpers: send signal/message to ALL workers, then wait for ALL
// ---------------------------------------------------------------------------

/**
 * Send a message + signal to ALL workers, then wait for ALL to finish.
 *
 * CRITICAL: For collective NCCL operations (suspend/resume), we must send
 * signals to ALL workers BEFORE waiting for any to complete.  Otherwise the
 * first worker blocks on the NCCL internal barrier waiting for the others.
 */
static void broadcast_and_wait(uint32_t msg, int sig, const char* phase_name) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: Send message + signal to ALL workers
    for (auto& w : workers) {
        w.comm->send_msg(msg);
        if (kill(w.pid, sig) != 0) {
            fprintf(stderr, "[%s] ERROR: kill(%d, %d) failed: %s\n",
                    phase_name, w.pid, sig, strerror(errno));
            exit(EXIT_FAILURE);
        }
    }
    printf("[%s] Signals sent to %zu workers\n", phase_name, workers.size());

    // Step 2: Wait for ALL workers to report FINISH_MSG
    for (auto& w : workers) {
        while (!w.comm->is_finished()) {
            if (kill(w.pid, 0) != 0 && errno == ESRCH) {
                fprintf(stderr, "[%s] ERROR: worker %d exited while waiting\n",
                        phase_name, w.pid);
                exit(EXIT_FAILURE);
            }
            if (elapsed_sec(t0) * 1000.0 > control_timeout_ms()) {
                fprintf(stderr,
                        "[%s] ERROR: timed out after %ld ms waiting for worker %d\n",
                        phase_name, control_timeout_ms(), w.pid);
                exit(EXIT_FAILURE);
            }
            usleep(1000);
        }
    }

    printf("[%s] All %zu workers completed (%.3f s)\n",
           phase_name, workers.size(), elapsed_sec(t0));
}

// ---------------------------------------------------------------------------
// cuda-checkpoint modes:
//   Legacy (--toggle):  single call toggles freeze/unfreeze
//   580    (--action):   lock → checkpoint → restore → unlock (4-step)
//
// The 580 mode is preferred because the legacy --toggle mode may have
// issues with the restore path on newer drivers.
// ---------------------------------------------------------------------------
static bool g_use_action_mode = false;  // set from command line -a flag
static std::string g_cuda_ckpt_path;

static bool init_cuda_checkpoint_path() {
    if (g_cuda_ckpt_path.empty()) {
        g_cuda_ckpt_path = get_cuda_checkpoint_path();
    }
    return !g_cuda_ckpt_path.empty();
}

/**
 * Detect whether cuda-checkpoint supports --action mode (580+).
 * Runs cuda-checkpoint --help and checks for '--action'.
 */
static bool detect_action_mode_support() {
    if (!init_cuda_checkpoint_path()) return false;
    std::string cmd = "\"" + g_cuda_ckpt_path + "\" --help 2>&1";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return false;
    char buf[512];
    bool found = false;
    while (fgets(buf, sizeof(buf), fp)) {
        if (strstr(buf, "--action")) {
            found = true;
            break;
        }
    }
    pclose(fp);
    return found;
}

/**
 * Run a single cuda-checkpoint command for one worker.
 * Thread-safe: uses fork+execvp instead of system().
 * Returns 0 on success, non-zero on failure.
 */
static int cuda_ckpt_run(int pid, const char* args, const char* label) {
    auto tc0 = std::chrono::high_resolution_clock::now();

    // Build argv for execvp. Parse args string into tokens.
    std::string pid_str = std::to_string(pid);
    std::vector<std::string> arg_tokens;
    {
        std::string token;
        for (const char* p = args; *p; ++p) {
            if (*p == ' ') {
                if (!token.empty()) { arg_tokens.push_back(token); token.clear(); }
            } else {
                token += *p;
            }
        }
        if (!token.empty()) arg_tokens.push_back(token);
    }

    // Build argv: [cuda-checkpoint, args..., --pid, <pid>, NULL]
    std::vector<const char*> argv;
    argv.push_back(g_cuda_ckpt_path.c_str());
    for (auto& t : arg_tokens) argv.push_back(t.c_str());
    argv.push_back("--pid");
    argv.push_back(pid_str.c_str());
    argv.push_back(nullptr);

    pid_t child = fork();
    if (child < 0) {
        perror("fork()");
        return -1;
    }
    if (child == 0) {
        // Child: exec cuda-checkpoint
        execvp(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127); // exec failed
    }

    // Parent: wait for child
    int status = 0;
    if (waitpid(child, &status, 0) < 0) {
        perror("waitpid()");
        return -1;
    }
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    printf("[%s] cuda-checkpoint %s PID %d (%.3f s, ret=%d)\n",
           label, args, pid, elapsed_sec(tc0), exit_code);
    return exit_code;
}

/**
 * Run cuda-checkpoint in PARALLEL for ALL workers using std::thread.
 * Each worker gets its own fork+exec. Returns number of failures (0 = all success).
 */
static int cuda_ckpt_all_parallel(const char* args, const char* phase_name) {
    size_t n = workers.size();
    std::vector<int> results(n, -1);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < n; i++) {
        threads.emplace_back([i, args, phase_name, &results]() {
            results[i] = cuda_ckpt_run(workers[i].pid, args, phase_name);
        });
    }
    for (auto& t : threads) t.join();

    int failures = 0;
    for (size_t i = 0; i < n; i++) {
        if (results[i] != 0) {
            failures++;
            fprintf(stderr, "[%s] ERROR: cuda-checkpoint %s failed for PID %d\n",
                    phase_name, args, workers[i].pid);
        }
    }
    return failures;
}

/**
 * Run a cuda-checkpoint action for ALL workers SEQUENTIALLY (legacy fallback).
 * Returns number of failures (0 = all success).
 */
static int cuda_ckpt_all(const char* args, const char* phase_name) {
    int failures = 0;
    for (auto& w : workers) {
        if (cuda_ckpt_run(w.pid, args, phase_name) != 0) {
            failures++;
            fprintf(stderr, "[%s] ERROR: cuda-checkpoint %s failed for PID %d\n",
                    phase_name, args, w.pid);
        }
    }
    return failures;
}

// ---- Legacy toggle mode ----

/** Returns number of failures (0 = all success). */
static int cuda_checkpoint_toggle_all(const char* phase_name, bool buffer_only) {
    if (buffer_only) {
        printf("[%s] Skipped (buffer_only mode)\n", phase_name);
        return 0;
    }

#if !defined(__HIP_PLATFORM_AMD__)
    if (!init_cuda_checkpoint_path()) {
        fprintf(stderr, "[%s] ERROR: cuda-checkpoint not found, skipping.\n", phase_name);
        return (int)workers.size();
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    // Driver 610 checkpoint jobs currently require per-process operations to
    // be invoked sequentially.  Data copies still run concurrently in the
    // workers; only the driver control pass is serialized.
    int failures = cuda_ckpt_all("--toggle", phase_name);
    printf("[%s] All %zu workers toggled (%d failures, %.3f s total)\n",
           phase_name, workers.size(), failures, elapsed_sec(t0));
    return failures;
#else
    printf("[%s] cuda-checkpoint not used on this platform\n", phase_name);
    return 0;
#endif
}

// ---- 580 action mode: lock → checkpoint / restore → unlock ----

/** Freeze all workers using 580 action mode. Returns number of failures. */
static int cuda_checkpoint_freeze_action(const char* phase_name, bool buffer_only) {
    if (buffer_only) {
        printf("[%s] Skipped (buffer_only mode)\n", phase_name);
        return 0;
    }
#if !defined(__HIP_PLATFORM_AMD__)
    if (!init_cuda_checkpoint_path()) {
        fprintf(stderr, "[%s] ERROR: cuda-checkpoint not found.\n", phase_name);
        return (int)workers.size();
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    int failures = 0;

    // Lock every member before checkpointing any.  Keep the driver operations
    // sequential for driver-610 job/IPC compatibility.
    const char* timeout = std::getenv("GPU_CR_LOCK_TIMEOUT_MS");
    std::string lock_args = "--action lock --timeout ";
    lock_args += timeout ? timeout : "600000";
    printf("[%s] Step 1: Lock all workers (sequential)...\n", phase_name);
    failures = cuda_ckpt_all(lock_args.c_str(), phase_name);
    if (failures > 0) {
        fprintf(stderr, "[%s] Lock failed for %d worker(s), aborting freeze\n",
                phase_name, failures);
        // Unlock any that succeeded
        cuda_ckpt_all("--action unlock", phase_name);
        return failures;
    }

    printf("[%s] Step 2: Checkpoint all workers (sequential)...\n", phase_name);
    failures = cuda_ckpt_all("--action checkpoint", phase_name);
    if (failures > 0) {
        fprintf(stderr, "[%s] Checkpoint failed for %d worker(s)\n",
                phase_name, failures);
        // Try to unlock the ones that were locked
        cuda_ckpt_all("--action unlock", phase_name);
        return failures;
    }

    printf("[%s] All %zu workers frozen via --action mode (%d failures, %.3f s)\n",
           phase_name, workers.size(), failures, elapsed_sec(t0));
    return failures;
#else
    printf("[%s] cuda-checkpoint not used on this platform\n", phase_name);
    return 0;
#endif
}

/** Restore all workers using 580 action mode. Returns number of failures. */
static int cuda_checkpoint_restore_action(const char* phase_name, bool buffer_only) {
    if (buffer_only) {
        printf("[%s] Skipped (buffer_only mode)\n", phase_name);
        return 0;
    }
#if !defined(__HIP_PLATFORM_AMD__)
    if (!init_cuda_checkpoint_path()) {
        fprintf(stderr, "[%s] ERROR: cuda-checkpoint not found.\n", phase_name);
        return (int)workers.size();
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    int failures = 0;

    printf("[%s] Step 1: Restore all workers (sequential)...\n", phase_name);
    failures = cuda_ckpt_all("--action restore", phase_name);
    if (failures > 0) {
        fprintf(stderr, "[%s] Restore failed for %d worker(s)\n",
                phase_name, failures);
    }

    printf("[%s] Step 2: Unlock all workers (sequential)...\n", phase_name);
    int unlock_failures = cuda_ckpt_all("--action unlock", phase_name);
    failures += unlock_failures;

    printf("[%s] All %zu workers restored via --action mode (%d failures, %.3f s)\n",
           phase_name, workers.size(), failures, elapsed_sec(t0));
    return failures;
#else
    printf("[%s] cuda-checkpoint not used on this platform\n", phase_name);
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Checkpoint flow
// ---------------------------------------------------------------------------
static void do_checkpoint(bool buffer_only, bool skip_nccl) {
    auto t_total = std::chrono::high_resolution_clock::now();

    printf("\n====== Multi-GPU Checkpoint (%zu workers, %s mode) ======\n",
           workers.size(), g_use_action_mode ? "action" : "toggle");

    // Phase 1: IPC Teardown (replaces NCCL Suspend)
    // Tears down all cuMem IPC mappings so cuda-checkpoint sees clean CUDA state
    if (!skip_nccl) {
        printf("\n--- Phase 1: IPC Teardown ---\n");
        broadcast_and_wait(IPC_TEARDOWN_MSG, CR_IPC_TEARDOWN_SIGNAL, "IPC-Teardown");
    } else {
        printf("\n--- Phase 1: IPC Teardown (skipped, -n flag) ---\n");
    }

    // Phase 2: Data Dump (each worker independently)
    printf("\n--- Phase 2: Data Dump ---\n");
    broadcast_and_wait(CKPT_MSG, CR_CKPT_SIGNAL, "Data-Dump");

    // Phase 3: CUDA Control Freeze
    printf("\n--- Phase 3: CUDA Control Freeze ---\n");
    int freeze_failures;
    if (g_use_action_mode) {
        freeze_failures = cuda_checkpoint_freeze_action("CUDA-Freeze", buffer_only);
    } else {
        freeze_failures = cuda_checkpoint_toggle_all("CUDA-Freeze", buffer_only);
    }
    if (freeze_failures > 0 && !buffer_only) {
        fprintf(stderr, "\n[WARNING] cuda-checkpoint freeze FAILED for %d worker(s).\n"
                        "  Checkpoint may be incomplete — restore will probably fail.\n"
                        "  Check worker logs for details.\n", freeze_failures);
    }

    printf("\n====== Checkpoint Complete (%.3f s total) ======\n\n",
           elapsed_sec(t_total));
}

// ---------------------------------------------------------------------------
// Restore flow
// ---------------------------------------------------------------------------
static void do_restore(bool buffer_only, bool skip_nccl) {
    auto t_total = std::chrono::high_resolution_clock::now();

    printf("\n====== Multi-GPU Restore (%zu workers, %s mode) ======\n",
           workers.size(), g_use_action_mode ? "action" : "toggle");

    // Phase 1: CUDA Control Unfreeze
    printf("\n--- Phase 1: CUDA Control Unfreeze ---\n");
    int unfreeze_failures;
    if (g_use_action_mode) {
        unfreeze_failures = cuda_checkpoint_restore_action("CUDA-Unfreeze", buffer_only);
    } else {
        unfreeze_failures = cuda_checkpoint_toggle_all("CUDA-Unfreeze", buffer_only);
    }
    if (unfreeze_failures > 0 && !buffer_only) {
        fprintf(stderr, "\n[FATAL] cuda-checkpoint unfreeze FAILED for %d worker(s).\n"
                        "  Possible causes:\n"
                        "    1. Try the other mode: %s\n"
                        "    2. Residual IPC/P2P state was active during freeze\n"
                        "    3. cuda-checkpoint version incompatible with CUDA driver\n"
                        "  Aborting restore.\n",
                unfreeze_failures,
                g_use_action_mode ? "remove -a flag to use --toggle" : "add -a flag to use --action mode");
        exit(EXIT_FAILURE);
    }

    // Phase 2: Data Restore (each worker independently)
    printf("\n--- Phase 2: Data Restore ---\n");
    broadcast_and_wait(RESTORE_MSG, CR_RESTORE_SIGNAL, "Data-Restore");

    // Phase 3: IPC Rebuild (replaces NCCL Resume)
    // Three-step: re-export → exchange info via shm → re-import
    if (!skip_nccl) {
        printf("\n--- Phase 3a: IPC Re-export ---\n");
        broadcast_and_wait(IPC_EXPORT_MSG, CR_IPC_REBUILD_SIGNAL, "IPC-Export");

        printf("\n--- Phase 3 (exchange): Cross-copy export info between workers ---\n");
        exchange_ipc_export_info();

        printf("\n--- Phase 3b: IPC Re-import ---\n");
        broadcast_and_wait(IPC_IMPORT_MSG, CR_IPC_REBUILD_SIGNAL, "IPC-Import");
    } else {
        printf("\n--- Phase 3: IPC Rebuild (skipped, -n flag) ---\n");
    }

    printf("\n====== Restore Complete (%.3f s total) ======\n\n",
           elapsed_sec(t_total));
}

// ---------------------------------------------------------------------------
// Init flow
// ---------------------------------------------------------------------------
static void do_init() {
    printf("\n====== Multi-GPU Init (%zu workers) ======\n", workers.size());
    broadcast_and_wait(INIT_MSG, CR_INIT_SIGNAL, "Init");
    printf("====== Init Complete ======\n\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [-i|-c|-r] -p pid1,pid2,... [-a] [-b] [-n]\n"
        "\n"
        "Actions (mutually exclusive):\n"
        "  -i    Initialize CR subsystem in all workers\n"
        "  -c    Checkpoint all workers\n"
        "  -r    Restore all workers\n"
        "\n"
        "Options:\n"
        "  -p    Comma-separated list of worker PIDs\n"
        "  -a    Use --action mode (580+): lock/checkpoint/restore/unlock\n"
        "        Default is --toggle mode (legacy). Try -a if restore fails.\n"
        "  -b    Buffer only: skip cuda-checkpoint (data layer only)\n"
        "  -n    No NCCL: skip NCCL suspend/resume phases\n"
        "\n"
        "Examples:\n"
        "  %s -i -p 1234,1235,1236,1237\n"
        "  %s -c -p 1234,1235,1236,1237        # toggle mode\n"
        "  %s -c -p 1234,1235,1236,1237 -a     # action mode (580+)\n"
        "  %s -r -p 1234,1235,1236,1237\n"
        "  %s -c -p 1234,1235 -n               # skip NCCL\n",
        prog, prog, prog, prog, prog, prog);
}

int main(int argc, char* argv[]) {
    int opt;
    int do_init_flag    = 0;
    int do_ckpt_flag    = 0;
    int do_restore_flag = 0;
    int buffer_only     = 0;
    int skip_nccl       = 0;
    const char* pid_str = nullptr;

    int use_action_mode = 0;

    while ((opt = getopt(argc, argv, "icrnbap:h")) != -1) {
        switch (opt) {
            case 'i': do_init_flag = 1; break;
            case 'c': do_ckpt_flag = 1; break;
            case 'r': do_restore_flag = 1; break;
            case 'n': skip_nccl = 1; break;
            case 'b': buffer_only = 1; break;
            case 'a': use_action_mode = 1; break;
            case 'p': pid_str = optarg; break;
            case 'h':
            default:
                usage(argv[0]);
                exit(opt == 'h' ? 0 : EXIT_FAILURE);
        }
    }

    // Validate args
    int action_count = do_init_flag + do_ckpt_flag + do_restore_flag;
    if (action_count != 1) {
        fprintf(stderr, "ERROR: Exactly one of -i, -c, -r must be specified.\n");
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (!pid_str) {
        fprintf(stderr, "ERROR: -p <pids> is required.\n");
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    // Parse PIDs
    std::vector<int> pids = parse_pids(pid_str);
    if (pids.empty()) {
        fprintf(stderr, "ERROR: No valid PIDs in '%s'\n", pid_str);
        exit(EXIT_FAILURE);
    }

    // Determine cuda-checkpoint mode
    // Auto-detect: if --action is supported, use it by default (580+ drivers).
    // --toggle mode has known issues with restore on 580+ drivers:
    //   "OS call failed or operation not supported on this OS"
    if (use_action_mode) {
        g_use_action_mode = true;
    } else if (detect_action_mode_support()) {
        g_use_action_mode = true;
        printf("NOTE: Auto-enabled --action mode (580+ driver detected).\n"
               "      To force legacy --toggle mode, do NOT use this auto-detection.\n");
    }

    printf("Multi-GPU CR Client: %zu worker(s), mode=%s, PIDs:",
           pids.size(), g_use_action_mode ? "action" : "toggle");
    for (int p : pids) printf(" %d", p);
    printf("\n");

    if (pids.size() > MAX_MULTI_GPU_PROCS) {
        fprintf(stderr, "ERROR: Too many PIDs (%zu > MAX_MULTI_GPU_PROCS=%d)\n",
                pids.size(), MAX_MULTI_GPU_PROCS);
        exit(EXIT_FAILURE);
    }

    // Set up shared-memory communication with each worker
    setup_workers(pids);

    // Execute
    if (do_init_flag) {
        do_init();
    } else if (do_ckpt_flag) {
        do_checkpoint(buffer_only != 0, skip_nccl != 0);
    } else if (do_restore_flag) {
        do_restore(buffer_only != 0, skip_nccl != 0);
    }

    // Cleanup
    for (auto& w : workers) {
        delete w.comm;
    }

    return 0;
}
