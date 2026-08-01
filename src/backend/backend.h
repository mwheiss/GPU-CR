#ifndef BACKEND_H
#define BACKEND_H

#include <mutex>

class Backend {
public:
    Backend(int id);
    virtual ~Backend();
    virtual void setup();
    virtual void* get_tmp_buf() = 0;
    virtual void* get_host_buffer() = 0;
};

class ShareMem : public Backend {
public:
    void *tmp_buf = nullptr;
    void* host_buf_ptr = nullptr;

    std::mutex fs_mutex;

    int id;

    ShareMem(int id);
    ~ShareMem();
    void setup() override;
    void* get_tmp_buf() override;
    void* get_host_buffer();
};

#endif
