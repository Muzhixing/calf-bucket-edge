#pragma once
// One bounded shared-memory job, pipes for ownership transfer, exec for
// isolation. This header deliberately has no camera, OpenCV, RKNN or motor
// dependency.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>
#ifdef __linux__
#include <sys/prctl.h>
#endif
extern char **environ;

template <class Job, class Reply> class WorkerChannel {
public:
  struct Memory {
    Job job;
    Reply reply;
    int status;
    char error[256];
  };
  static_assert(std::is_trivially_copyable<Job>::value &&
                    std::is_trivially_copyable<Reply>::value,
                "IPC values must not own pointers");
  Memory *memory = nullptr;
  pid_t pid = -1;
  int file = -1, send_fd = -1, receive_fd = -1;
  bool ready = false, pending = false;
  std::string error;

  WorkerChannel() {
    char path[] = "/tmp/bucket-worker-XXXXXX";
    file = mkstemp(path);
    if (file < 0)
      throw std::runtime_error("worker shared file");
    unlink(path);
    if (ftruncate(file, sizeof(Memory))) {
      close(file);
      throw std::runtime_error("worker shared size");
    }
    void *p = mmap(nullptr, sizeof(Memory), PROT_READ | PROT_WRITE, MAP_SHARED,
                   file, 0);
    if (p == MAP_FAILED) {
      close(file);
      throw std::runtime_error("worker mmap");
    }
    memory = static_cast<Memory *>(p);
  }
  void start(const std::string &binary, const std::string &mode,
             const std::vector<std::string> &extra) {
    stop();
    ready = pending = false;
    error.clear();
    memory->status = 0;
    int commands[2], replies[2];
    if (pipe(commands))
      throw std::runtime_error("worker pipe");
    if (pipe(replies)) {
      close(commands[0]);
      close(commands[1]);
      throw std::runtime_error("worker pipe");
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, file, file);
    posix_spawn_file_actions_adddup2(&actions, commands[0], commands[0]);
    posix_spawn_file_actions_adddup2(&actions, replies[1], replies[1]);
    posix_spawn_file_actions_addclose(&actions, commands[1]);
    posix_spawn_file_actions_addclose(&actions, replies[0]);
    std::vector<std::string> args{binary, mode, std::to_string(file),
                                  std::to_string(commands[0]),
                                  std::to_string(replies[1])};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char *> argv;
    for (auto &arg : args)
      argv.push_back(arg.data());
    argv.push_back(nullptr);
    int rc = posix_spawn(&pid, binary.c_str(), &actions, nullptr, argv.data(),
                         environ);
    posix_spawn_file_actions_destroy(&actions);
    close(commands[0]);
    close(replies[1]);
    if (rc) {
      close(commands[1]);
      close(replies[0]);
      pid = -1;
      throw std::runtime_error("worker exec");
    }
    send_fd = commands[1];
    receive_fd = replies[0];
  }
  bool poll_result(int timeout_ms = 0) {
    if (pid < 0)
      return false;
    pollfd fd{receive_fd, POLLIN, 0};
    int rc;
    do {
      rc = poll(&fd, 1, timeout_ms);
    } while (rc < 0 && errno == EINTR);
    if (rc == 0)
      return false;
    char marker;
    if (rc < 0 || !(fd.revents & POLLIN) || read(receive_fd, &marker, 1) != 1) {
      error = "worker exited";
      ready = false;
      return false;
    }
    if (memory->status < 0) {
      error = memory->error;
      ready = false;
      pending = false;
      return false;
    }
    ready = true;
    pending = false;
    return memory->status == 2;
  }
  void submit(const Job &job) {
    if (!ready || pending)
      throw std::runtime_error("worker not ready");
    memory->job = job;
    char command = 'R';
    if (write(send_fd, &command, 1) != 1)
      throw std::runtime_error("worker write");
    pending = true;
  }
  void stop() {
    if (pid > 0) {
      kill(pid, SIGTERM);
      int status;
      bool reaped = false;
      for (int i = 0; i < 20; i++) {
        pid_t done = waitpid(pid, &status, WNOHANG);
        if (done == pid || (done < 0 && errno == ECHILD)) {
          reaped = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      if (!reaped) {
        kill(pid, SIGKILL);
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
      }
      pid = -1;
    }
    if (send_fd >= 0) {
      close(send_fd);
      send_fd = -1;
    }
    if (receive_fd >= 0) {
      close(receive_fd);
      receive_fd = -1;
    }
    ready = pending = false;
  }
  ~WorkerChannel() {
    stop();
    if (memory)
      munmap(memory, sizeof(Memory));
    if (file >= 0)
      close(file);
  }
  WorkerChannel(const WorkerChannel &) = delete;
  WorkerChannel &operator=(const WorkerChannel &) = delete;

  template <class Handler>
  static int serve(int file, int input, int output, Handler &handler) {
#ifdef __linux__
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    if (getppid() == 1)
      return 3;
#endif
    // exec workers must not retain the parent's camera, sockets or other pipes.
    DIR *fds = opendir("/proc/self/fd");
    if (!fds)
      fds = opendir("/dev/fd");
    if (fds) {
      int listing = dirfd(fds);
      while (auto entry = readdir(fds)) {
        char *end = nullptr;
        long value = std::strtol(entry->d_name, &end, 10);
        if (end && *end == '\0' && value > 2 && value != file &&
            value != input && value != output && value != listing)
          close(int(value));
      }
      closedir(fds);
    }
    auto p = mmap(nullptr, sizeof(Memory), PROT_READ | PROT_WRITE, MAP_SHARED,
                  file, 0);
    if (p == MAP_FAILED)
      return 3;
    auto m = static_cast<Memory *>(p);
    try {
      handler.initialize();
      m->status = 1;
      char marker = 'R';
      if (write(output, &marker, 1) != 1)
        return 3;
      while (read(input, &marker, 1) == 1) {
        handler.run(m->job, m->reply);
        m->status = 2;
        if (write(output, &marker, 1) != 1)
          break;
      }
    } catch (const std::exception &e) {
      m->status = -1;
      std::snprintf(m->error, sizeof(m->error), "%s", e.what());
      char marker = 'E';
      if (write(output, &marker, 1) != 1)
        return 3;
    }
    munmap(m, sizeof(Memory));
    return 0;
  }
};
