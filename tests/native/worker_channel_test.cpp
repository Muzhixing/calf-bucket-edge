#include "worker_channel.hpp"
#include <cassert>
#include <iostream>
struct Job {
  int value;
};
struct Reply {
  int value;
};
using Channel = WorkerChannel<Job, Reply>;
struct Handler {
  std::string mode;
  void initialize() {
    if (mode == "load-failure")
      throw std::runtime_error("model unavailable");
  }
  void run(const Job &job, Reply &reply) {
    if (mode == "hang")
      std::this_thread::sleep_for(std::chrono::seconds(30));
    if (mode == "crash")
      raise(SIGSEGV);
    if (mode == "inference-failure")
      throw std::runtime_error("inference failed");
    reply.value = job.value * 3;
  }
};
int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN);
  if (argc > 1 && std::string(argv[1]) == "--worker") {
    Handler handler{argv[5]};
    return Channel::serve(std::stoi(argv[2]), std::stoi(argv[3]),
                          std::stoi(argv[4]), handler);
  }
  Channel channel;
  for (auto mode : {"load-failure", "normal", "inference-failure", "crash",
                    "hang", "normal"}) {
    channel.start(argv[0], "--worker", {mode});
    for (int i = 0; i < 200 && !channel.ready && channel.error.empty(); i++)
      channel.poll_result(10);
    if (std::string(mode) == "load-failure") {
      assert(!channel.ready);
      assert(!channel.error.empty());
      channel.stop();
      continue;
    }
    assert(channel.ready);
    channel.submit(Job{7});
    if (std::string(mode) == "hang") {
      assert(!channel.poll_result(20));
      auto begin = std::chrono::steady_clock::now();
      channel.stop();
      assert(std::chrono::steady_clock::now() - begin <
             std::chrono::seconds(1));
      continue;
    }
    bool result = false;
    for (int i = 0; i < 200 && !result && channel.error.empty(); i++)
      result = channel.poll_result(10);
    if (std::string(mode) == "normal")
      assert(result && channel.memory->reply.value == 21);
    else
      assert(!channel.error.empty());
    channel.stop();
  }
  std::cout << "worker load failure / crash / hang / restart / bounded "
               "shutdown passed\n";
}
