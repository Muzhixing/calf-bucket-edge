#pragma once
#include "worker_channel.hpp"

struct DetectorJob {
  unsigned char image[720 * 1280 * 3];
};
struct DetectorReply {
  size_t used;
  unsigned char bytes[1024 * 1024];
};
using DetectorChannel = WorkerChannel<DetectorJob, DetectorReply>;

struct DetectorCodec {
  unsigned char *bytes;
  size_t &offset;
  size_t capacity;
  template <class T> void put(const T &value) {
    check(offset + sizeof(T) <= capacity, "detector reply capacity");
    std::memcpy(bytes + offset, &value, sizeof(T));
    offset += sizeof(T);
  }
  template <class T> T get() {
    check(offset + sizeof(T) <= capacity, "detector reply truncated");
    T value;
    std::memcpy(&value, bytes + offset, sizeof(T));
    offset += sizeof(T);
    return value;
  }
  void boxes(const std::vector<Box> &items) {
    put(items.size());
    for (auto &box : items)
      put(box);
  }
  std::vector<Box> boxes() {
    auto size = get<size_t>();
    check(size <= 8400, "detector box count");
    std::vector<Box> result;
    for (size_t i = 0; i < size; i++)
      result.push_back(get<Box>());
    return result;
  }
};

int detector_worker(int argc, char **argv) {
  check(argc == 8, "detector worker arguments");
  struct Handler {
    std::string path, cores;
    std::unique_ptr<Model> model;
    void initialize() {
      check(sha256(path) == "4a59b48b634c25ca79d843cc746d8e062ca1161ec5ef7c0c0c"
                            "96533c78d3fce7",
            "frozen model hash mismatch");
      model = std::make_unique<Model>(path);
      if (cores == "0_1_2")
        check(!rknn_set_core_mask(model->ctx, RKNN_NPU_CORE_0_1_2),
              "NPU cores");
      else if (cores != "auto") {
        check(cores == "0" || cores == "1" || cores == "2", "NPU core");
        check(!rknn_set_core_mask(model->ctx, static_cast<rknn_core_mask>(
                                                  1 << std::stoi(cores))),
              "NPU core");
      }
    }
    void run(const DetectorJob &job, DetectorReply &reply) {
      cv::Mat image(720, 1280, CV_8UC3, const_cast<unsigned char *>(job.image));
      Timings timings;
      auto boxes = model->detect_with_recovery(image, timings);
      reply.used = 0;
      DetectorCodec codec{reply.bytes, reply.used, sizeof(reply.bytes)};
      codec.boxes(boxes);
      codec.boxes(model->tracking_candidates);
      codec.put(timings.size());
      for (auto &entry : timings) {
        codec.put(entry.first.size());
        for (auto c : entry.first)
          codec.put(c);
        codec.put(entry.second);
      }
    }
  } handler{argv[6], argv[7], nullptr};
  return DetectorChannel::serve(std::stoi(argv[2]), std::stoi(argv[3]),
                                std::stoi(argv[4]), handler);
}

struct ProcessDetector {
  DetectorChannel channel;
  std::string binary, path, cores, state = "starting", code = "MODEL_STARTING";
  double started = 0, submitted = 0, retry_at = 0;
  int failures = 0;
  std::vector<Box> tracking_candidates;
  ProcessDetector(std::string binary, std::string path, std::string cores)
      : binary(std::move(binary)), path(std::move(path)),
        cores(std::move(cores)) {}
  void fail(const std::string &reason) {
    channel.stop();
    state = "unavailable";
    code = reason;
    retry_at = now() + std::min(8., .5 * std::pow(2., std::min(4, failures++)));
  }
  std::vector<Box> detect_with_recovery(const cv::Mat &left, Timings &timing) {
    tracking_candidates.clear();
    try {
      if (channel.pid < 0 && now() >= retry_at) {
        channel.start(binary, "--detector-worker", {"ipc-v1", path, cores});
        started = now();
        state = "starting";
        code = "MODEL_STARTING";
      }
      if (channel.pid < 0)
        return {};
      // A result for a previous frame is discarded, never attached to this
      // frame.
      channel.poll_result();
      if (!channel.error.empty()) {
        fail("MODEL_FAILED");
        return {};
      }
      if (!channel.ready) {
        if (now() - started > 8)
          fail("MODEL_START_TIMEOUT");
        return {};
      }
      if (channel.pending) {
        state = "degraded";
        code = "INFERENCE_PENDING";
        if (now() - submitted > 1)
          fail("INFERENCE_TIMEOUT");
        return {};
      }
      cv::Mat shared(720, 1280, CV_8UC3, channel.memory->job.image);
      left.copyTo(shared);
      char command = 'R';
      check(write(channel.send_fd, &command, 1) == 1, "detector submit");
      channel.pending = true;
      submitted = now();
      if (!channel.poll_result(80)) {
        state = "degraded";
        code = "INFERENCE_PENDING";
        return {};
      }
      check(channel.memory->reply.used <= sizeof(channel.memory->reply.bytes),
            "detector reply length");
      size_t offset = 0;
      auto &reply = channel.memory->reply;
      DetectorCodec codec{reply.bytes, offset, reply.used};
      auto boxes = codec.boxes();
      tracking_candidates = codec.boxes();
      size_t count = codec.get<size_t>();
      check(count < 100, "timing count");
      for (size_t i = 0; i < count; i++) {
        size_t length = codec.get<size_t>();
        check(length < 128, "timing key");
        std::string key;
        for (size_t j = 0; j < length; j++)
          key += codec.get<char>();
        timing[key] = codec.get<double>();
      }
      state = "ready";
      code = "OK";
      failures = 0;
      return boxes;
    } catch (const std::exception &e) {
      std::cerr << "detector: " << e.what() << '\n';
      fail("MODEL_FAILED");
      return {};
    }
  }
};
