#pragma once
// Frozen V2 detector/stereo core derived from benchmark commit c22d909a.
// No serial, controller or actuator code.
#include "rknn_api.h"
#include "score_order.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <optional>
#include <poll.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
using Clock = std::chrono::steady_clock;
double now() {
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
using Timings = std::map<std::string, double>;
struct Timer {
  double start = now();
  std::string name;
  Timings &t;
  Timer(Timings &x, std::string s) : name(s), t(x) {}
  ~Timer() { t[name] += (now() - start) * 1000; }
};
void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void affinity(std::initializer_list<int> cores) {
  cpu_set_t s;
  CPU_ZERO(&s);
  for (int c : cores)
    CPU_SET(c, &s);
  check(!sched_setaffinity(0, sizeof(s), &s), "affinity");
}
struct Box {
  int l, t, r, b, anchor;
  double score;
};
void restore_horizontal_boxes(std::vector<Box> &boxes, int width = 1280) {
  for (auto &b : boxes) {
    int left = b.l;
    b.l = std::clamp(width - b.r, 0, width - 1);
    b.r = std::clamp(width - left, 0, width - 1);
  }
}
std::string boxes_json(const std::vector<Box> &b) {
  std::ostringstream s;
  s << std::setprecision(12) << '[';
  for (size_t i = 0; i < b.size(); ++i) {
    if (i)
      s << ',';
    auto &v = b[i];
    s << "{\"left\":" << v.l << ",\"top\":" << v.t << ",\"right\":" << v.r
      << ",\"bottom\":" << v.b << ",\"score\":" << v.score
      << ",\"anchor\":" << v.anchor << '}';
  }
  s << ']';
  return s.str();
}
std::string timings_json(const Timings &t) {
  std::ostringstream s;
  s << std::setprecision(12) << '{';
  bool first = true;
  for (auto &v : t) {
    if (!first)
      s << ',';
    first = false;
    s << '"' << v.first << "\":" << v.second;
  }
  s << '}';
  return s.str();
}
std::string number(std::optional<double> v) {
  if (!v)
    return "null";
  std::ostringstream s;
  s << std::setprecision(12) << *v;
  return s.str();
}
bool same(const std::optional<Box> &a, const std::optional<Box> &b) {
  if (!a || !b)
    return false;
  double i = std::max(0, std::min(a->r, b->r) - std::max(a->l, b->l)) *
             std::max(0, std::min(a->b, b->b) - std::max(a->t, b->t));
  double u = (a->r - a->l) * (a->b - a->t) + (b->r - b->l) * (b->b - b->t) - i;
  return u > 0 && i / u >= .5;
}

struct Model {
  // Low-score boxes are association evidence only, never new detections/range targets.
  std::vector<Box> tracking_candidates;
  unsigned view_frame = 0;
  rknn_context ctx = 0;
  std::array<rknn_output, 9> out{};
  bool held = false;
  Model(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    check(bool(f), "model open");
    std::vector<char> data((std::istreambuf_iterator<char>(f)), {});
    check(!rknn_init(&ctx, data.data(), data.size(), 0, nullptr), "rknn_init");
    check(!rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO), "core mask");
    rknn_input_output_num io{};
    check(!rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) &&
              io.n_input == 1 && io.n_output == 9,
          "tensor count");
    rknn_sdk_version v{};
    check(!rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &v, sizeof(v)),
          "version query");
    std::cerr << "RKNN_API=" << v.api_version << " DRIVER=" << v.drv_version
              << " OPENCV=" << CV_VERSION << '\n';
    for (int i = 0; i < 9; i++) {
      rknn_tensor_attr a{};
      a.index = i;
      check(!rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)),
            "output attr");
      int width = 80 / (1 << (i / 3)), ch = i % 3 == 0 ? 64 : 1;
      check(a.n_elems == unsigned(width * width * ch) &&
                a.type == RKNN_TENSOR_INT8,
            "frozen output shape/type");
      std::cerr << "OUTPUT " << i << ' ' << a.name << " zp=" << a.zp
                << " scale=" << a.scale << '\n';
    }
  }
  ~Model() {
    if (held)
      rknn_outputs_release(ctx, 9, out.data());
    if (ctx)
      rknn_destroy(ctx);
  }
  std::vector<Box> detect(const cv::Mat &left, Timings &t) {
    tracking_candidates.clear();
    cv::Mat input, resized;
    {
      Timer tm(t, "preprocess_ms");
      cv::resize(left, resized, {640, 360}, 0, 0, cv::INTER_LINEAR);
      cv::copyMakeBorder(resized, input, 140, 140, 0, 0, cv::BORDER_CONSTANT,
                         cv::Scalar(0, 0, 0));
      // B1 C0 is trained and validated with RGB input.
      cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    }
    rknn_input in{};
    in.buf = input.data;
    in.size = 640 * 640 * 3;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    {
      Timer tm(t, "npu_inputs_set_ms");
      check(!rknn_inputs_set(ctx, 1, &in), "inputs_set");
    }
    {
      Timer tm(t, "npu_run_ms");
      check(!rknn_run(ctx, nullptr), "run");
    }
    out = {};
    for (int i = 0; i < 9; i++) {
      out[i].index = i;
      out[i].want_float = 1;
    }
    {
      Timer tm(t, "npu_outputs_get_ms");
      check(!rknn_outputs_get(ctx, 9, out.data(), nullptr), "get");
      held = true;
    }
    struct Candidate {
      std::array<double, 4> b;
      float score;
      int anchor;
    };
    std::vector<Candidate> c;
    std::vector<Box> result;
    {
      Timer tm(t, "postprocess_ms");
      int offset = 0;
      for (int branch = 0; branch < 3; branch++) {
        int w = 80 / (1 << branch), n = w * w, stride = 640 / w;
        auto reg = (float *)out[branch * 3].buf,
             conf = (float *)out[branch * 3 + 1].buf;
        for (int i = 0; i < n; i++) {
          if (!(conf[i] >= .25f))
            continue;
          std::array<double, 4> d{};
          for (int side = 0; side < 4; side++) {
            float m = -INFINITY, den = 0;
            std::array<float, 16> e{};
            for (int k = 0; k < 16; k++)
              m = std::max(m, reg[(side * 16 + k) * n + i]);
            for (int k = 0; k < 16; k++) {
              e[k] = std::exp(reg[(side * 16 + k) * n + i] - m);
              den += e[k];
            }
            for (int k = 0; k < 16; k++)
              d[side] += double(e[k] / den) * k;
          }
          double x = i % w + .5, y = i / w + .5;
          c.push_back({{(x - d[0]) * stride, (y - d[1]) * stride,
                        (x + d[2]) * stride, (y + d[3]) * stride},
                       conf[i],
                       offset + i});
        }
        offset += n;
      }
      std::vector<float> scores;
      for (const auto &v : c)
        scores.push_back(v.score);
      auto order = score_order(scores);
      auto unsorted = std::move(c);
      c.clear();
      c.reserve(order.size());
      for (int i : order)
        c.push_back(unsorted[i]);
      std::vector<Candidate> keep;
      for (auto &v : c) {
        bool suppressed = false;
        for (auto &k : keep) {
          double iw = std::max(0., std::min(k.b[2], v.b[2]) -
                                       std::max(k.b[0], v.b[0]) + 1e-5),
                 ih = std::max(0., std::min(k.b[3], v.b[3]) -
                                       std::max(k.b[1], v.b[1]) + 1e-5);
          double intersection = iw * ih;
          double u = (k.b[2] - k.b[0]) * (k.b[3] - k.b[1]) +
                     (v.b[2] - v.b[0]) * (v.b[3] - v.b[1]) - intersection;
          if (intersection / u > .45) {
            suppressed = true;
            break;
          }
        }
        if (!suppressed)
          keep.push_back(v);
      }
      for (auto &v : keep) {
        auto clamp = [](int x, int hi) { return std::clamp(x, 0, hi); };
        Box decoded{
             clamp(int(v.b[0]) * 2, 1279), clamp((int(v.b[1]) - 140) * 2, 719),
             clamp(int(v.b[2]) * 2, 1279), clamp((int(v.b[3]) - 140) * 2, 719),
             v.anchor, v.score};
        if (v.score >= .536f) result.push_back(decoded);
        else if (tracking_candidates.size() < 32) tracking_candidates.push_back(decoded);
      }
    }
    {
      Timer tm(t, "npu_release_ms");
      check(!rknn_outputs_release(ctx, 9, out.data()), "release");
      held = false;
    }
    return result;
  }
  std::vector<Box> detect_with_recovery(const cv::Mat &left, Timings &t) {
    auto primary = detect(left, t);
    t["recovery_view_attempted"] = 0;
    t["recovery_view_used"] = 0;
    // Bounded test-time augmentation: retain every normal inference and try
    // one mirrored view on every third frame when it misses. No stale boxes are reused.
    if (++view_frame % 3 != 0 || !primary.empty()) return primary;
    auto primary_weak = tracking_candidates;
    cv::Mat mirrored;
    cv::flip(left, mirrored, 1);
    Timings extra;
    double start = now();
    auto recovered = detect(mirrored, extra);
    t["recovery_view_ms"] = (now() - start) * 1000;
    t["recovery_view_attempted"] = 1;
    // Weak alternate boxes do not create a new target. Keep the normal-view
    // weak associations and the existing multi-frame lock/occlusion checks.
    tracking_candidates = std::move(primary_weak);
    restore_horizontal_boxes(recovered, left.cols);
    if (!recovered.empty()) t["recovery_view_used"] = 1;
    return recovered;
  }
};

float percentile(std::vector<float> v, double q) {
  check(!v.empty(), "empty quantile");
  double x = (v.size() - 1) * q;
  size_t a = std::floor(x), b = std::ceil(x);
  std::nth_element(v.begin(), v.begin() + b, v.end());
  float hi = v[b],
        lo = a == b ? hi : *std::max_element(v.begin(), v.begin() + b);
  return float(double(lo) + (double(hi) - lo) * (x - a));
}
struct Geometry {
  cv::Mat lm1, lm2, rm1, rm2;
  double factor=0, fx=1, fy=1;
  Geometry()=default;
  Geometry(const std::string &path) {
    cv::FileStorage f(path, cv::FileStorage::READ);
    check(f.isOpened(), "calibration open");
    cv::Mat k1, k2, d1, d2, r1, r2, p1, p2, t;
    f["K1"] >> k1;
    f["K2"] >> k2;
    f["D1"] >> d1;
    f["D2"] >> d2;
    f["R1"] >> r1;
    f["R2"] >> r2;
    f["P1"] >> p1;
    f["P2"] >> p2;
    f["T"] >> t;
    cv::initUndistortRectifyMap(k1, d1, r1, p1, {1280, 720}, CV_16SC2, lm1,
                                lm2);
    cv::initUndistortRectifyMap(k2, d2, r2, p2, {1280, 720}, CV_16SC2, rm1,
                                rm2);
    fx = p1.at<double>(0, 0);
    fy = p1.at<double>(1, 1);
    factor = fx * cv::norm(t) / 1000;
  }
};
struct Range {
  std::optional<double> z;
  Timings timing;
  double start = 0, end = 0;
  cv::Rect roi;
  cv::Mat thumb;
  int reason = 0; // 0 valid, 1 insufficient disparity, 2 quality, 3 range limit
};
struct Stereo {
  Geometry &g;
  double scale;
  cv::Ptr<cv::StereoSGBM> matcher;
  Stereo(Geometry &geo, double s) : g(geo), scale(s) {
    matcher =
        cv::StereoSGBM::create(0, int(160 * s), 7, 8 * 3 * 49, 32 * 3 * 49, 1,
                               63, 12, 120, 2, cv::StereoSGBM::MODE_SGBM_3WAY);
  }
  std::optional<double> distance(const cv::Mat &disp, const Box &b,
                                 Range &audit) {
    for (int branch = 0; branch < 2; branch++) {
      double expand = branch ? .35 : .25, ratio = branch ? .18 : .35;
      int dx = int(std::max(1, b.r - b.l) * expand),
          dy = int(std::max(1, b.b - b.t) * expand);
      int x1 = std::clamp(b.l - dx, 0, 1279),
          x2 = std::clamp(b.r + dx, 0, 1279), y1 = std::clamp(b.t - dy, 0, 719),
          y2 = std::clamp(b.b + dy, 0, 719);
      if (branch) {
        int w = x2 - x1, h = y2 - y1;
        x2 = x1 + int(w * .8);
        y2 = y1 + int(h * .8);
        x1 += int(w * .2);
        y1 += int(h * .2);
      }
      audit.reason = 1;
      audit.roi = cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
      if (x2 <= x1 || y2 <= y1)
        continue;
      double count = (x2 - x1) * (y2 - y1);
      std::vector<float> v;
      v.reserve(size_t(count));
      for (int y = y1; y < y2; y++) {
        auto row = disp.ptr<float>(y);
        for (int x = x1; x < x2; x++)
          if (row[x] > 0 && std::isfinite(row[x]))
            v.push_back(row[x]);
      }
      if (v.size() < count * ratio)
        continue;
      float lo = percentile(v, .1), hi = percentile(v, .9);
      v.erase(std::remove_if(v.begin(), v.end(),
                             [=](float x) { return x < lo || x > hi; }),
              v.end());
      if (v.size() < count * ratio * .6)
        continue;
      float med = percentile(v, .5);
      std::vector<float> abs;
      abs.reserve(v.size());
      for (float x : v)
        abs.push_back(std::abs(x - med));
      double sigma = 1.4826 * (double(percentile(abs, .5)) + 1e-6);
      v.erase(std::remove_if(
                  v.begin(), v.end(),
                  [=](float x) { return std::abs(x - med) > 3 * sigma; }),
              v.end());
      if (v.size() < count * ratio * .6)
        continue;
      double median = percentile(v, .5),
             mean = std::accumulate(v.begin(), v.end(), 0.) / v.size(), var = 0;
      for (float x : v)
        var += (x - mean) * (x - mean);
      double sd = std::sqrt(var / v.size());
      audit.reason = 2;
      if (median < (branch ? .8 : 1.2) ||
          sd / std::max(1e-6, median) > (branch ? .75 : .55))
        continue;
      double z = g.factor / median;
      audit.reason = 3;
      if (z >= .05 && z <= 1.2) {
        audit.reason = 0;
        return z;
      }
    }
    return {};
  }
  Range measure(cv::Mat left, cv::Mat raw_right, Box box) {
    Range r;
    r.start = now();
    cv::Mat right, lg, rg, raw, disp;
    {
      Timer t(r.timing, "right_remap_ms");
      cv::remap(raw_right, right, g.rm1, g.rm2, cv::INTER_LINEAR);
    }
    {
      Timer t(r.timing, "grayscale_ms");
      cv::cvtColor(left, lg, cv::COLOR_BGR2GRAY);
      cv::cvtColor(right, rg, cv::COLOR_BGR2GRAY);
    }
    {
      Timer t(r.timing, "disparity_total_ms");
      if (scale < 1) {
        cv::resize(lg, lg, {640, 360}, 0, 0, cv::INTER_AREA);
        cv::resize(rg, rg, {640, 360}, 0, 0, cv::INTER_AREA);
      }
      {
        Timer s(r.timing, "sgbm_ms");
        matcher->compute(lg, rg, raw);
      }
      raw.convertTo(disp, CV_32F, 1. / 16);
      if (scale < 1) {
        disp /= scale;
        cv::resize(disp, disp, {1280, 720}, 0, 0, cv::INTER_LINEAR);
      }
      cv::threshold(disp, disp, 0, 0, cv::THRESH_TOZERO);
      cv::medianBlur(disp, disp, 5);
    }
    {
      Timer t(r.timing, "roi_ms");
      r.z = distance(disp, box, r);
    }
    // Keep the existing duplicate audit burden in the language comparison.
    {
      Timer t(r.timing, "audit_duplicate_roi_ms");
      Range detail;
      auto audit = distance(disp, box, detail);
      check(audit == r.z, "range audit");
    }
    cv::Mat small;
    cv::resize(disp, small, {336, 189}, 0, 0, cv::INTER_NEAREST);
    small.convertTo(r.thumb, CV_8U, 255. / 160.);
    r.end = now();
    r.timing["worker_total_ms"] = (r.end - r.start) * 1000;
    return r;
  }
};

// Same isolation as the Python preview: one shared pair and a persistent stereo
// process, with one in-flight job. Fork before creating the NPU
// context/threads.
struct ProcessStereo {
  static constexpr size_t image_bytes = 720 * 1280 * 3;
  static constexpr size_t thumb_bytes = 336 * 189;
  const std::array<std::string, 7> keys = {
      "right_remap_ms", "grayscale_ms", "disparity_total_ms",
      "sgbm_ms",        "roi_ms",       "audit_duplicate_roi_ms",
      "worker_total_ms"};
  struct Wire {
    int valid, roi[4], reason;
    double z, start, end;
    double timings[7];
  };
  int send_fd = -1, receive_fd = -1;
  pid_t pid = -1;
  void *memory = nullptr;
  bool pending = false;
  double submitted = 0;
  static void transfer(int fd, void *data, size_t bytes, bool write_data) {
    auto p = static_cast<char *>(data);
    while (bytes) {
      ssize_t n = write_data ? ::write(fd, p, bytes) : ::read(fd, p, bytes);
      if (n < 0 && errno == EINTR)
        continue;
      check(n > 0, "stereo IPC closed");
      p += n;
      bytes -= n;
    }
  }
  ProcessStereo(Geometry &g, double scale) {
    memory = mmap(nullptr, 2 * image_bytes + thumb_bytes,
                  PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    check(memory != MAP_FAILED, "shared mmap");
    int commands[2], results[2];
    check(!pipe(commands) && !pipe(results), "pipes");
    pid = fork();
    check(pid >= 0, "fork");
    if (pid == 0) {
      signal(SIGINT, SIG_IGN);
      signal(SIGTERM, SIG_IGN);
      signal(SIGHUP, SIG_IGN);
      close(commands[1]);
      close(results[0]);
      try {
        affinity({4, 6, 7});
        setpriority(PRIO_PROCESS, 0, 5);
        cv::setNumThreads(3);
        Stereo s(g, scale);
        cv::Mat left(720, 1280, CV_8UC3, memory),
            right(720, 1280, CV_8UC3,
                  static_cast<char *>(memory) + image_bytes);
        for (;;) {
          Box box;
          transfer(commands[0], &box, sizeof(box), false);
          if (box.l < 0)
            break;
          auto r = s.measure(left, right, box);
          Wire wire{};
          wire.valid = bool(r.z);
          wire.z = r.z.value_or(0);
          wire.reason = r.reason;
          wire.roi[0] = r.roi.x;
          wire.roi[1] = r.roi.y;
          wire.roi[2] = r.roi.width;
          wire.roi[3] = r.roi.height;
          std::memcpy(static_cast<char *>(memory) + 2 * image_bytes,
                      r.thumb.data, thumb_bytes);
          wire.start = r.start;
          wire.end = r.end;
          for (size_t i = 0; i < keys.size(); i++)
            wire.timings[i] = r.timing[keys[i]];
          transfer(results[1], &wire, sizeof(wire), true);
        }
        _exit(0);
      } catch (const std::exception &e) {
        std::cerr << "stereo child: " << e.what() << '\n';
        _exit(3);
      }
    }
    close(commands[0]);
    close(results[1]);
    send_fd = commands[1];
    receive_fd = results[0];
  }
  void submit(const cv::Mat &left, const cv::Mat &right, Box box, Timings &t) {
    check(!pending, "shared pair overwrite");
    {
      Timer timer(t, "shared_copy_ms");
      cv::Mat l(720, 1280, CV_8UC3, memory),
          r(720, 1280, CV_8UC3, static_cast<char *>(memory) + image_bytes);
      left.copyTo(l);
      right.copyTo(r);
    }
    submitted = now();
    transfer(send_fd, &box, sizeof(box), true);
    pending = true;
  }
  bool ready() {
    pollfd fd{receive_fd, POLLIN, 0};
    return pending && poll(&fd, 1, 0) > 0;
  }
  Range get() {
    check(pending, "no pending stereo result");
    Wire wire;
    transfer(receive_fd, &wire, sizeof(wire), false);
    pending = false;
    Range r;
    if (wire.valid)
      r.z = wire.z;
    r.reason = wire.reason;
    r.roi = cv::Rect(wire.roi[0], wire.roi[1], wire.roi[2], wire.roi[3]);
    r.thumb =
        cv::Mat(189, 336, CV_8U, static_cast<char *>(memory) + 2 * image_bytes)
            .clone();
    r.start = wire.start;
    r.end = wire.end;
    for (size_t i = 0; i < keys.size(); i++)
      r.timing[keys[i]] = wire.timings[i];
    r.timing["submit_to_worker_ms"] = (r.start - submitted) * 1000;
    return r;
  }
  ~ProcessStereo() {
    if (pid > 0) {
      Box stop{};
      stop.l = -1;
      const auto sent = ::write(send_fd, &stop, sizeof(stop));
      if (sent != sizeof(stop))
        std::cerr << "stereo worker already closed during shutdown\n";
      close(send_fd);
      close(receive_fd);
      int status;
      auto deadline=now()+.2;
      while(waitpid(pid,&status,WNOHANG)==0 && now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if(waitpid(pid,&status,WNOHANG)==0) { kill(pid,SIGKILL);while(waitpid(pid,&status,0)<0 && errno==EINTR) {} }
    }
    if (memory && memory != MAP_FAILED)
      munmap(memory, 2 * image_bytes + thumb_bytes);
  }
};

// A persistent worker matches ThreadPoolExecutor's lifetime; creating a new
// A55 thread every frame also resets the scheduler's utilization history.
struct RemapTail {
  Geometry &g;
  std::mutex mutex;
  std::condition_variable condition;
  bool pending = false, stop = false;
  cv::Mat raw, dst;
  std::exception_ptr error;
  std::thread thread;
  explicit RemapTail(Geometry &geometry) : g(geometry) {
    thread = std::thread([this]() {
      affinity({2});
      std::unique_lock<std::mutex> lock(mutex);
      for (;;) {
        condition.wait(lock, [&]() { return stop || pending; });
        if (stop)
          return;
        try {
          cv::remap(raw, dst, g.lm1.rowRange(480, 720),
                    g.lm2.rowRange(480, 720), cv::INTER_LINEAR);
        } catch (...) {
          error = std::current_exception();
        }
        pending = false;
        condition.notify_all();
      }
    });
  }
  void submit(cv::Mat input, cv::Mat output) {
    std::lock_guard<std::mutex> lock(mutex);
    raw = input;
    dst = output;
    pending = true;
    condition.notify_one();
  }
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&]() { return !pending; });
    if (error)
      std::rethrow_exception(error);
  }
  ~RemapTail() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
      condition.notify_one();
    }
    thread.join();
  }
};

struct RectifiedReplay {
  struct Item {
    long seq = 0;
    double arrival = 0, remap_ms = 0;
    bool remap_opencl = false;
    cv::Mat raw, left;
  };
  std::atomic<bool> stop{false};
  std::mutex mutex;
  std::condition_variable condition;
  Item latest;
  std::exception_ptr error;
  std::thread thread;
  RectifiedReplay(const std::vector<cv::Mat> &images, Geometry &g,
                  double start) {
    thread = std::thread([&, start]() {
      try {
        affinity({5});
        RemapTail tail(g);
        for (int i = 0; !stop; i++) {
          double deadline = start + i / 30.;
          std::this_thread::sleep_until(
              Clock::time_point(std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double>(deadline))));
          i = std::max(i, int((now() - start) * 30));
          Item item;
          item.seq = i + 1;
          item.arrival = start + i / 30.;
          item.raw = images[i % images.size()];
          cv::Mat raw_left = item.raw(cv::Rect(0, 0, 1280, 720));
          item.left.create(720, 1280, CV_8UC3);
          double t = now();
          tail.submit(raw_left, item.left.rowRange(480, 720));
          cv::Mat dst = item.left.rowRange(0, 480);
          cv::remap(raw_left, dst, g.lm1.rowRange(0, 480),
                    g.lm2.rowRange(0, 480), cv::INTER_LINEAR);
          tail.wait();
          item.remap_ms = (now() - t) * 1000;
          {
            std::lock_guard<std::mutex> guard(mutex);
            latest = std::move(item);
          }
          condition.notify_one();
        }
      } catch (...) {
        std::lock_guard<std::mutex> guard(mutex);
        error = std::current_exception();
        condition.notify_one();
      }
    });
  }
  Item after(long seq) {
    std::unique_lock<std::mutex> lock(mutex);
    check(condition.wait_for(lock, std::chrono::seconds(3),
                             [&]() { return error || latest.seq > seq; }),
          "replay rectifier timeout");
    if (error)
      std::rethrow_exception(error);
    return latest;
  }
  ~RectifiedReplay() {
    stop = true;
    if (thread.joinable())
      thread.join();
  }
};
