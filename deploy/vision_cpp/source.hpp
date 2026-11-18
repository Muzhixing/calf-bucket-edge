#pragma once
// Latest-only camera capture and rectification. Buffers are immutable after
// publication; a slow consumer drops frames instead of accumulating latency.
struct Source {
  using Item = RectifiedReplay::Item;
  std::atomic<bool> stop{false};
  std::mutex mutex;
  std::condition_variable condition;
  Item raw, latest;
  std::exception_ptr error;
  std::thread capture, rectifier;
  Source(Geometry &g, const std::string &device,
         const std::vector<cv::Mat> &images) {
    capture = std::thread([this, device, images]() {
      try {
        affinity({4});
        cv::VideoCapture camera;
        if (images.empty()) {
          check(camera.open(device, cv::CAP_V4L2), "camera open");
          camera.set(cv::CAP_PROP_FOURCC,
                     cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
          camera.set(cv::CAP_PROP_FRAME_WIDTH, 2560);
          camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
          camera.set(cv::CAP_PROP_FPS, 30);
          camera.set(cv::CAP_PROP_BUFFERSIZE, 2);
          std::cerr << "CAMERA " << device << " "
                    << camera.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                    << camera.get(cv::CAP_PROP_FRAME_HEIGHT) << " @"
                    << camera.get(cv::CAP_PROP_FPS) << '\n';
        }
        double start = now();
        long i = 0;
        while (!stop) {
          Item item;
          if (images.empty()) {
            check(camera.read(item.raw), "camera read failed");
            item.arrival = now();
          } else {
            std::this_thread::sleep_until(
                Clock::time_point(std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(start + i / 30.))));
            i = std::max(i, long((now() - start) * 30));
            item.raw = images[i % images.size()];
            item.arrival = start + i / 30.;
          }
          check(item.raw.rows == 720 && item.raw.cols == 2560,
                "camera must provide 2560x720 SBS");
          item.seq = ++i;
          {
            std::lock_guard<std::mutex> lock(mutex);
            raw = std::move(item);
          }
          condition.notify_all();
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex);
        error = std::current_exception();
        condition.notify_all();
      }
    });
    rectifier = std::thread([this, &g]() {
      try {
        affinity({5});
        RemapTail tail(g);
        long seq = 0;
        while (!stop) {
          Item item;
          {
            std::unique_lock<std::mutex> lock(mutex);
            if (!condition.wait_for(lock, std::chrono::seconds(3), [&] {
                  return stop || error || raw.seq > seq;
                }))
              throw std::runtime_error("camera frame timeout");
            if (stop)
              return;
            if (error)
              std::rethrow_exception(error);
            item = raw;
          }
          seq = item.seq;
          auto left = item.raw(cv::Rect(0, 0, 1280, 720));
          item.left.create(720, 1280, CV_8UC3);
          double t = now();
          tail.submit(left, item.left.rowRange(480, 720));
          cv::Mat upper = item.left.rowRange(0, 480);
          cv::remap(left, upper, g.lm1.rowRange(0, 480), g.lm2.rowRange(0, 480),
                    cv::INTER_LINEAR);
          tail.wait();
          item.remap_ms = (now() - t) * 1000;
          {
            std::lock_guard<std::mutex> lock(mutex);
            latest = std::move(item);
          }
          condition.notify_all();
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex);
        error = std::current_exception();
        condition.notify_all();
      }
    });
  }
  Item after(long seq) {
    std::unique_lock<std::mutex> lock(mutex);
    check(condition.wait_for(lock, std::chrono::seconds(3),
                             [&] { return error || latest.seq > seq; }),
          "rectifier timeout");
    if (error)
      std::rethrow_exception(error);
    return latest;
  }
  ~Source() {
    stop = true;
    condition.notify_all();
    if (rectifier.joinable())
      rectifier.join();
    if (capture.joinable())
      capture.join();
  }
};
