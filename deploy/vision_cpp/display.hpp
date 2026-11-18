#pragma once
#include <arpa/inet.h>
#include <fcntl.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <sys/socket.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

struct Packet {
  long seq = 0, source = 0;
  double published = 0, fps = 0, hz = 0, dx = 0, dy = 0;
  cv::Mat raw, left, thumb;
  cv::Rect roi;
  std::optional<Box> box;
  std::optional<double> z;
  std::string status, json;
  bool aligned = false;
};
void text_at(cv::Mat &im, const std::string &s, cv::Point p, double scale,
             cv::Scalar color, int thick = 1) {
  cv::putText(im, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(15, 24, 20),
              thick + 2, cv::LINE_AA);
  cv::putText(im, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, color, thick,
              cv::LINE_AA);
}
std::string fixed(double x, int n = 1) {
  std::ostringstream s;
  s << std::fixed << std::setprecision(n) << x;
  return s.str();
}
cv::Mat overlay(const Packet &p) {
  cv::Mat im = p.left.clone();
  auto white = cv::Scalar(240, 245, 242);
  auto color = p.aligned ? cv::Scalar(120, 230, 150) : cv::Scalar(70, 190, 255);
  cv::drawMarker(im, {640, 360}, white, cv::MARKER_CROSS, 28, 2, cv::LINE_AA);
  // The small central rectangle is the inclusive two-axis tolerance region.
  cv::rectangle(im, {589, 331}, {691, 389}, cv::Scalar(140, 155, 145), 1,
                cv::LINE_AA);
  if (p.box) {
    auto b = *p.box;
    cv::rectangle(im, {b.l, b.t}, {b.r, b.b}, color, 2, cv::LINE_AA);
    cv::Point center((b.l + b.r) / 2, (b.t + b.b) / 2);
    cv::line(im, {640, 360}, center, color, 1, cv::LINE_AA);
    cv::circle(im, center, 5, color, -1, cv::LINE_AA);
    text_at(im, "dx " + fixed(p.dx) + "  dy " + fixed(p.dy) + " px", {20, 40},
            .65, white, 2);
  }
  if (p.z && p.roi.area() > 0)
    cv::rectangle(im, p.roi, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
  return im;
}
struct Screen {
  bool hdmi;
  std::string out;
  cv::Mat base;
  std::vector<cv::Mat> tiles;
  GstElement *pipeline = nullptr, *src = nullptr;
  long rendered = 0;
  double start = now();
  Screen(bool enabled, const std::string &assets, const std::string &folder)
      : hdmi(enabled), out(folder) {
    if (!hdmi)
      return;
    base = cv::imread(assets + "/hdmi_base.png");
    check(base.rows == 720 && base.cols == 1280, "HDMI assets");
    for (int i = 0; i < 6; ++i) {
      tiles.push_back(
          cv::imread(assets + "/label" + std::to_string(i) + ".png"));
      check(!tiles.back().empty(), "HDMI label");
    }
    gst_init(nullptr, nullptr);
    int fd = open("/dev/dri/by-path/platform-display-subsystem-card",
                  O_RDWR | O_CLOEXEC);
    check(fd >= 0, "DRM open");
    auto *res = drmModeGetResources(fd);
    int connector = 0;
    bool supported_mode = false;
    if (res) {
      for (int i = 0; i < res->count_connectors; ++i) {
        auto *c = drmModeGetConnectorCurrent(fd, res->connectors[i]);
        if (c) {
          if (c->connector_type == DRM_MODE_CONNECTOR_HDMIA &&
              c->connector_type_id == 1 &&
              c->connection == DRM_MODE_CONNECTED) {
            connector = c->connector_id;
            for (int m = 0; m < c->count_modes; ++m)
              supported_mode |=
                  c->modes[m].hdisplay == 1280 && c->modes[m].vdisplay == 720;
          }
          drmModeFreeConnector(c);
        }
      }
      drmModeFreeResources(res);
    }
    close(fd);
    check(connector > 0 && supported_mode, "HDMI-A-1 must support 1280x720");
    std::string description =
        "appsrc name=src is-live=true format=time do-timestamp=true "
        "block=false max-buffers=1 leaky-type=downstream "
        "caps=video/x-raw,format=BGRx,width=1280,height=720,framerate=30/1 ! "
        "kmssink name=screen driver-name=rockchip connector-id=" +
        std::to_string(connector) +
        " force-modesetting=true restore-crtc=true enable-last-sample=false "
        "sync=false";
    GError *err = nullptr;
    pipeline = gst_parse_launch(description.c_str(), &err);
    if (err) {
      std::string msg = err->message;
      g_error_free(err);
      if (pipeline)
        gst_object_unref(pipeline);
      pipeline = nullptr;
      throw std::runtime_error(msg);
    }
    check(pipeline, "GStreamer pipeline");
    src = gst_bin_get_by_name(GST_BIN(pipeline), "src");
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) ==
        GST_STATE_CHANGE_FAILURE) {
      gst_object_unref(src);
      gst_object_unref(pipeline);
      src = pipeline = nullptr;
      throw std::runtime_error("HDMI start failed; stop display-manager first");
    }
  }
  cv::Mat draw(const Packet &p, const cv::Mat &annotated) {
    cv::Mat im = base.clone(), small;
    cv::resize(annotated, small, {896, 504});
    small.copyTo(im(cv::Rect(16, 112, 896, 504)));
    cv::resize(p.raw(cv::Rect(0, 0, 1280, 720)), small, {336, 189});
    small.copyTo(im(cv::Rect(928, 39, 336, 189)));
    cv::resize(p.raw(cv::Rect(1280, 0, 1280, 720)), small, {336, 189});
    small.copyTo(im(cv::Rect(928, 274, 336, 189)));
    if (p.z && !p.thumb.empty()) {
      cv::applyColorMap(p.thumb, small, cv::COLORMAP_TURBO);
      small.setTo(cv::Scalar(), p.thumb == 0);
      small.copyTo(im(cv::Rect(928, 507, 336, 189)));
    }
    tiles[p.box ? (p.aligned ? 2 : 1) : 0].copyTo(
        im(cv::Rect(248, 36, 175, 38)));
    tiles[p.status == "new"    ? 3
          : p.status == "held" ? 4
                               : 5]
        .copyTo(im(cv::Rect(524, 36, 175, 38)));
    auto white = cv::Scalar(235, 239, 229);
    text_at(im, p.z ? fixed(*p.z * 100, 1) + " cm" : "-- cm", {16, 64}, 1.22,
            white, 2);
    text_at(im, fixed(p.fps) + " FPS / " + fixed(p.hz) + " Hz", {710, 62}, .53,
            white);
    text_at(im,
            p.box ? "dx " + fixed(p.dx) + "  dy " + fixed(p.dy) + " px"
                  : "dx --  dy --",
            {532, 650}, .6, white);
    text_at(im,
            "frame " + std::to_string(p.seq) + " / depth " +
                (p.z ? std::to_string(p.source) : "--") + "  [C++]",
            {532, 687}, .5, white);
    return im;
  }
  void push(const cv::Mat &im) {
    cv::Mat bgra;
    cv::cvtColor(im, bgra, cv::COLOR_BGR2BGRA);
    size_t size = bgra.total() * bgra.elemSize();
    GstBuffer *buf = gst_buffer_new_allocate(nullptr, size, nullptr);
    check(buf, "GStreamer buffer");
    gst_buffer_fill(buf, 0, bgra.data, size);
    check(gst_app_src_push_buffer(GST_APP_SRC(src), buf) == GST_FLOW_OK,
          "HDMI buffer push failed");
    ++rendered;
    auto *bus = gst_element_get_bus(pipeline);
    auto *msg = gst_bus_pop_filtered(
        bus, GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    gst_object_unref(bus);
    if (msg) {
      GError *err = nullptr;
      gchar *debug = nullptr;
      std::string reason = "HDMI EOS";
      if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        gst_message_parse_error(msg, &err, &debug);
        reason = err->message;
        g_error_free(err);
        g_free(debug);
      }
      gst_message_unref(msg);
      throw std::runtime_error(reason);
    }
  }
  ~Screen() {
    if (pipeline) {
      auto *sink = gst_bin_get_by_name(GST_BIN(pipeline), "screen");
      GstStructure *stats = nullptr;
      g_object_get(sink, "stats", &stats, nullptr);
      guint64 actual = 0, dropped = 0;
      if (stats) {
        gst_structure_get_uint64(stats, "rendered", &actual);
        gst_structure_get_uint64(stats, "dropped", &dropped);
        gst_structure_free(stats);
      }
      gst_object_unref(sink);
      std::ofstream(out + "/hdmi_stats.json")
          << "{\"submitted\":" << rendered << ",\"rendered\":" << actual
          << ",\"dropped\":" << dropped << ",\"elapsed_s\":" << now() - start
          << "}\n";
      gst_element_set_state(pipeline, GST_STATE_NULL);
      gst_object_unref(src);
      gst_object_unref(pipeline);
    }
  }
};

struct Display {
  std::mutex mutex;
  std::condition_variable condition;
  Packet pending, published;
  std::shared_ptr<std::vector<uchar>> jpeg;
  std::exception_ptr error;
  std::atomic<bool> stop{false};
  std::thread renderer, listener;
  std::vector<std::thread> clients;
  std::atomic<int> active{0};
  int server = -1;
  std::string html, folder;
  bool hdmi;
  long displayed = 0;
  Display(bool on_hdmi, int port, const std::string &assets,
          const std::string &output)
      : folder(output), hdmi(on_hdmi) {
    std::ifstream f(assets + "/preview.html");
    html.assign(std::istreambuf_iterator<char>(f), {});
    if (!hdmi) {
      server = socket(AF_INET, SOCK_STREAM, 0);
      check(server >= 0, "HTTP socket");
      int one = 1;
      setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      addr.sin_port = htons(port);
      if (bind(server, (sockaddr *)&addr, sizeof(addr)) || listen(server, 8)) {
        close(server);
        server = -1;
        throw std::runtime_error("HTTP bind/listen");
      }
      fcntl(server, F_SETFL, fcntl(server, F_GETFL, 0) | O_NONBLOCK);
      for (int i = 0; i < 8; ++i)
        clients.emplace_back([this]() {
          while (!stop) {
            pollfd p{server, POLLIN, 0};
            if (poll(&p, 1, 200) <= 0)
              continue;
            int fd = accept(server, nullptr, nullptr);
            if (fd < 0)
              continue;
            serve(fd);
            close(fd);
          }
        });
    }
    renderer = std::thread([this, assets]() {
      try {
        affinity({0, 1, 3, 7});
        Screen screen(hdmi, assets, folder);
        long seq = 0;
        bool saved_depth = false, saved_stable = false;
        while (!stop) {
          Packet p;
          {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait_for(lock, std::chrono::milliseconds(200),
                               [&] { return stop || pending.seq > seq; });
            if (stop)
              break;
            if (pending.seq <= seq)
              continue;
            p = pending;
          }
          seq = p.seq;
          auto im = overlay(p);
          if (displayed == 0) {
            cv::imwrite(folder + "/rectified_overlay.jpg", im);
            cv::imwrite(folder + "/raw_left.jpg",
                        p.raw(cv::Rect(0, 0, 1280, 720)));
            cv::imwrite(folder + "/raw_right.jpg",
                        p.raw(cv::Rect(1280, 0, 1280, 720)));
          }
          if (hdmi) {
            auto composite = screen.draw(p, im);
            screen.push(composite);
            if (!saved_stable && p.seq >= 150) {
              cv::imwrite(folder + "/hdmi_stable.png", composite);
              saved_stable = true;
            }
            if (displayed == 0)
              cv::imwrite(folder + "/hdmi_first.png", composite);
            if (p.z && !saved_depth) {
              cv::imwrite(folder + "/hdmi_depth.png", composite);
              saved_depth = true;
            }
          }
          std::shared_ptr<std::vector<uchar>> encoded;
          if (!hdmi) {
            encoded = std::make_shared<std::vector<uchar>>();
            cv::imencode(".jpg", im, *encoded, {cv::IMWRITE_JPEG_QUALITY, 80});
          }
          {
            std::lock_guard<std::mutex> lock(mutex);
            published = p;
            jpeg = encoded;
            ++displayed;
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
  void submit(Packet p) {
    std::lock_guard<std::mutex> lock(mutex);
    if (error)
      std::rethrow_exception(error);
    pending = std::move(p);
    condition.notify_all();
  }
  static bool sendall(int fd, const void *data, size_t n) {
    auto p = static_cast<const char *>(data);
    while (n) {
      ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
      if (w < 0 && errno == EINTR)
        continue;
      if (w <= 0)
        return false;
      p += w;
      n -= w;
    }
    return true;
  }
  static bool sendtext(int fd, const std::string &s) {
    return sendall(fd, s.data(), s.size());
  }
  void serve(int fd) {
    try {
      timeval timeout{2, 0};
      setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
      setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
      char req[2048];
      int n = recv(fd, req, sizeof(req) - 1, 0);
      if (n <= 0)
        return;
      req[n] = 0;
      std::istringstream request(req);
      std::string method, path;
      request >> method >> path;
      path = path.substr(0, path.find('?'));
      if (method != "GET")
        return;
      auto response = [&](const std::string &type, const std::string &body) {
        sendtext(fd, "HTTP/1.1 200 OK\r\nConnection: close\r\nCache-Control: "
                     "no-store\r\nContent-Type: " +
                         type + "\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body);
      };
      if (path == "/") {
        response("text/html; charset=utf-8", html);
        return;
      }
      if (path == "/status") {
        std::string s;
        {
          std::lock_guard<std::mutex> lock(mutex);
          s = published.json;
          if (s.empty())
            s = "{\"frame_age_s\":999}";
          else {
            s.pop_back();
            s += ",\"frame_age_s\":" + fixed(now() - published.published, 4) +
                 ",\"displayed_frames\":" + std::to_string(displayed) + "}";
          }
        }
        // A slow HTTP client must never hold up detection or the renderer.
        response("application/json", s);
        return;
      }
      if (path != "/stream.mjpg" && path != "/frame.jpg") {
        sendtext(fd, "HTTP/1.1 404 Not Found\r\nConnection: "
                     "close\r\nContent-Length: 0\r\n\r\n");
        return;
      }
      bool stream = path == "/stream.mjpg";
      if (stream &&
          !sendtext(fd, "HTTP/1.1 200 OK\r\nConnection: "
                        "close\r\nCache-Control: no-store\r\nContent-Type: "
                        "multipart/x-mixed-replace; boundary=frame\r\n\r\n"))
        return;
      long previous = 0;
      while (!stop) {
        std::shared_ptr<std::vector<uchar>> image;
        long seq;
        {
          std::unique_lock<std::mutex> lock(mutex);
          condition.wait_for(lock, std::chrono::milliseconds(300), [&] {
            return stop || error || (jpeg && published.seq > previous);
          });
          if (stop || error)
            return;
          if (!jpeg || published.seq <= previous)
            continue;
          image = jpeg;
          seq = published.seq;
        }
        std::string header =
            (stream ? "--frame\r\n"
                    : "HTTP/1.1 200 OK\r\nConnection: close\r\n") +
            std::string("Content-Type: image/jpeg\r\nX-Frame-Id: ") +
            std::to_string(seq) +
            "\r\nContent-Length: " + std::to_string(image->size()) + "\r\n\r\n";
        if (!sendtext(fd, header) ||
            !sendall(fd, image->data(), image->size()) || !sendtext(fd, "\r\n"))
          return;
        previous = seq;
        if (!stream)
          return;
      }
    } catch (...) {
    }
  }
  ~Display() {
    stop = true;
    condition.notify_all();
    if (listener.joinable())
      listener.join();
    for (auto &c : clients)
      c.join();
    if (server >= 0)
      close(server);
    if (renderer.joinable())
      renderer.join();
  }
};
