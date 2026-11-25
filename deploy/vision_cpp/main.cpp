// Native fixed-camera vision only. This executable contains no serial or motor
// interfaces. Geometric offsets are observations, not actuator commands.
#include "core.hpp"
#include "display.hpp"
#include "source.hpp"
#include <filesystem>
#include <openssl/sha.h>
namespace fs = std::filesystem;
volatile sig_atomic_t interrupted = 0;
void signal_stop(int) { interrupted = 1; }
std::string sha256(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  check(bool(f), "hash file open");
  SHA256_CTX ctx;
  SHA256_Init(&ctx);
  char buf[65536];
  while (f) {
    f.read(buf, sizeof(buf));
    SHA256_Update(&ctx, buf, f.gcount());
  }
  check(f.eof(), "hash read");
  unsigned char d[SHA256_DIGEST_LENGTH];
  SHA256_Final(d, &ctx);
  std::ostringstream s;
  s << std::hex << std::setfill('0');
  for (auto b : d)
    s << std::setw(2) << int(b);
  return s.str();
}
#include "detector_process.hpp"
struct Guidance {
  std::optional<Box> previous;
  long target_start = 0, aligned_since = 0;
  std::string cycle, run;
  Guidance(std::string id) : run(std::move(id)) {}
  std::string update(Packet &p, Geometry &g) {
    if (!same(previous, p.box)) {
      target_start = p.seq;
      aligned_since = 0;
      cycle.clear();
    }
    previous = p.box;
    std::string state = "no_target", offset = "null", mm = "null",
                center = "null", ref = "null", cycle_json = "null";
    bool began = false;
    if (p.box) {
      auto b = *p.box;
      double x = (b.l + b.r) / 2., y = (b.t + b.b) / 2.;
      p.dx = x - 640;
      p.dy = y - 360;
      p.aligned = std::abs(p.dx) <= 51.2 && std::abs(p.dy) <= 28.8;
      offset = "[" + number(p.dx) + "," + number(p.dy) + "]";
      center = "[" + number(x) + "," + number(y) + "]";
      bool valid = p.z && p.source >= target_start && p.source <= p.seq;
      if (valid)
        mm = "[" + number(std::nearbyint(p.dx * (*p.z) * 10000 / g.fx) / 10) +
             "," + number(std::nearbyint(p.dy * (*p.z) * 10000 / g.fy) / 10) +
             "]";
      if (!p.aligned) {
        aligned_since = 0;
        cycle.clear();
        state = "adjust";
      } else {
        if (!aligned_since)
          aligned_since = p.seq;
        state = "aligned_wait_depth";
        if (valid && p.source >= aligned_since) {
          if (cycle.empty() && p.status == "new") {
            cycle = run + "/" + std::to_string(aligned_since);
            began = true;
          }
          if (!cycle.empty()) {
            state = "ready_dry_run";
            ref = number(std::nearbyint(*p.z * 1000));
            cycle_json = "\"" + cycle + "\"";
          }
        }
      }
    }
    return "{\"image_center_px\":[640,360],\"tolerance_px\":[51.2,28.8],"
           "\"target_center_px\":" +
           center + ",\"offset_px\":" + offset +
           ",\"estimated_offset_mm\":" + mm +
           ",\"aligned_xy\":" + (p.aligned ? "true" : "false") +
           ",\"state\":\"" + state + "\",\"reference_depth_mm\":" + ref +
           ",\"cycle_id\":" + cycle_json +
           ",\"feed_preview_started\":" + (began ? "true" : "false") + "}";
  }
};
int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN);
  signal(SIGINT, signal_stop);
  signal(SIGTERM, signal_stop);
  signal(SIGHUP, signal_stop);
  if (argc > 1 && std::string(argv[1]) == "--detector-worker")
    return detector_worker(argc, argv);
  try {
    std::map<std::string, std::string> a{
        {"--seconds", "3600"},
        {"--port", "8788"},
        {"--depth-scale", "0.5"},
        {"--id", "live"},
        {"--device",
         "/dev/v4l/by-id/usb-USB_Camera_USB_Camera_01.00.00-video-index0"}};
    bool hdmi = false, disable_model = false, raw_preview = false;
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k == "--disable-model") {
        disable_model = true;
        continue;
      }
      if (k == "--raw-preview") {
        raw_preview = true;
        continue;
      }
      if (k == "--hdmi") {
        hdmi = true;
        continue;
      }
      check(k == "--seconds" || k == "--port" || k == "--depth-scale" ||
                k == "--id" || k == "--device" || k == "--model" ||
                k == "--calibration" || k == "--calibration-sha" ||
                k == "--assets" || k == "--output" || k == "--replay" ||
                k == "--cores" || k == "--tracking-overlay",
            "unknown argument");
      check(i + 1 < argc, "missing argument value");
      a[k] = argv[++i];
    }
    double seconds = std::stod(a.at("--seconds")),
           scale = std::stod(a.at("--depth-scale"));
    int port = std::stoi(a.at("--port"));
    check(std::isfinite(seconds) && seconds > 0 &&
              (scale == .5 || scale == 1) && port > 0 && port < 65536,
          "invalid duration/scale/port");
    std::string id = a.at("--id");
    check(!id.empty() && id.find_first_not_of(
                             "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVW"
                             "XYZ0123456789_.-") == std::string::npos,
          "invalid run id");
    std::string modelpath = a.count("--model") ? a["--model"] : "";
    std::string calpath = a.count("--calibration") ? a["--calibration"] : "";
    std::string hash = "unavailable";
    std::string calibration_hash = "unavailable";
    bool calibration_valid = false;
    std::unique_ptr<Geometry> calibrated;
    if (!raw_preview) {
      try {
        calibration_hash = sha256(calpath);
        check(calibration_hash == a.at("--calibration-sha"),
              "calibration matrix hash mismatch");
        calibrated = std::make_unique<Geometry>(calpath);
        calibration_valid = true;
      } catch (const std::exception &e) {
        std::cerr << "calibration unavailable: " << e.what() << '\n';
      }
    }
    Geometry raw_geometry;
    Geometry &geo = calibrated ? *calibrated : raw_geometry;
    try {
      hash = sha256(modelpath);
    } catch (const std::exception &) {
    }
    std::string output = a.at("--output");
    check(fs::create_directory(output), "output exists or cannot be created");
    std::cerr << "MODEL=" << modelpath << " SHA256=" << hash
              << "\nCALIBRATION=" << calpath << " SHA256=" << calibration_hash
              << "\nRGB / 640x360+140 vertical padding / score .536 / NMS .45 "
                 "/ depth_scale="
              << scale << " / serial disabled\nOUTPUT=" << fs::absolute(output)
              << '\n';
    std::ofstream(output + "/binding.json")
        << "{\"model_sha256\":\"" << hash << "\",\"calibration_sha256\":\""
        << calibration_hash << "\",\"depth_scale\":" << scale
        << ",\"backend\":\"cpp\",\"input\":\""
        << (a.count("--replay") ? "replay" : "camera")
        << "\",\"serial_enabled\":false}\n";
    affinity({4, 5, 6, 7});
    cv::setNumThreads(1);
    std::unique_ptr<ProcessStereo> stereo;
    if (calibration_valid) {
      try {
        stereo = std::make_unique<ProcessStereo>(geo, scale);
      } catch (const std::exception &e) {
        std::cerr << "range unavailable: " << e.what() << '\n';
      }
    }
    affinity({6});
    ProcessDetector model(fs::absolute(argv[0]).string(), modelpath,
                          a.count("--cores") ? a["--cores"] : "auto");
    std::vector<cv::Mat> images;
    if (a.count("--replay")) {
      std::ifstream list(a["--replay"]);
      std::string path;
      std::map<std::string, cv::Mat> cache;
      while (std::getline(list, path)) {
        if (!cache.count(path))
          cache[path] = cv::imread(path);
        auto im = cache[path];
        check(im.cols == 2560 && im.rows == 720, "replay image");
        images.push_back(im);
      }
      check(!images.empty(), "empty replay corpus");
    }
    Display display(hdmi, port, a.at("--assets"), output,
                    a.count("--tracking-overlay") ? a["--tracking-overlay"]
                                                  : "");
    Source reader(geo, a["--device"], images);
    Guidance guidance(id);
    std::ofstream log(output + "/frames.jsonl");
    check(bool(log), "frame log");
    log << std::setprecision(12);
    std::optional<Box> previous, source_box, accepted_box;
    std::optional<double> value;
    long epoch = 0, source_epoch = 0, source = 0, accepted_source = 0, seq = 0,
         count = 0, newcount = 0, attemptcount = 0;
    double source_time = 0, accepted_time = 0, start = now(),
           rate_start = start;
    long rate_count = 0, rate_new = 0;
    double fps = 0, hz = 0;
    cv::Mat thumb;
    cv::Rect roi;
    std::string last_reason = "no_measurement";
    const char *reasons[] = {"valid", "insufficient_disparity",
                             "disparity_quality", "range_limit"};
    while (!interrupted && now() - start < seconds) {
      auto item = reader.after(seq);
      seq = item.seq;
      Timings timing;
      timing["left_remap_ms"] = item.remap_ms;
      timing["left_remap_opencl"] = item.remap_opencl ? 1 : 0;
      auto detections = (!disable_model && calibration_valid)
                            ? model.detect_with_recovery(item.left, timing)
                            : std::vector<Box>{};
      std::optional<Box> box;
      if (!detections.empty())
        box = detections[0];
      if (!same(previous, box)) {
        ++epoch;
        value.reset();
        accepted_box.reset();
        thumb.release();
        last_reason = box ? "target_changed" : "no_target";
      }
      previous = box;
      if (accepted_box &&
          (now() - accepted_time > 1.5 || !same(accepted_box, box))) {
        value.reset();
        accepted_box.reset();
        thumb.release();
        last_reason = "expired";
      }
      std::optional<double> fresh;
      std::optional<Range> attempt;
      bool discarded = false;
      long attempted_source = source;
      double attempted_time = source_time;
      try {
        if (stereo && stereo->ready()) {
          attempt = stereo->get();
          ++attemptcount;
          if (epoch == source_epoch && same(box, source_box) &&
              now() - source_time <= 1.5) {
            fresh = attempt->z;
            last_reason = reasons[attempt->reason];
            if (fresh) {
              value = value ? .9 * *fresh + .1 * *value : *fresh;
              accepted_source = source;
              accepted_time = source_time;
              accepted_box = source_box;
              thumb = attempt->thumb;
              roi = attempt->roi;
              ++newcount;
              ++rate_new;
            }
          } else {
            discarded = true;
            last_reason = "obsolete_target_or_frame";
          }
        }
        if (box && stereo && !stereo->pending) {
          source = seq;
          source_time = item.arrival;
          source_box = box;
          source_epoch = epoch;
          stereo->submit(item.left, item.raw(cv::Rect(1280, 0, 1280, 720)),
                         *box, timing);
        }
        if (stereo && stereo->pending && now() - stereo->submitted > 2)
          throw std::runtime_error("range timeout");
      } catch (const std::exception &e) {
        std::cerr << "range disabled: " << e.what() << '\n';
        stereo.reset();
        value.reset();
        accepted_box.reset();
        thumb.release();
      }
      double end = now();
      ++count;
      ++rate_count;
      if (end - rate_start >= 1) {
        fps = rate_count / (end - rate_start);
        hz = rate_new / (end - rate_start);
        rate_count = rate_new = 0;
        rate_start = end;
      }
      Packet p;
      p.seq = seq;
      p.source = accepted_source;
      p.published = end;
      p.fps = fps;
      p.hz = hz;
      p.raw = item.raw;
      p.left = item.left;
      p.box = box;
      p.z = value;
      p.status = fresh ? "new" : value ? "held" : "invalid";
      p.thumb = thumb;
      p.roi = roi;
      auto g =
          calibration_valid ? guidance.update(p, geo) : std::string("null");
      std::ostringstream row;
      row << std::setprecision(12);
      std::string selected = box ? boxes_json({*box}) : "[]";
      selected = box ? selected.substr(1, selected.size() - 2) : "null";
      row << "{\"frame_id\":" << seq
          << ",\"backend\":\"cpp\",\"arrival\":" << item.arrival
          << ",\"published_at\":" << end
          << ",\"detected\":" << (box ? "true" : "false")
          << ",\"selected\":" << selected
          << ",\"detections\":" << boxes_json(detections)
          << ",\"tracking_candidates\":"
          << boxes_json(model.tracking_candidates)
          << ",\"distance_m\":" << number(value)
          << ",\"raw_distance_m\":" << number(fresh)
          << ",\"measurement_status\":\"" << p.status
          << "\",\"distance_source_frame\":"
          << (value ? std::to_string(accepted_source) : "null")
          << ",\"distance_age_ms\":"
          << (value ? number((end - accepted_time) * 1000) : "null")
          << ",\"distance_reason\":\"" << last_reason
          << "\",\"distance_kind\":\"camera_forward_Z_bucket_body\",\"depth_"
             "scale\":"
          << scale << ",\"processing_fps\":" << fps
          << ",\"fresh_distance_hz\":" << hz << ",\"guidance\":" << g
          << ",\"detector\":" << timings_json(timing) << ",\"sampling_roi\":";
      if (value)
        row << '[' << roi.x << ',' << roi.y << ',' << roi.x + roi.width << ','
            << roi.y + roi.height << ']';
      else
        row << "null";
      row << ",\"attempt_source\":"
          << (attempt ? std::to_string(attempted_source) : "null")
          << ",\"attempt_source_age_ms\":"
          << (attempt ? number((end - attempted_time) * 1000) : "null")
          << ",\"discarded\":" << (discarded ? "true" : "false")
          << ",\"worker\":"
          << (attempt ? timings_json(attempt->timing) : "null") << '}';
      std::string detector_state =
          disable_model || !calibration_valid ? "unavailable" : model.state;
      std::string detector_code = disable_model        ? "MODEL_DISABLED"
                                  : !calibration_valid ? "CALIBRATION_INVALID"
                                                       : model.code;
      auto base = row.str();
      base.pop_back();
      row.str("");
      row.clear();
      row << base;
      row << ",\"components\":{\"detector\":{\"state\":\"" << detector_state
          << "\",\"code\":\"" << detector_code << "\",\"message\":\""
          << (detector_state == "ready" ? "正常" : "检测不可用，正在重新检查")
          << "\"},"
          << "\"calibration\":{\"state\":\""
          << (calibration_valid ? "ready" : "unavailable") << "\",\"code\":\""
          << (calibration_valid ? "OK" : "CALIBRATION_INVALID")
          << "\",\"message\":\""
          << (calibration_valid ? "正常" : "标定不可用，仅显示原始画面")
          << "\"},"
          << "\"ranging\":{\"state\":\"" << (stereo ? "ready" : "unavailable")
          << "\",\"code\":\"" << (stereo ? "OK" : "RANGING_UNAVAILABLE")
          << "\",\"message\":\"" << (stereo ? "正常" : "测距不可用") << "\"}}}";
      p.json = row.str();
      log << p.json << '\n';
      if (count % 30 == 0)
        log.flush();
      display.submit(std::move(p));
    }
    // Workers have bounded shutdown; inference failure never owns camera
    // lifetime.
    stereo.reset();
    double duration = now() - start;
    std::ostringstream summary;
    summary << "{\"seconds\":" << duration << ",\"frames\":" << count
            << ",\"processing_fps\":" << count / duration
            << ",\"new_measurements\":" << newcount
            << ",\"fresh_distance_hz\":" << newcount / duration
            << ",\"attempts\":" << attemptcount
            << ",\"attempt_hz\":" << attemptcount / duration << "}";
    std::ofstream(output + "/summary.json") << summary.str() << '\n';
    std::cerr << "COMPLETE " << summary.str() << '\n';
  } catch (const std::exception &e) {
    std::cerr << "ERROR " << e.what() << '\n';
    return 1;
  }
  return 0;
}
