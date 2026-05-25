#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask 网页服务模块
"""

import time
import cv2
from flask import Flask, Response, render_template_string


def create_app(state_store, lifecycle):
    app = Flask(__name__)

    def generate_frames():
        while lifecycle.is_active():
            frame = state_store.get_latest_display_frame()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                time.sleep(0.1)

    @app.route('/')
    def index():
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>双目测距系统</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px;
                       background-color: #1a1a1a; color: #ffffff; }
                .container { max-width: 1200px; margin: 0 auto; }
                h1 { text-align: center; color: #4CAF50; }
                .video-container { text-align: center; margin: 20px 0;
                                   background-color: #2a2a2a; padding: 20px;
                                   border-radius: 10px; }
                .video-container img { max-width: 100%; height: auto;
                                       border: 2px solid #4CAF50; border-radius: 5px; }
                .info-panel { background-color: #2a2a2a; padding: 20px;
                              border-radius: 10px; margin-top: 20px; }
                .distance-display { font-size: 24px; font-weight: bold;
                                    color: #4CAF50; text-align: center; padding: 10px; }
                .detect-display { font-size: 20px; font-weight: bold;
                                  color: #ffd166; text-align: center; padding: 6px; }
                .status { text-align: center; color: #888; margin-top: 10px; }
            </style>
            <script>
                function updateDistance() {
                    fetch('/distance')
                        .then(response => response.text())
                        .then(data => { document.getElementById('distance').textContent = data; })
                        .catch(error => { document.getElementById('distance').textContent = 'N/A'; });
                }
                function updateStatus() {
                    fetch('/status')
                        .then(response => response.text())
                        .then(data => { document.getElementById('detect').textContent = data; })
                        .catch(error => { document.getElementById('detect').textContent = '未知'; });
                }
                setInterval(updateDistance, 500);
                setInterval(updateStatus, 500);
                window.onload = function() {
                    updateDistance();
                    updateStatus();
                };
            </script>
        </head>
        <body>
            <div class="container">
                <h1>双目测距系统</h1>
                <div class="video-container">
                    <img src="/video" alt="实时视频流">
                </div>
                <div class="info-panel">
                    <div class="distance-display">
                        当前距离: <span id="distance">加载中...</span>
                    </div>
                    <div class="detect-display">
                        检测状态: <span id="detect">加载中...</span>
                    </div>
                    <div class="status">
                        提示：低光 / 强反光区域距离可能显示为 N/A（视差不稳定时自动屏蔽）
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return render_template_string(html)

    @app.route('/video')
    def video_feed():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/distance')
    def get_distance():
        d = state_store.get_latest_distance()
        if d is not None:
            return f"{d:.3f} m"
        return "N/A"

    @app.route('/status')
    def get_status():
        detected = state_store.get_latest_detected()
        return "bucket" if detected else "未检测到 bucket"

    return app


def run_web_server(state_store, lifecycle, host='0.0.0.0', port=5050):
    print(f"Flask 服务器启动: http://<板卡IP>:{port}/")
    app = create_app(state_store, lifecycle)
    app.run(host=host, port=port,
            debug=False, threaded=True, use_reloader=False)
