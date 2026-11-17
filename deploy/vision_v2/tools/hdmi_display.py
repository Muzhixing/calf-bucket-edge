"""Rockchip HDMI-A-1 scanout through DRM/KMS. No window system or display socket."""
import ctypes as C
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import gi
from PIL import Image, ImageDraw, ImageFont

gi.require_version('Gst', '1.0')
from gi.repository import Gst


def hdmi_connector():
    card = Path('/dev/dri/by-path/platform-display-subsystem-card').resolve(strict=True)
    path = Path('/sys/class/drm') / (card.name + '-HDMI-A-1')
    if (path/'status').read_text().strip() != 'connected' or '1280x720' not in (path/'modes').read_text().splitlines():
        raise RuntimeError('HDMI-A-1 must be connected and advertise 1280x720')
    # Prefixes of the public libdrm structs. The vendor kernel lacks sysfs connector_id.
    u32 = C.c_uint32
    class Resources(C.Structure):
        _fields_ = [('nfb', C.c_int), ('fb', C.POINTER(u32)), ('ncrtc', C.c_int),
                    ('crtc', C.POINTER(u32)), ('count', C.c_int), ('ids', C.POINTER(u32))]
    class Connector(C.Structure):
        _fields_ = [('id', u32), ('encoder', u32), ('type', u32), ('index', u32), ('connected', C.c_int)]
    drm = C.CDLL('libdrm.so.2')
    drm.drmModeGetResources.restype = C.POINTER(Resources)
    drm.drmModeGetConnectorCurrent.restype = C.POINTER(Connector)
    with card.open('rb', buffering=0) as device:
        res = drm.drmModeGetResources(device.fileno())
        if res:
            try:
                for cid in res.contents.ids[:res.contents.count]:
                    conn = drm.drmModeGetConnectorCurrent(device.fileno(), cid)
                    if conn:
                        try:
                            c = conn.contents
                            if (c.type, c.index, c.connected) == (11, 1, 1):  # HDMI-A-1, connected
                                return cid
                        finally:
                            drm.drmModeFreeConnector(conn)
            finally:
                drm.drmModeFreeResources(res)
    raise RuntimeError('Connected Rockchip HDMI-A-1 connector unavailable')


class HdmiDisplay:
    def __init__(self, folder, model, calibration):
        self.folder = folder
        self.connector = hdmi_connector()
        Gst.init(None)
        self.pipeline = Gst.parse_launch(
            'appsrc name=src is-live=true format=time do-timestamp=true block=false '
            'max-buffers=1 leaky-type=downstream caps=video/x-raw,format=BGRx,width=1280,height=720,framerate=30/1 '
            '! kmssink name=screen driver-name=rockchip connector-id=%d force-modesetting=true '
            'restore-crtc=true enable-last-sample=false sync=false' % self.connector)
        self.src = self.pipeline.get_by_name('src')
        self.sink = self.pipeline.get_by_name('screen')
        self.bus = self.pipeline.get_bus()
        self.condition = threading.Condition()
        self.latest = self.error = None
        self.stopping = False
        self.submitted = 0
        self.depth_snapshot = False
        self.last_report = 0
        self.first = self.last = None
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 21, index=2)
        template = Image.new('RGB', (1280,720), (17,24,24))
        draw = ImageDraw.Draw(template)
        for xy, label in [((16,2),'桶身距离'),((248,2),'检测 / 对齐'),((524,2),'距离状态'),
                          ((16,78),'校正左目'),((928,4),'原始左目'),((928,235),'原始右目'),
                          ((928,469),'最近有效视差'),((1021,585),'暂无有效视差'),((16,624),'桶口中心未独立定位 · 圆点为桶框中心'),
                          ((16,657),'橙线框为距离来源帧的采样区')]:
            draw.text(xy,label,font=font,fill=(191,210,204))
        self.base = cv2.cvtColor(np.asarray(template),cv2.COLOR_RGB2BGR)
        cv2.putText(self.base, f'{model}  /  {calibration}', (158,101), 0,.60,(192,206,200),1,cv2.LINE_AA)
        self.labels = {}
        for word in ['未检出桶','待对齐','两轴已对齐','新测','沿用','无有效值']:
            tile = Image.new('RGB',(175,38),(17,24,24))
            ImageDraw.Draw(tile).text((0,0),word,font=font,fill=(229,239,235))
            self.labels[word] = cv2.cvtColor(np.asarray(tile),cv2.COLOR_RGB2BGR)
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError('Cannot open HDMI KMS pipeline')
        self.thread = threading.Thread(target=self.render, name='hdmi-output', daemon=True)
        self.thread.start()

    def offer(self, row, im, views):
        with self.condition:
            if self.error:
                raise RuntimeError('HDMI output failed') from self.error
            self.latest = (row, im, views)
            self.condition.notify()

    def compose(self, row, im, views):
        raw_left, raw_right, disparity, roi = views
        canvas = self.base.copy()
        cv2.resize(im,(896,504),dst=canvas[112:616,16:912])
        if roi is not None:
            x1,y1,x2,y2 = [round(n*.7) for n in roi]
            cv2.rectangle(canvas,(16+x1,112+y1),(16+x2,112+y2),(0,153,255),2)
        for raw,y in [(raw_left,39),(raw_right,274)]:
            cv2.resize(raw,(336,189),dst=canvas[y:y+189,928:1264])
        if disparity is not None:
            colored = cv2.applyColorMap(disparity, cv2.COLORMAP_TURBO)
            colored[disparity==0] = 0
            canvas[507:696,928:1264] = colored
        detect = '未检出桶' if not row['detected'] else ('两轴已对齐' if row['guidance']['aligned_xy'] else '待对齐')
        state = {'new':'新测','held':'沿用','invalid':'无有效值'}[row['measurement_status']]
        canvas[36:74,248:423] = self.labels[detect]
        canvas[36:74,524:699] = self.labels[state]
        def text(msg, x, y, size=.68):
            cv2.putText(canvas,msg,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,(228,241,235),2,cv2.LINE_AA)
        valid = row['detected'] and row['distance_m'] is not None
        text(f"{row['distance_m']*100:.0f} cm" if valid else '-- cm',16,64,1.22)
        text(f"{row['processing_fps'] or 0:.0f} FPS / {row['fresh_distance_hz'] or 0:.1f} Hz",690,62,.55)
        offset = row['guidance']['offset_px']
        if offset:
            text(f"dx {offset[0]:+.0f}   dy {offset[1]:+.0f} px",532,650)
        text(f"frame {row['frame_id']}   Z src {row['distance_source_frame'] or '--'}",532,687,.59)
        return canvas

    def render(self):
        try:
            os.sched_setaffinity(0,{0,1,2,3,7})
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping or self.latest is not None)
                    if self.stopping:
                        return
                    packet, self.latest = self.latest, None
                message = self.bus.pop_filtered(Gst.MessageType.ERROR)
                if message:
                    raise RuntimeError(str(message.parse_error()))
                frame = self.compose(*packet)
                data = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA).tobytes()
                result = self.src.emit('push-buffer',Gst.Buffer.new_wrapped(data))
                if result != Gst.FlowReturn.OK:
                    raise RuntimeError(str(result))
                self.last = time.monotonic()
                if self.first is None:
                    self.first = self.last
                self.submitted += 1
                has_depth = packet[2][2] is not None
                if self.submitted == 1 or (has_depth and not self.depth_snapshot):
                    if not cv2.imwrite(str(self.folder/'hdmi_preview.jpg'),frame):
                        raise OSError('Failed to save HDMI preview')
                    self.depth_snapshot = has_depth
                if self.last - self.last_report >= 5:
                    tmp = self.folder/'hdmi_stats.tmp'
                    tmp.write_text(json.dumps(self.stats())+'\n')
                    tmp.replace(self.folder/'hdmi_stats.json')
                    self.last_report = self.last
        except Exception as exc:
            self.error = exc

    def stats(self):
        return dict(connector_id=self.connector, submitted=self.submitted,
                    rendered=self.sink.get_property('stats').get_value('rendered'),
                    elapsed_s=(self.last-self.first) if self.first is not None else 0)

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.thread.join()
        try:
            stats = self.stats()
            message = self.bus.pop_filtered(Gst.MessageType.ERROR)
            if self.error or message or not stats['rendered']:
                raise RuntimeError(str(self.error or (message.parse_error() if message else 'No KMS frames rendered')))
            return stats
        finally:
            self.pipeline.set_state(Gst.State.NULL)
