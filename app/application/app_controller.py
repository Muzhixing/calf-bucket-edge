#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用控制器模块

该模块实现了分层架构中的应用层控制器，
通过异步事件总线协调基础设施层与表现层之间的数据交互。

主要功能：
- 初始化和管理测距服务
- 启动和管理 Web 服务器线程
- 处理程序生命周期（启动、运行、停止）
- 提供统一的程序入口点
"""

import threading
import time

from app.domain.events import RangingUpdateEvent
from app.presentation import web


class AppController:
    """
    应用控制器类
    
    负责协调双目测距系统的各个组件，包括测距服务、事件总线和 Web 服务器。
    采用分层架构模式，通过事件驱动实现层间解耦。
    
    Attributes:
        ranging_service: 测距服务实例，负责双目视觉测距功能（基础设施层）
        event_bus: 异步事件总线（应用层）
        state_store: 状态缓存（应用层）
        serial_gateway: 串口输出层（基础设施层，占位）
        web_host: Web 服务器监听地址，默认为 '0.0.0.0'（监听所有网络接口）
        web_port: Web 服务器监听端口，默认为 5050
        _web_thread: Web 服务器线程对象，用于后台运行 Flask 服务
    
    Example:
        >>> from app.infrastructure.ranging_service import RangingService
        >>> from app.application.event_bus import AsyncEventBus
        >>> from app.application.state_store import RangingStateStore
        >>> ranging = RangingService(...)
        >>> controller = AppController(ranging, AsyncEventBus(), RangingStateStore(), web_port=8080)
        >>> controller.run()
    """

    def __init__(self, ranging_service, event_bus, state_store, serial_gateway=None,
                 web_host='0.0.0.0', web_port=5050):
        """
        初始化应用控制器
        
        Args:
            ranging_service: 测距服务实例，必须实现 start()、stop() 和 is_active() 方法
            event_bus: 异步事件总线
            state_store: 状态缓存
            serial_gateway: 串口输出层（占位）
            web_host: Web 服务器监听地址，'0.0.0.0' 表示监听所有网络接口，
                     '127.0.0.1' 表示仅监听本地回环接口
            web_port: Web 服务器监听端口，确保端口未被占用
        """
        self.ranging_service = ranging_service
        self.event_bus = event_bus
        self.state_store = state_store
        self.serial_gateway = serial_gateway
        self.web_host = web_host
        self.web_port = web_port
        self._web_thread = None  # Web 服务器线程，延迟初始化
        self._wire_events()

    def _wire_events(self):
        self.event_bus.subscribe(RangingUpdateEvent, self.state_store.handle_ranging_update)
        if self.serial_gateway is not None:
            self.event_bus.subscribe(RangingUpdateEvent, self.serial_gateway.handle_ranging_update)

    def is_active(self):
        return self.ranging_service.is_active()

    def run(self):
        """
        启动应用主循环
        
        执行以下操作：
        1. 打印系统启动信息和配置
        2. 启动测距服务
        3. 在后台线程中启动 Web 服务器
        4. 保持主循环运行，直到测距服务停止或用户中断
        
        支持通过 Ctrl+C 优雅退出，会正确清理资源。
        """
        # 打印系统启动信息和配置
        print("=" * 60)
        print("双目测距系统启动（分层架构 + 事件驱动）")
        print("=" * 60)
        print(f"摄像头设备: /dev/video{self.ranging_service.camera_device}")
        print(f"分辨率: {self.ranging_service.frame_width}x{self.ranging_service.frame_height}")
        print(f"显示模式: {'启用' if self.ranging_service.enable_display else '禁用（仅 Flask）'}")
        print(f"Flask: http://<板卡IP>:{self.web_port}/")
        
        # 如果启用了推送功能，显示推送配置
        if self.ranging_service.enable_push:
            print("推送模式: webrtc")
            print(f"WebRTC 信令: {self.ranging_service.webrtc_signal_url or '未配置'}")
            print(f"WebRTC STUN: {self.ranging_service.webrtc_stun_urls or '未配置'}")
        
        print("退出方式: Ctrl+C（终端）")
        print("=" * 60)

        try:
            # 启动事件总线与串口层（如启用）
            self.event_bus.start()
            if self.serial_gateway is not None:
                self.serial_gateway.start()

            # 启动测距服务（开始处理视频流和测距计算）
            self.ranging_service.start()
            
            # 在后台线程中启动 Web 服务器
            # 使用 daemon 线程，确保主程序退出时自动终止
            self._web_thread = threading.Thread(
                target=web.run_web_server,
                kwargs={"state_store": self.state_store,
                        "lifecycle": self,
                        "host": self.web_host,
                        "port": self.web_port},
                daemon=True  # 守护线程，主程序退出时自动终止
            )
            self._web_thread.start()

            # 主循环：保持程序运行，直到测距服务停止
            # 定期检查服务状态，避免 CPU 占用过高
            while self.ranging_service.is_active():
                time.sleep(0.5)

        except KeyboardInterrupt:
            # 用户按下 Ctrl+C，优雅退出
            print("\n用户中断程序")
            self.ranging_service.stop()
        finally:
            # 确保资源清理：停止测距服务
            self.ranging_service.stop()
            if self.serial_gateway is not None:
                self.serial_gateway.stop()
            self.event_bus.stop()
            # 等待一段时间，确保资源完全释放
            time.sleep(1.0)
            print("程序退出。")
