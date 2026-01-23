#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPS NMEA 解析模块

该模块提供 GPS NMEA 数据解析功能，支持 GGA 和 RMC 消息格式，
并提供坐标转换、格式化输出等功能。
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict

import serial


@dataclass
class FixState:
    """
    GPS 定位状态数据类
    
    用于存储从 NMEA 消息中解析出的 GPS 定位信息，包括位置、速度、时间等。
    数据来源于 GGA 和 RMC 两种 NMEA 消息类型。
    
    Attributes:
        fix_quality: 定位质量，0=无定位, 1=GPS定位, 2=DGPS定位, 3=PPS定位等
        sats: 参与定位的卫星数量
        hdop: 水平精度因子 (Horizontal Dilution of Precision)
        altitude_m: 海拔高度（米）
        valid: 数据有效性，True表示有效(A)，False表示无效(V)
        speed_knots: 速度（节）
        course_deg: 航向角（度，0-360）
        date_ddmmyy: 日期（格式：DDMMYY）
        time_hhmmss: UTC时间（格式：HHMMSS或HHMMSS.SS）
        lat_deg: 纬度（十进制度数，-90到90）
        lon_deg: 经度（十进制度数，-180到180）
    """
    # 来自 GGA 消息的字段
    fix_quality: Optional[int] = None  # 0=无定位, 1=GPS, 2=DGPS, 3=PPS, 4=RTK, 5=RTK浮点解等
    sats: Optional[int] = None  # 参与定位的卫星数量
    hdop: Optional[float] = None  # 水平精度因子，值越小精度越高
    altitude_m: Optional[float] = None  # 海拔高度（米）

    # 来自 RMC 消息的字段
    valid: Optional[bool] = None  # A=True(有效), V=False(无效)
    speed_knots: Optional[float] = None  # 速度（节，1节=1.852公里/小时）
    course_deg: Optional[float] = None  # 航向角（度，正北为0°，顺时针增加）
    date_ddmmyy: Optional[str] = None  # 日期字符串，格式：DDMMYY
    time_hhmmss: Optional[str] = None  # UTC时间字符串，格式：HHMMSS或HHMMSS.SS

    # GGA 和 RMC 消息共有的字段
    lat_deg: Optional[float] = None  # 纬度（十进制度数）
    lon_deg: Optional[float] = None  # 经度（十进制度数）


def nmea_checksum_ok(line: str) -> bool:
    """
    验证 NMEA 消息的校验和
    
    NMEA 消息格式：$<消息体>*<校验和>
    校验和计算：对消息体（$之后，*之前的所有字符）进行异或运算，结果为十六进制。
    
    Args:
        line: NMEA 消息字符串，格式如 "$GNGGA,...*5A"
    
    Returns:
        bool: 校验和正确返回 True，否则返回 False
    """
    line = line.strip()
    if not line.startswith("$") or "*" not in line:
        return False
    body, cs = line[1:].split("*", 1)
    try:
        expected = int(cs[:2], 16)  # 解析十六进制校验和
    except ValueError:
        return False
    # 计算消息体的异或校验和
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    return calc == expected


def dm_to_deg(dm: str, hemi: str) -> Optional[float]:
    """
    将 NMEA 格式的度分（ddmm.mmmm 或 dddmm.mmmm）转换为十进制度数
    
    NMEA 格式说明：
    - 纬度：ddmm.mmmm（度分格式，如 3112.3456 表示 31度12.3456分）
    - 经度：dddmm.mmmm（度分格式，如 12123.4567 表示 121度23.4567分）
    
    Args:
        dm: 度分格式的字符串，如 "3112.3456" 或 "12123.4567"
        hemi: 半球指示符，"N"/"S" 用于纬度，"E"/"W" 用于经度
    
    Returns:
        Optional[float]: 转换后的十进制度数，南纬或西经为负值，转换失败返回 None
    
    Example:
        >>> dm_to_deg("3112.3456", "N")
        31.20576
        >>> dm_to_deg("12123.4567", "E")
        121.390945
    """
    if not dm or not hemi:
        return None
    try:
        v = float(dm)
    except ValueError:
        return None
    # 提取度数和分数：整数部分除以100得到度数，余数为分数
    deg = int(v // 100)
    minutes = v - deg * 100
    # 转换为十进制度数：度数 + 分数/60
    dec = deg + minutes / 60.0
    # 南纬和西经为负值
    if hemi.upper() in ("S", "W"):
        dec = -dec
    return dec


def parse_gga(fields: list, st: FixState) -> None:
    """
    解析 GGA (Global Positioning System Fix Data) NMEA 消息
    
    GGA 消息格式：
    $GNGGA,hhmmss.ss,lat,N,lon,E,fix_quality,sats,hdop,alt,M,geoid,M,age,diff_station*cs
    
    字段说明：
    - fields[0]: 消息类型（如 "GNGGA"）
    - fields[1]: UTC时间（HHMMSS.SS）
    - fields[2]: 纬度（度分格式）
    - fields[3]: 纬度半球（N/S）
    - fields[4]: 经度（度分格式）
    - fields[5]: 经度半球（E/W）
    - fields[6]: 定位质量（0-9）
    - fields[7]: 卫星数量
    - fields[8]: 水平精度因子（HDOP）
    - fields[9]: 海拔高度（米）
    
    Args:
        fields: NMEA 消息字段列表（已分割的逗号分隔值）
        st: FixState 对象，用于存储解析结果
    """
    # $GNGGA,hhmmss.ss,lat,N,lon,E,fix,sats,hdop,alt,M,...
    if len(fields) < 10:
        return
    
    # 解析 UTC 时间
    if fields[1]:
        st.time_hhmmss = fields[1]

    # 解析纬度和经度
    lat = dm_to_deg(fields[2], fields[3])
    lon = dm_to_deg(fields[4], fields[5])
    if lat is not None:
        st.lat_deg = lat
    if lon is not None:
        st.lon_deg = lon

    # 解析定位质量
    if fields[6]:
        try:
            st.fix_quality = int(fields[6])
        except ValueError:
            pass
    
    # 解析卫星数量
    if fields[7]:
        try:
            st.sats = int(fields[7])
        except ValueError:
            pass
    
    # 解析水平精度因子
    if fields[8]:
        try:
            st.hdop = float(fields[8])
        except ValueError:
            pass
    
    # 解析海拔高度
    if fields[9]:
        try:
            st.altitude_m = float(fields[9])
        except ValueError:
            pass


def parse_rmc(fields: list, st: FixState) -> None:
    """
    解析 RMC (Recommended Minimum Specific GPS/Transit Data) NMEA 消息
    
    RMC 消息格式：
    $GNRMC,hhmmss.ss,status,lat,N,lon,E,speed_knots,course,date,mag_var,mag_var_dir*cs
    
    字段说明：
    - fields[0]: 消息类型（如 "GNRMC"）
    - fields[1]: UTC时间（HHMMSS.SS）
    - fields[2]: 状态（A=有效，V=无效）
    - fields[3]: 纬度（度分格式）
    - fields[4]: 纬度半球（N/S）
    - fields[5]: 经度（度分格式）
    - fields[6]: 经度半球（E/W）
    - fields[7]: 速度（节）
    - fields[8]: 航向角（度）
    - fields[9]: 日期（DDMMYY）
    
    Args:
        fields: NMEA 消息字段列表（已分割的逗号分隔值）
        st: FixState 对象，用于存储解析结果
    """
    # $GNRMC,hhmmss.ss,status,lat,N,lon,E,speed_knots,course,date,...
    if len(fields) < 10:
        return
    
    # 解析 UTC 时间
    if fields[1]:
        st.time_hhmmss = fields[1]
    
    # 解析数据有效性状态（A=有效，V=无效）
    if fields[2]:
        st.valid = (fields[2].upper() == "A")

    # 解析纬度和经度
    lat = dm_to_deg(fields[3], fields[4])
    lon = dm_to_deg(fields[5], fields[6])
    if lat is not None:
        st.lat_deg = lat
    if lon is not None:
        st.lon_deg = lon

    # 解析速度（节）
    if fields[7]:
        try:
            st.speed_knots = float(fields[7])
        except ValueError:
            pass
    
    # 解析航向角（度）
    if fields[8]:
        try:
            st.course_deg = float(fields[8])
        except ValueError:
            pass
    
    # 解析日期
    if fields[9]:
        st.date_ddmmyy = fields[9]


def enu_from_latlon(lat: float, lon: float, lat0: float, lon0: float) -> Dict[str, float]:
    """
    将经纬度坐标转换为本地 ENU（东-北-天）坐标系中的米单位坐标
    
    使用等距圆柱投影（Equirectangular Projection）进行局部切平面近似计算。
    该方法适用于小范围区域（通常几十公里内），计算简单快速。
    
    Args:
        lat: 目标点纬度（十进制度数）
        lon: 目标点经度（十进制度数）
        lat0: 原点纬度（十进制度数）
        lon0: 原点经度（十进制度数）
    
    Returns:
        Dict[str, float]: 包含 "east_m"（东向距离，米）和 "north_m"（北向距离，米）的字典
    
    Note:
        - 使用 WGS84 椭球体的平均半径（6378137.0 米）
        - 东向距离：正值表示目标点在原点东侧
        - 北向距离：正值表示目标点在原点北侧
    """
    R = 6378137.0  # WGS84 椭球体平均半径（米）
    phi = math.radians(lat)
    phi0 = math.radians(lat0)
    lam = math.radians(lon)
    lam0 = math.radians(lon0)

    # 计算纬度和经度差（弧度）
    dphi = phi - phi0
    dlam = lam - lam0
    
    # 使用平均纬度计算经度方向的缩放因子
    # 东向距离 = 半径 × 经度差 × 平均纬度的余弦
    east = R * dlam * math.cos((phi + phi0) / 2.0)
    # 北向距离 = 半径 × 纬度差
    north = R * dphi
    return {"east_m": east, "north_m": north}


def fmt(st: FixState, origin: Optional[tuple]) -> str:
    """
    格式化 GPS 定位状态为可读字符串
    
    将 FixState 对象中的信息格式化为易读的字符串格式，包括：
    - 定位质量和卫星信息
    - 经纬度坐标（如果提供了原点，还会显示相对原点的 ENU 坐标）
    - 海拔高度
    - 速度（节、公里/小时、米/秒）
    - 航向角和时间信息
    
    Args:
        st: FixState 对象，包含 GPS 定位状态信息
        origin: 可选的原点坐标元组 (lat0, lon0)，用于计算相对位置
    
    Returns:
        str: 格式化后的字符串，各字段用 " | " 分隔
    """
    parts = []

    # 定位质量和卫星信息
    fixq = st.fix_quality if st.fix_quality is not None else -1
    valid = st.valid if st.valid is not None else False
    parts.append(f"fix_quality={fixq} valid={'A' if valid else 'V'} sats={st.sats} hdop={st.hdop}")

    # 经纬度坐标
    if st.lat_deg is not None and st.lon_deg is not None:
        parts.append(f"lat={st.lat_deg:.6f} lon={st.lon_deg:.6f}")
        # 如果提供了原点，计算并显示相对原点的 ENU 坐标
        if origin is not None:
            en = enu_from_latlon(st.lat_deg, st.lon_deg, origin[0], origin[1])
            parts.append(f"E={en['east_m']:+.2f}m N={en['north_m']:+.2f}m (origin)")
    else:
        parts.append("lat/lon=NA")

    # 海拔高度
    if st.altitude_m is not None:
        parts.append(f"alt={st.altitude_m:.1f}m")
    else:
        parts.append("alt=NA")

    # 速度转换（节 -> 公里/小时 -> 米/秒）
    if st.speed_knots is not None:
        kmh = st.speed_knots * 1.852  # 1节 = 1.852公里/小时
        ms = kmh / 3.6  # 1公里/小时 = 1/3.6米/秒
        parts.append(f"speed={st.speed_knots:.2f}kn {kmh:.2f}km/h {ms:.2f}m/s")
    else:
        parts.append("speed=NA")

    # 航向角和时间信息
    if st.course_deg is not None:
        parts.append(f"course={st.course_deg:.1f}°")
    if st.time_hhmmss:
        parts.append(f"utc={st.time_hhmmss}")
    if st.date_ddmmyy:
        parts.append(f"date={st.date_ddmmyy}")

    return " | ".join(parts)


def main():
    """
    主函数：实时读取 GPS NMEA 数据并解析显示
    
    从串口读取 GPS 设备的 NMEA 消息，解析 GGA 和 RMC 消息，
    并将定位信息格式化为可读字符串输出。支持坐标转换、速度单位转换等功能。
    
    支持的命令行参数：
    - --port: 串口设备路径（默认：/dev/ttyS9）
    - --baud: 波特率（默认：9600，常用值：9600 或 115200）
    - --origin: 原点坐标（纬度 经度），用于计算相对位置的 ENU 坐标
    - --set-origin-on-fix: 自动将第一个有效定位点设置为原点
    - --no-checksum: 跳过 NMEA 校验和验证
    """
    ap = argparse.ArgumentParser(
        description="实时 GPS NMEA 数据读取器，支持坐标转换（十进制度、公里/小时、ENU米）"
    )
    ap.add_argument("--port", default="/dev/ttyS9", 
                    help="串口设备路径，例如 /dev/ttyS9 或 COM3")
    ap.add_argument("--baud", type=int, default=9600, 
                    help="波特率，常用值：9600 或 115200")
    ap.add_argument("--origin", nargs=2, type=float, metavar=("LAT0", "LON0"),
                    help="可选的原点坐标（纬度 经度，十进制度数），用于计算 ENU 相对坐标")
    ap.add_argument("--set-origin-on-fix", action="store_true",
                    help="如果设置，自动将第一个有效定位点（状态为A且fix_quality>0）设置为原点")
    ap.add_argument("--no-checksum", action="store_true", 
                    help="不验证 NMEA 校验和（不推荐，可能导致错误数据）")
    args = ap.parse_args()

    origin = tuple(args.origin) if args.origin else None
    st = FixState()

    # 配置串口参数
    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,  # 8位数据位
        parity=serial.PARITY_NONE,  # 无奇偶校验
        stopbits=serial.STOPBITS_ONE,  # 1位停止位
        timeout=1.0,  # 读取超时时间（秒）
        xonxoff=False,  # 不使用软件流控
        rtscts=False,  # 不使用硬件流控（RTS/CTS）
        dsrdtr=False,  # 不使用硬件流控（DSR/DTR）
    )

    print(f"[OK] 已打开串口 {args.port} @ {args.baud} 波特率")
    print("[INFO] 等待 NMEA 数据... (按 Ctrl+C 退出)")

    last_print = 0.0
    try:
        while True:
            # 从串口读取一行数据
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line.startswith("$"):
                continue
            
            # 验证校验和（如果启用）
            if (not args.no_checksum) and ("*" in line) and (not nmea_checksum_ok(line)):
                continue

            # 分割字段：去除开头的 '$' 和末尾的校验和
            body = line[1:]
            if "*" in body:
                body = body.split("*", 1)[0]
            fields = body.split(",")
            msg = fields[0]

            # 根据消息类型解析
            if msg.endswith("GGA"):
                parse_gga(fields, st)
            elif msg.endswith("RMC"):
                parse_rmc(fields, st)

            # 自动设置原点：当获得第一个有效定位时
            if args.set_origin_on_fix and origin is None:
                if (st.lat_deg is not None and st.lon_deg is not None and
                        (st.fix_quality or 0) > 0 and (st.valid is True)):
                    origin = (st.lat_deg, st.lon_deg)
                    print(f"\n[OK] 原点已设置为 lat0={origin[0]:.6f}, lon0={origin[1]:.6f}\n")

            # 限制输出频率为最大 2 Hz（每 0.5 秒一次）
            now = time.time()
            if now - last_print >= 0.5:
                print(fmt(st, origin))
                last_print = now

    except KeyboardInterrupt:
        print("\n[EXIT] 程序退出")
    finally:
        ser.close()


if __name__ == "__main__":
    main()