<template>
  <div class="viewer-root">
    <header class="viewer-header">
      <div class="title">
        <h1>WebRTC 观看端</h1>
        <p>连接服务器端渲染后的视频流</p>
      </div>
      <router-link class="back-link" to="/">返回编辑器</router-link>
    </header>

    <section class="panel">
      <div class="field">
        <label>信令 WebSocket</label>
        <input v-model="signalUrl" type="text" placeholder="ws://server:port/ws" />
      </div>
      <div class="field">
        <label>设备 ID（可选）</label>
        <input v-model="deviceId" type="text" placeholder="robot-001" />
      </div>
      <div class="field">
        <label>ICE Servers（可选，逗号分隔）</label>
        <input v-model="iceServersInput" type="text" placeholder="stun:stun.l.google.com:19302" />
      </div>
      <div class="field checkbox">
        <label>
          <input v-model="clientOffers" type="checkbox" />
          浏览器主动发起 Offer
        </label>
      </div>
      <div class="actions">
        <button class="primary" :disabled="isConnecting || isConnected" @click="connect">
          {{ isConnecting ? '连接中...' : '连接' }}
        </button>
        <button class="ghost" :disabled="!isConnected && !isConnecting" @click="disconnect">断开</button>
      </div>
      <div class="status">
        <span :class="['dot', statusClass]"></span>
        <span>{{ statusText }}</span>
      </div>
    </section>

    <section class="stage">
      <video ref="videoRef" autoplay playsinline muted></video>
      <div class="hint" v-if="!isConnected">未连接，点击“连接”开始</div>
    </section>

    <section class="panel logs">
      <div class="log-header">
        <span>日志</span>
        <button class="ghost" @click="clearLogs">清空</button>
      </div>
      <div class="log-list">
        <div v-for="(item, idx) in logs" :key="idx" class="log-item">
          <span class="time">{{ item.time }}</span>
          <span class="msg">{{ item.msg }}</span>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref, onUnmounted, computed } from 'vue'

const signalUrl = ref('')
const deviceId = ref('')
const iceServersInput = ref('')
const clientOffers = ref(true)

const isConnected = ref(false)
const isConnecting = ref(false)
const logs = ref([])
const videoRef = ref(null)

let ws = null
let pc = null

const statusText = computed(() => {
  if (isConnecting.value) return '连接中'
  return isConnected.value ? '已连接' : '未连接'
})

const statusClass = computed(() => {
  if (isConnecting.value) return 'pending'
  return isConnected.value ? 'ok' : 'idle'
})

function log(msg) {
  const now = new Date()
  const time = now.toLocaleTimeString('zh-CN', { hour12: false })
  logs.value.unshift({ time, msg })
  if (logs.value.length > 200) logs.value.length = 200
}

function clearLogs() {
  logs.value = []
}

function parseIceServers() {
  const urls = iceServersInput.value
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
  if (urls.length === 0) return undefined
  return [{ urls }]
}

async function connect() {
  if (!signalUrl.value) {
    log('请填写信令 WebSocket 地址')
    return
  }
  await disconnect()
  isConnecting.value = true

  const url = deviceId.value
    ? `${signalUrl.value}${signalUrl.value.includes('?') ? '&' : '?'}deviceId=${encodeURIComponent(deviceId.value)}`
    : signalUrl.value

  ws = new WebSocket(url)

  ws.onopen = async () => {
    log(`信令已连接: ${url}`)
    pc = new RTCPeerConnection({ iceServers: parseIceServers() })

    pc.ontrack = (event) => {
      const [stream] = event.streams
      if (videoRef.value && stream) {
        videoRef.value.srcObject = stream
      }
    }

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        ws.send(JSON.stringify({
          type: 'candidate',
          candidate: event.candidate.candidate,
          sdpMid: event.candidate.sdpMid,
          sdpMLineIndex: event.candidate.sdpMLineIndex
        }))
      }
    }

    pc.onconnectionstatechange = () => {
      log(`连接状态: ${pc.connectionState}`)
      isConnected.value = pc.connectionState === 'connected'
      if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        isConnected.value = false
      }
    }

    pc.addTransceiver('video', { direction: 'recvonly' })

    if (clientOffers.value) {
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      ws.send(JSON.stringify({ type: 'offer', sdp: pc.localDescription.sdp }))
      log('已发送 offer')
    }

    isConnecting.value = false
  }

  ws.onmessage = async (event) => {
    try {
      const msg = JSON.parse(event.data)
      if (msg.type === 'answer' && msg.sdp) {
        await pc.setRemoteDescription({ type: 'answer', sdp: msg.sdp })
        log('收到 answer')
      } else if (msg.type === 'offer' && msg.sdp) {
        await pc.setRemoteDescription({ type: 'offer', sdp: msg.sdp })
        const answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        ws.send(JSON.stringify({ type: 'answer', sdp: pc.localDescription.sdp }))
        log('收到 offer 并回复 answer')
      } else if (msg.type === 'candidate' && msg.candidate) {
        await pc.addIceCandidate({
          candidate: msg.candidate,
          sdpMid: msg.sdpMid,
          sdpMLineIndex: msg.sdpMLineIndex
        })
      } else if (msg.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }))
      }
    } catch (err) {
      log(`信令消息解析失败: ${err}`)
    }
  }

  ws.onclose = () => {
    log('信令连接已关闭')
    isConnected.value = false
    isConnecting.value = false
  }

  ws.onerror = () => {
    log('信令连接错误')
    isConnected.value = false
    isConnecting.value = false
  }
}

async function disconnect() {
  if (pc) {
    pc.getSenders().forEach((sender) => {
      try { sender.track && sender.track.stop() } catch { /* noop */ }
    })
    pc.close()
    pc = null
  }
  if (ws) {
    ws.close()
    ws = null
  }
  if (videoRef.value) {
    videoRef.value.srcObject = null
  }
  isConnected.value = false
  isConnecting.value = false
}

onUnmounted(() => {
  disconnect()
})
</script>

<style scoped>
.viewer-root {
  min-height: 100vh;
  background: radial-gradient(circle at 20% 20%, #20262d, #0f1114 60%);
  color: #f1f1f1;
  padding: 24px;
  display: grid;
  gap: 18px;
}

.viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.viewer-header h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: 0.5px;
}

.viewer-header p {
  margin: 6px 0 0;
  color: #9aa4b2;
}

.back-link {
  color: #7de3ff;
  text-decoration: none;
  font-weight: 600;
}

.panel {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  padding: 16px;
  display: grid;
  gap: 12px;
}

.field {
  display: grid;
  gap: 6px;
}

.field label {
  font-size: 12px;
  color: #8f9aa8;
}

.field input {
  background: #0f1419;
  color: #f1f1f1;
  border: 1px solid #27303a;
  border-radius: 8px;
  padding: 10px 12px;
}

.field.checkbox {
  align-items: center;
}

.actions {
  display: flex;
  gap: 10px;
}

button {
  border: none;
  padding: 10px 16px;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
}

button.primary {
  background: #2dd4ff;
  color: #0b0f12;
}

button.ghost {
  background: transparent;
  color: #c2cbd6;
  border: 1px solid #2b3742;
}

.status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #c2cbd6;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  background: #2b3742;
}

.dot.ok {
  background: #3ddc84;
}

.dot.pending {
  background: #fbbf24;
}

.stage {
  position: relative;
  background: #0b0f12;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  min-height: 360px;
  display: grid;
  place-items: center;
  overflow: hidden;
}

.stage video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}

.hint {
  position: absolute;
  color: #9aa4b2;
  font-size: 14px;
}

.logs {
  max-height: 220px;
  overflow: hidden;
}

.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.log-list {
  max-height: 160px;
  overflow: auto;
  display: grid;
  gap: 6px;
  font-size: 12px;
}

.log-item {
  display: grid;
  grid-template-columns: 80px 1fr;
  gap: 8px;
  color: #c8d1dc;
}

.log-item .time {
  color: #7b8794;
}

@media (max-width: 960px) {
  .viewer-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
}
</style>
