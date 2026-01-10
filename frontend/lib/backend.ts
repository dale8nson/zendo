export const BACKEND_HTTP = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8080'

export function backendUrl(path: string): string {
  const base = BACKEND_HTTP.endsWith('/') ? BACKEND_HTTP.slice(0, -1) : BACKEND_HTTP
  return `${base}${path.startsWith('/') ? '' : '/'}${path}`
}

export function backendWsUrl(path: string): string {
  const http = new URL(backendUrl(path))
  const isSecure = http.protocol === 'https:'
  http.protocol = isSecure ? 'wss:' : 'ws:'
  return http.toString()
}

export async function rustUpscale(imageBase64: string): Promise<string> {
  const res = await fetch(backendUrl('/api/upscale'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: imageBase64 }),
    cache: 'no-cache',
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Upscale failed: ${res.status} ${res.statusText} ${text}`)
  }
  const data = await res.json()
  if (!data.image_base64) throw new Error('Upscale response missing image_base64')
  return data.image_base64 as string
}

export function sdxlGenerateViaWs(params: any, onProgress?: (step: number, total: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(backendWsUrl('/ws/sdxl'))
    ws.onerror = (evt) => {
      console.error('WS /ws/sdxl error', evt)
      try { ws.close() } catch {}
      reject(new Error('WebSocket error'))
    }
    ws.onclose = (evt) => {
      console.log('WS /ws/sdxl closed', { code: evt.code, reason: evt.reason })
    }
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'generate', params }))
    }
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        if (msg.type === 'ready') {
          console.log('WS /ws/sdxl ready')
        } else if (msg.type === 'progress') {
          onProgress?.(msg.step, msg.total)
        } else if (msg.type === 'result') {
          try { ws.close() } catch {}
          resolve(msg.image_base64)
        } else if (msg.type === 'error') {
          try { ws.close() } catch {}
          reject(new Error(msg.message || 'generation error'))
        }
      } catch (e) {
        // ignore non-JSON
      }
    }
  })
}
