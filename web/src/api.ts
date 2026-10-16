import type { Classification } from './tree'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export interface Candidate {
  lineage: string[]
  common_name: string
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_URL}${path}`, init)
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    throw new Error(body.detail ?? `${resp.status} ${resp.statusText}`)
  }
  return resp.json()
}

export function classify(image: Blob): Promise<Classification> {
  const form = new FormData()
  form.append('file', image)
  return request('/classify', { method: 'POST', body: form })
}

export function candidates(): Promise<Candidate[]> {
  return request('/candidates')
}

export function health(): Promise<{ threshold: number }> {
  return request('/health')
}
