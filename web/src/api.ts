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

export function health(): Promise<{ threshold: number; agent: boolean }> {
  return request('/health')
}

export interface FieldCard {
  name: string
  rank: string
  common: string
  text: string
  red_list: string | null
  hide_location: boolean
  sources: string[]
}

export interface AgentReply {
  thread_id: string
  status: 'question' | 'done'
  trace: string[]
  question?: string
  card?: FieldCard
}

export function agentStart(image: Blob, region: string): Promise<AgentReply> {
  const form = new FormData()
  form.append('file', image)
  if (region) form.append('region', region)
  return request('/agent/start', { method: 'POST', body: form })
}

export function agentAnswer(threadId: string, answer: string): Promise<AgentReply> {
  return request('/agent/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId, answer }),
  })
}
