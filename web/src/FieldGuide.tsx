import { useState } from 'react'
import { agentAnswer, agentStart, type AgentReply } from './api'

// as GBIF spells them, so the region filter matches its records
const DEPARTMENTS = [
  'Amazonas',
  'Antioquia',
  'Arauca',
  'Atlántico',
  'Bolívar',
  'Boyacá',
  'Caldas',
  'Caquetá',
  'Casanare',
  'Cauca',
  'Cesar',
  'Chocó',
  'Córdoba',
  'Cundinamarca',
  'Distrito Capital de Bogotá',
  'Guainía',
  'Guaviare',
  'Huila',
  'La Guajira',
  'Magdalena',
  'Meta',
  'Nariño',
  'Norte de Santander',
  'Putumayo',
  'Quindío',
  'Risaralda',
  'San Andrés Providencia y Santa Catalina',
  'Santander',
  'Sucre',
  'Tolima',
  'Valle del Cauca',
  'Vaupés',
  'Vichada',
]
const RED_LIST: Record<string, string> = {
  LC: 'Least concern',
  NT: 'Near threatened',
  VU: 'Vulnerable',
  EN: 'Endangered',
  CR: 'Critically endangered',
}

export function FieldGuide({ image }: { image: Blob }) {
  const [region, setRegion] = useState('')
  const [reply, setReply] = useState<AgentReply>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()

  async function step(call: () => Promise<AgentReply>) {
    setBusy(true)
    setError(undefined)
    try {
      setReply(await call())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const card = reply?.card
  return (
    <section className="guide">
      <header className="guide-head">
        <h2>Field guide</h2>
        <p>
          When BioCLIP stops before the species, a local language model checks which candidates live in your region and
          asks you about what you saw. It only chooses among BioCLIP's candidates.
        </p>
      </header>

      <div className="guide-body">
        <div className="guide-start">
          <label>
            Where was the photo taken?
            <select value={region} onChange={(e) => setRegion(e.target.value)}>
              <option value="">I don't know</option>
              {DEPARTMENTS.map((d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
          </label>
          <button className="button" disabled={busy} onClick={() => step(() => agentStart(image, region))}>
            {reply ? 'Start again' : 'Ask the field guide'}
          </button>
        </div>

        {reply && (
          <ol className="trace">
            {reply.trace.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ol>
        )}

        {reply?.status === 'question' && (
          <div className="question">
            <p>{reply.question}</p>
            {['yes', 'no', "can't tell"].map((a) => (
              <button key={a} disabled={busy} onClick={() => step(() => agentAnswer(reply.thread_id, a))}>
                {a}
              </button>
            ))}
          </div>
        )}

        {busy && <p className="status">Thinking…</p>}
        {error && <p className="status error">{error}</p>}

        {card && (
          <article className="card">
            <p className="result-rank">{card.rank}</p>
            <h3 className={card.rank === 'species' ? 'sci' : ''}>{card.name}</h3>
            {card.common && <p className="card-common">{card.common}</p>}
            <p>{card.text}</p>
            {card.red_list && (
              <p className={card.hide_location ? 'badge threatened' : 'badge'}>
                IUCN: {RED_LIST[card.red_list] ?? card.red_list}
                {card.hide_location && ' · exact location hidden to protect it'}
              </p>
            )}
            <p className="sources">
              Sources:{' '}
              {card.sources.map((s, i) => (
                <a key={s} href={s} target="_blank" rel="noreferrer">
                  {new URL(s).hostname.replace('www.', '')}
                  {i < card.sources.length - 1 ? ', ' : ''}
                </a>
              ))}
            </p>
          </article>
        )}
      </div>
    </section>
  )
}
