import { displayName, walk, type TreeNode } from './tree'

interface Props {
  title: string
  subtitle: string
  nodes: TreeNode[]
  threshold: number
  commonNames: Map<string, string>
  variant: 'clip' | 'bioclip'
}

const pct = (p: number) => `${(p * 100).toFixed(1)}%`

export function ModelPanel({ title, subtitle, nodes, threshold, commonNames, variant }: Props) {
  const steps = walk(nodes, threshold)
  // ranks with a single option (kingdom, phylum and class when every candidate is a bird) say nothing
  const firstChoice = steps.findIndex((s) => s.options.length > 1)
  const fixed = steps.slice(0, firstChoice)
  const shown = steps.slice(firstChoice)

  const answer = steps.filter((s) => s.trusted).at(-1)!
  const best = answer.options[0]
  const stoppedEarly = answer.rank !== 'species'
  const common = best.rank === 'species' ? commonNames.get(displayName(best)) : undefined

  return (
    <section className={`panel ${variant}`}>
      <header>
        <h2>{title}</h2>
        <p className="subtitle">{subtitle}</p>
      </header>

      <div className="answer">
        <span className="answer-rank">{answer.rank}</span>
        <span className={best.rank === 'species' ? 'answer-name italic' : 'answer-name'}>{displayName(best)}</span>
        {common && <span className="answer-common">{common}</span>}
        <span className="answer-p">{pct(best.p)}</span>
        {stoppedEarly && <p className="hedge">Not sure below this rank, so it stops here.</p>}
      </div>

      <p className="breadcrumb">{fixed.map((s) => displayName(s.options[0])).join(' › ')}</p>

      <ol className="ladder">
        {shown.map((step) => (
          <li key={step.rank} className={step.trusted ? 'trusted' : 'untrusted'}>
            <span className="rank">{step.rank}</span>
            <ul>
              {step.options
                .filter((o, i) => i === 0 || o.p >= 0.005)
                .slice(0, 3)
                .map((o, i) => (
                  <li key={o.path.join('/')} className={i === 0 ? 'option top' : 'option'}>
                    <span className={o.rank === 'species' ? 'name italic' : 'name'}>{displayName(o)}</span>
                    <span className="bar">
                      <span className="fill" style={{ width: pct(o.p) }} />
                      {i === 0 && <span className="cut" style={{ left: pct(threshold) }} />}
                    </span>
                    <span className="p">{pct(o.p)}</span>
                  </li>
                ))}
            </ul>
          </li>
        ))}
      </ol>
    </section>
  )
}
