import { displayName, walk, type TreeNode } from './tree'

interface Props {
  model: string
  about: string
  nodes: TreeNode[]
  threshold: number
  commonNames: Map<string, string>
  variant: 'clip' | 'bioclip'
}

const pct = (p: number) => `${(p * 100).toFixed(1)}%`

export function ModelPanel({ model, about, nodes, threshold, commonNames, variant }: Props) {
  const steps = walk(nodes, threshold)
  // ranks with a single option (kingdom, phylum and class when every candidate is a bird) say nothing
  const firstChoice = steps.findIndex((s) => s.options.length > 1)
  const root = steps[firstChoice - 1].options[0]
  const shown = steps.slice(firstChoice)

  const answer = steps.filter((s) => s.trusted).at(-1)!
  const best = answer.options[0]
  const common = best.rank === 'species' ? commonNames.get(displayName(best)) : undefined

  return (
    <article className={`result ${variant}`}>
      <header className="result-head">
        <p className="result-model">
          {model} <span>{about}</span>
        </p>
        <p className="result-rank">{answer.rank}</p>
        <p className={best.rank === 'species' ? 'result-name sci' : 'result-name'}>{displayName(best)}</p>
        {common && <p className="result-common">{common}</p>}
        <p className="result-p">{pct(best.p)}</p>
      </header>
      {answer.rank !== 'species' && <p className="hedge">Not sure past this rank, so it stops here.</p>}

      <div className="clade">
        <p className="clade-root">{displayName(root)}</p>
        <ol>
          {shown.map((step, depth) => {
            const [top, ...rest] = step.options
            const alternatives = rest.filter((o) => o.p >= 0.005).slice(0, 2)
            return (
              <li key={step.rank} className={step.trusted ? 'kept' : 'dropped'} style={{ marginLeft: depth * 16 }}>
                <span className="clade-rank">{step.rank}</span>
                <span className={top.rank === 'species' ? 'clade-name sci' : 'clade-name'}>{displayName(top)}</span>
                <span className="meter">
                  <span className="meter-fill" style={{ width: pct(top.p) }} />
                  <span className="meter-cut" style={{ left: pct(threshold) }} />
                </span>
                <span className="num">{pct(top.p)}</span>
                {alternatives.length > 0 && (
                  <span className="alternatives">
                    or {alternatives.map((o) => `${displayName(o)} ${pct(o.p)}`).join(', ')}
                  </span>
                )}
              </li>
            )
          })}
        </ol>
      </div>
    </article>
  )
}
