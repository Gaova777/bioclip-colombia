import { useEffect, useState } from 'react'
import { candidates, classify } from './api'
import { ModelPanel } from './ModelPanel'
import type { Classification } from './tree'

const SAMPLES = [
  { file: 'amazilia_tzacatl_yarumal.jpg', label: 'Rufous-tailed hummingbird', credit: 'Esteban Villa Restrepo, CC BY' },
  { file: 'saucerottia_saucerottei_anori.jpg', label: 'Steely-vented hummingbird', credit: 'Christoph Moning, CC BY' },
  { file: 'uranomitra_franciae_cali.jpg', label: 'Andean emerald', credit: 'Anthony Batista, CC BY' },
]
const sampleUrl = (file: string) => `${import.meta.env.BASE_URL}samples/${file}`

export default function App() {
  const [preview, setPreview] = useState<string>()
  const [result, setResult] = useState<Classification>()
  const [threshold, setThreshold] = useState(0.8)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>()
  const [commonNames, setCommonNames] = useState(new Map<string, string>())

  useEffect(() => {
    candidates()
      .then((list) => setCommonNames(new Map(list.map((c) => [c.lineage.slice(-2).join(' '), c.common_name]))))
      .catch(() => setError('The API is not answering. Is the backend running?'))
  }, [])

  async function run(image: Blob, url: string) {
    setPreview(url)
    setResult(undefined)
    setError(undefined)
    setLoading(true)
    try {
      setResult(await classify(image))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  async function runSample(file: string) {
    const blob = await (await fetch(sampleUrl(file))).blob()
    run(blob, sampleUrl(file))
  }

  return (
    <main>
      <header className="hero">
        <h1>bioclip-colombia</h1>
        <p>
          A species classifier that knows when to say <em>"I'm not sure"</em>. Upload a bird photo: both models
          walk down the tree of life and stop at the first rank where they are less confident than the threshold.
        </p>
      </header>

      <section className="input">
        <div className="photo">
          {preview ? <img src={preview} alt="photo to classify" /> : <div className="placeholder">No photo yet</div>}
        </div>
        <div className="controls">
          <label className="upload">
            Upload a photo
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) run(file, URL.createObjectURL(file))
              }}
            />
          </label>
          <p className="or">or try one of these</p>
          <div className="samples">
            {SAMPLES.map((s) => (
              <button key={s.file} onClick={() => runSample(s.file)} title={`Photo: ${s.credit}`}>
                <img src={sampleUrl(s.file)} alt={s.label} />
                <span>{s.label}</span>
              </button>
            ))}
          </div>

          <label className="threshold">
            <span>
              Threshold <strong>{Math.round(threshold * 100)}%</strong>
            </span>
            <input
              type="range"
              min={0.05}
              max={0.99}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <small>Higher means the model answers less deep but is wrong less often.</small>
          </label>
        </div>
      </section>

      {loading && <p className="status">Classifying…</p>}
      {error && <p className="status error">{error}</p>}

      {result && (
        <div className="panels">
          <ModelPanel
            title="CLIP"
            subtitle="General-purpose model, prompted with common names"
            nodes={result.clip}
            threshold={threshold}
            commonNames={commonNames}
            variant="clip"
          />
          <ModelPanel
            title="BioCLIP"
            subtitle="Trained on the tree of life, prompted with the full taxonomy"
            nodes={result.bioclip}
            threshold={threshold}
            commonNames={commonNames}
            variant="bioclip"
          />
        </div>
      )}

      <footer>
        Candidates: the 200 most observed bird species in Colombia on iNaturalist. Models: BioCLIP (Stevens et al.,
        CVPR 2024) and OpenAI CLIP ViT-B/16 through OpenCLIP. Sample photos from iNaturalist:{' '}
        {SAMPLES.map((s) => s.credit).join('; ')}.{' '}
        <a href="https://github.com/Gaova777/bioclip-colombia">Code on GitHub</a>
      </footer>
    </main>
  )
}
