import { useEffect, useState } from 'react'
import { candidates, classify, health } from './api'
import { FieldGuide } from './FieldGuide'
import { ModelPanel } from './ModelPanel'
import type { Classification } from './tree'

const SAMPLES = [
  {
    file: 'amazilia_tzacatl_yarumal.jpg',
    label: 'Rufous-tailed hummingbird',
    place: 'Yarumal',
    credit: 'Esteban Villa Restrepo, CC BY',
  },
  {
    file: 'saucerottia_saucerottei_anori.jpg',
    label: 'Steely-vented hummingbird',
    place: 'Anorí',
    credit: 'Christoph Moning, CC BY',
  },
  {
    file: 'uranomitra_franciae_cali.jpg',
    label: 'Andean emerald',
    place: 'Cali',
    credit: 'Anthony Batista, CC BY',
  },
]
const sampleUrl = (file: string) => `${import.meta.env.BASE_URL}samples/${file}`

export default function App() {
  const [photo, setPhoto] = useState(sampleUrl(SAMPLES[0].file))
  const [result, setResult] = useState<Classification>()
  const [image, setImage] = useState<Blob>()
  const [agent, setAgent] = useState(false)
  const [threshold, setThreshold] = useState(0.8)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>()
  const [dragging, setDragging] = useState(false)
  const [commonNames, setCommonNames] = useState(new Map<string, string>())

  useEffect(() => {
    // the default threshold is the one picked on the validation set (scripts/calibrate.py)
    health()
      .then((h) => {
        setThreshold(h.threshold)
        setAgent(h.agent)
      })
      .catch(() => {})
    candidates()
      .then((list) => setCommonNames(new Map(list.map((c) => [c.lineage.slice(-2).join(' '), c.common_name]))))
      .catch(() => setError('The API is not answering. Is the backend running?'))
  }, [])

  async function run(image: Blob, url: string) {
    setPhoto(url)
    setImage(image)
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

  function runFile(file?: File) {
    if (file) run(file, URL.createObjectURL(file))
  }

  return (
    <>
      <nav className="topbar">
        <span className="brand">bioclip-colombia</span>
        <a href="https://github.com/Gaova777/bioclip-colombia">GitHub</a>
      </nav>
      <header className="cover" style={{ backgroundImage: `url(${photo})` }}>
        <div className="cover-text">
          <p className="eyebrow">Aves de Colombia · 200 species</p>
          <h1>Which bird is this, and how sure are we?</h1>
          <p>
            BioCLIP and CLIP walk down the tree of life, order, family, genus, species, and stop at the first rank where
            they are less sure than the threshold. A genus you can trust beats a species that's a guess.
          </p>
        </div>
      </header>

      <main>
        <section className="field">
          <label
            className={dragging ? 'drop dragging' : 'drop'}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              runFile(e.dataTransfer.files[0])
            }}
          >
            <svg className="drop-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M9 4 7.2 6H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-3.2L15 4H9zm3 4.5a4.5 4.5 0 1 1 0 9 4.5 4.5 0 0 1 0-9zm0 2a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z" />
            </svg>
            <span className="drop-title">Drop a bird photo here</span>
            <span className="button">Choose a photo</span>
            <input type="file" accept="image/*" onChange={(e) => runFile(e.target.files?.[0])} />
          </label>

          <div className="samples">
            <p className="eyebrow">Or try one from iNaturalist</p>
            <div className="tiles">
              {SAMPLES.map((s) => (
                <button key={s.file} onClick={() => runSample(s.file)}>
                  <img src={sampleUrl(s.file)} alt={s.label} />
                  <span>
                    {s.label}
                    <small>{s.place}</small>
                  </span>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="threshold">
          <label htmlFor="threshold">
            Threshold <span className="num">{Math.round(threshold * 100)}%</span>
          </label>
          <input
            id="threshold"
            type="range"
            min={0.05}
            max={0.99}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
          <p>Move it up and the answers get shorter but are wrong less often.</p>
        </section>

        {loading && <p className="status">Looking at the photo…</p>}
        {error && <p className="status error">{error}</p>}
        {!result && !loading && !error && <p className="status">Pick a photo to see what each model says.</p>}

        {result && (
          <section className="results">
            <ModelPanel
              model="BioCLIP"
              about="trained on the tree of life"
              nodes={result.bioclip}
              threshold={threshold}
              commonNames={commonNames}
              variant="bioclip"
            />
            <ModelPanel
              model="CLIP"
              about="general purpose"
              nodes={result.clip}
              threshold={threshold}
              commonNames={commonNames}
              variant="clip"
            />
          </section>
        )}

        {/* key: a new photo starts a new conversation */}
        {result && agent && image && <FieldGuide key={photo} image={image} />}
      </main>

      <footer>
        <p>
          Candidates: the 200 most observed bird species in Colombia on iNaturalist. Models: BioCLIP (Stevens et al.,
          CVPR 2024) and OpenAI CLIP ViT-B/16, through OpenCLIP, zero-shot.
        </p>
        <p>Photos from iNaturalist: {SAMPLES.map((s) => s.credit).join(' · ')}.</p>
        <p>
          <a href="https://github.com/Gaova777/bioclip-colombia">Code on GitHub</a>
        </p>
      </footer>
    </>
  )
}
