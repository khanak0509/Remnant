import { useEffect, useRef, useState } from 'react'
import Chart from './Chart'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const LABELS = { ok: 'OK', watch: 'Watch', maintain: 'Maintain' }

export default function App() {
  const [engines, setEngines] = useState({})
  const [history, setHistory] = useState({})
  const [cycle, setCycle] = useState(0)
  const [selected, setSelected] = useState(null)
  const [running, setRunning] = useState(false)
  const [speed, setSpeed] = useState(0.3)
  const source = useRef(null)

  function start() {
    stop()
    setEngines({})
    setHistory({})
    setCycle(0)
    const es = new EventSource(`${API}/fleet/stream?speed=${speed}`)
    es.onmessage = (msg) => {
      const data = JSON.parse(msg.data)
      setCycle(data.cycle)
      setEngines((prev) => {
        const next = { ...prev }
        data.engines.forEach((e) => (next[e.unit] = e))
        return next
      })
      setHistory((prev) => {
        const next = { ...prev }
        data.engines.forEach((e) => (next[e.unit] = [...(next[e.unit] || []), e]))
        return next
      })
    }
    es.addEventListener('done', () => stop())
    es.onerror = () => stop()
    source.current = es
    setRunning(true)
  }

  function stop() {
    if (source.current) source.current.close()
    source.current = null
    setRunning(false)
  }

  useEffect(() => stop, [])

  const list = Object.values(engines).sort((a, b) => a.unit - b.unit)
  const flagged = list.filter((e) => e.flag).length
  const watch = list.filter((e) => e.status === 'watch').length
  const sel = selected ?? (list.length ? list[0].unit : null)

  return (
    <div className="page">
      <header>
        <div>
          <h1>Remnant</h1>
          <p className="sub">Turbofan fleet replay, FD001 test engines</p>
        </div>
        <div className="controls">
          <label>
            Delay
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} disabled={running}>
              <option value={0.05}>0.05s</option>
              <option value={0.1}>0.1s</option>
              <option value={0.3}>0.3s</option>
              <option value={1}>1s</option>
            </select>
          </label>
          {running ? <button onClick={stop}>Stop</button> : <button onClick={start}>Start replay</button>}
        </div>
      </header>

      <section className="stats">
        <div><span>{cycle}</span>cycle</div>
        <div><span>{list.length}</span>engines seen</div>
        <div><span>{watch}</span>watch</div>
        <div><span>{flagged}</span>maintain</div>
      </section>

      <section className="chart-card">
        <h2>Engine {sel ?? '-'}: predicted vs true RUL</h2>
        <Chart points={sel ? history[sel] || [] : []} />
      </section>

      <section className="grid">
        {list.map((e) => (
          <button
            key={e.unit}
            className={`cell ${e.status} ${e.unit === sel ? 'sel' : ''}`}
            onClick={() => setSelected(e.unit)}
            title={`P(RUL < 20) = ${e.p_fail}`}
          >
            <div className="unit">#{e.unit}</div>
            <div className="rul">{Math.round(e.rul)}</div>
            <div className="int">{Math.round(e.lower)} to {Math.round(e.upper)}</div>
            <div className="tag"><i />{LABELS[e.status]}</div>
          </button>
        ))}
        {!list.length && <p className="empty">Press Start replay to stream the test fleet cycle by cycle.</p>}
      </section>
    </div>
  )
}
