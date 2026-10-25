import { useState } from 'react'

const W = 720
const H = 260
const M = { top: 12, right: 16, bottom: 32, left: 40 }

export default function Chart({ points }) {
  const [hover, setHover] = useState(null)
  if (!points.length) return <div className="chart-empty">No data yet</div>

  const maxX = Math.max(points[points.length - 1].cycle, 10)
  const maxY = Math.max(140, ...points.map((p) => Math.max(p.upper, p.true_rul))) * 1.05
  const x = (c) => M.left + ((c - points[0].cycle) / Math.max(maxX - points[0].cycle, 1)) * (W - M.left - M.right)
  const y = (v) => H - M.bottom - (v / maxY) * (H - M.top - M.bottom)
  const line = (key) => points.map((p, i) => `${i ? 'L' : 'M'}${x(p.cycle)},${y(p[key])}`).join('')
  const band = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.cycle)},${y(p.upper)}`).join('') +
    [...points].reverse().map((p) => `L${x(p.cycle)},${y(p.lower)}`).join('') + 'Z'
  const ticks = [0, 25, 50, 75, 100, 125, 150, 175, 200].filter((t) => t < maxY)

  function move(ev) {
    const r = ev.currentTarget.getBoundingClientRect()
    const px = ((ev.clientX - r.left) / r.width) * W
    let best = points[0]
    points.forEach((p) => { if (Math.abs(x(p.cycle) - px) < Math.abs(x(best.cycle) - px)) best = p })
    setHover(best)
  }

  return (
    <div className="chart">
      <div className="legend">
        <span><i className="sw pred" />Predicted RUL</span>
        <span><i className="sw band" />90% interval</span>
        <span><i className="sw true" />True RUL</span>
        <span><i className="sw hz" />Horizon (20)</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={move} onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
            <text className="axis" x={M.left - 6} y={y(t) + 4} textAnchor="end">{t}</text>
          </g>
        ))}
        <text className="axis" x={M.left} y={H - 8}>cycle {points[0].cycle}</text>
        <text className="axis" x={W - M.right} y={H - 8} textAnchor="end">cycle {maxX}</text>
        <line className="hz" x1={M.left} x2={W - M.right} y1={y(20)} y2={y(20)} />
        <path className="band" d={band} />
        <path className="true" d={line('true_rul')} />
        <path className="pred" d={line('rul')} />
        {hover && (
          <g>
            <line className="cross" x1={x(hover.cycle)} x2={x(hover.cycle)} y1={M.top} y2={H - M.bottom} />
            <circle className="dot-pred" cx={x(hover.cycle)} cy={y(hover.rul)} r="4" />
            <circle className="dot-true" cx={x(hover.cycle)} cy={y(hover.true_rul)} r="4" />
          </g>
        )}
      </svg>
      {hover && (
        <div className="tip" style={{ left: `${(x(hover.cycle) / W) * 100}%` }}>
          <b>Cycle {hover.cycle}</b>
          <div>Predicted {hover.rul} ({hover.lower} to {hover.upper})</div>
          <div>True {hover.true_rul}</div>
          <div>P(RUL &lt; 20) {hover.p_fail}</div>
        </div>
      )}
    </div>
  )
}
