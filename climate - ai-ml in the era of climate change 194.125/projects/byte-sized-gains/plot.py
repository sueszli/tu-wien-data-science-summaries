import csv
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "results.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    marks = []
    for row in rows:
        x = 60 + 460 * float(row["model_footprint_mb"]) / 600
        accuracy = float(row["top1_accuracy_pct"])
        y = 220 - 168 * accuracy / 6
        label = escape(row["configuration"])
        shift = 10 if row["configuration"] != "8-bit" else 16
        marks.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2166ac"/>'
            f'<text x="{x+8:.1f}" y="{y+shift:.1f}" font-size="12">{label}</text>'
        )
    ticks = []
    for value in (0, 2, 4, 6):
        y = 220 - 168 * value / 6
        ticks.append(f'<path d="M56 {y:.0f}h4" stroke="#747474"/><text x="48" y="{y+4:.0f}" text-anchor="end">{value}</text>')
    for value in (0, 150, 300, 450, 600):
        x = 60 + 460 * value / 600
        ticks.append(f'<path d="M{x:.0f} 220v4" stroke="#747474"/><text x="{x:.0f}" y="239" text-anchor="middle">{value}</text>')
    output = f'''<svg xmlns="http://www.w3.org/2000/svg" width="600" height="280" viewBox="0 0 600 280" role="img" aria-labelledby="title">
  <title id="title">Median top-1 accuracy against model memory footprint by quantization level</title>
  <rect width="600" height="280" fill="white"/>
  <g fill="#222" font-family="sans-serif" font-size="12">
    <text x="60" y="28" font-size="16" font-weight="bold">SmolLM-135M on LAMBADA</text>
    <text x="60" y="45">Top-1 accuracy (%)</text>
    <g stroke="#747474" fill="none"><path d="M60 52v168h460"/></g>
    {''.join(ticks)}
    <text x="290" y="262" text-anchor="middle">Reported model footprint (MB)</text>
    {''.join(marks)}
  </g>
</svg>
'''
    (ROOT / "accuracy-vs-memory.svg").write_text(output)


if __name__ == "__main__":
    main()
