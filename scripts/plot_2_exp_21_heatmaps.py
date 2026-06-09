from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


CELL = 22
LEFT = 70
TOP = 32
RIGHT = 18
BOTTOM = 36


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = _clip01(t)
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b, strict=True))


def _sequential(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return "#f7f7f7"
    t = _clip01((value - vmin) / (vmax - vmin))
    stops = [
        (0.00, (247, 251, 255)),
        (0.35, (198, 219, 239)),
        (0.70, (107, 174, 214)),
        (1.00, (8, 81, 156)),
    ]
    for (left_t, left_c), (right_t, right_c) in zip(stops, stops[1:], strict=True):
        if t <= right_t:
            return _hex(_lerp(left_c, right_c, (t - left_t) / max(right_t - left_t, 1e-9)))
    return _hex(stops[-1][1])


def _diverging(value: float, vmax_abs: float) -> str:
    if vmax_abs <= 0:
        return "#f7f7f7"
    t = _clip01(value / vmax_abs)
    if t >= 0:
        return _hex(_lerp((247, 247, 247), (178, 24, 43), t))
    return _hex(_lerp((247, 247, 247), (33, 102, 172), -t))


def _heatmap_svg(frame: pd.DataFrame, value_col: str, title: str, signed: bool) -> str:
    days = sorted(frame["day"].astype(int).unique())
    hours = sorted(frame["hour"].astype(int).unique())
    width = LEFT + CELL * len(hours) + RIGHT
    height = TOP + CELL * len(days) + BOTTOM
    values = frame[value_col].astype(float)
    finite = values.dropna()
    if signed:
        vmax_abs = float(finite.abs().quantile(0.98)) if len(finite) else 0.0
        color = lambda v: _diverging(float(v), vmax_abs)
        legend_left = -vmax_abs
        legend_right = vmax_abs
    else:
        vmin = float(finite.quantile(0.02)) if len(finite) else 0.0
        vmax = float(finite.quantile(0.98)) if len(finite) else 0.0
        color = lambda v: _sequential(float(v), vmin, vmax)
        legend_left = vmin
        legend_right = vmax
    lookup = {(int(row.day), int(row.hour)): float(getattr(row, value_col)) for row in frame.itertuples()}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:10px;fill:#222}.title{font-size:13px;font-weight:700}.axis{fill:#555}.cell{stroke:#fff;stroke-width:1}</style>',
        f'<text class="title" x="10" y="18">{html.escape(title)}</text>',
    ]
    for i, hour in enumerate(hours):
        x = LEFT + i * CELL + CELL / 2
        parts.append(f'<text class="axis" x="{x:.1f}" y="{TOP - 7}" text-anchor="middle">{hour}</text>')
    for j, day in enumerate(days):
        y = TOP + j * CELL + CELL / 2 + 3
        parts.append(f'<text class="axis" x="{LEFT - 10}" y="{y:.1f}" text-anchor="end">day {day}</text>')
    for j, day in enumerate(days):
        for i, hour in enumerate(hours):
            value = lookup.get((day, hour), 0.0)
            x = LEFT + i * CELL
            y = TOP + j * CELL
            parts.append(f'<rect class="cell" x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="{color(value)}"><title>day={day} hour={hour} {value_col}={value:.6g}</title></rect>')
    legend_y = TOP + len(days) * CELL + 20
    parts.append(f'<text class="axis" x="{LEFT}" y="{legend_y}">{legend_left:.4g}</text>')
    parts.append(f'<text class="axis" x="{width - RIGHT}" y="{legend_y}" text-anchor="end">{legend_right:.4g}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _profile_svg(frame: pd.DataFrame, title: str) -> str:
    width = 760
    height = 360
    left = 58
    top = 28
    plot_w = width - left - 24
    plot_h = height - top - 48
    sources = [s for s in ["residual", "residual_hat", "hour_component", "baseline_abs_error", "corrected_abs_error"] if s in set(frame["source"])]
    colors = {
        "residual": "#2166ac",
        "residual_hat": "#b2182b",
        "hour_component": "#4d9221",
        "baseline_abs_error": "#7570b3",
        "corrected_abs_error": "#e6ab02",
    }
    pivot = frame.pivot_table(index="hour", columns="source", values="value", aggfunc="mean").sort_index()
    values = pivot[sources].to_numpy().reshape(-1)
    vmin = float(pd.Series(values).quantile(0.02))
    vmax = float(pd.Series(values).quantile(0.98))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    hours = list(pivot.index.astype(int))
    def xy(hour_idx: int, value: float) -> tuple[float, float]:
        x = left + plot_w * hour_idx / max(len(hours) - 1, 1)
        y = top + plot_h * (1.0 - (value - vmin) / (vmax - vmin))
        return x, y
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:11px;fill:#222}.title{font-size:14px;font-weight:700}.axis{fill:#666}.line{fill:none;stroke-width:2}</style>',
        f'<text class="title" x="10" y="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#aaa"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#aaa"/>',
        f'<text class="axis" x="{left - 8}" y="{top + 4}" text-anchor="end">{vmax:.4g}</text>',
        f'<text class="axis" x="{left - 8}" y="{top + plot_h}" text-anchor="end">{vmin:.4g}</text>',
    ]
    for hour in range(0, 24, 3):
        if hour in hours:
            i = hours.index(hour)
            x, _ = xy(i, vmin)
            parts.append(f'<text class="axis" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{hour}</text>')
    for source in sources:
        points = []
        for i, hour in enumerate(hours):
            value = float(pivot.loc[hour, source])
            x, y = xy(i, value)
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline class="line" points="{" ".join(points)}" stroke="{colors[source]}"/>')
    legend_x = left
    legend_y = height - 12
    for source in sources:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" fill="{colors[source]}"/>')
        parts.append(f'<text x="{legend_x + 14}" y="{legend_y}">{html.escape(source)}</text>')
        legend_x += 138
    parts.append("</svg>")
    return "\n".join(parts)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_visualization_dir(viz_dir: Path) -> list[Path]:
    out_dir = viz_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    profiles = viz_dir / "profiles_by_hour.csv"
    if profiles.exists():
        frame = pd.read_csv(profiles)
        path = out_dir / "profiles_by_hour.svg"
        _write_text(path, _profile_svg(frame, title=str(viz_dir.parent)))
        written.append(path)
    heatmap_cols = {
        "sales": False,
        "baseline": False,
        "corrected": False,
        "residual": True,
        "residual_hat": True,
        "hour_component": True,
        "day_component": True,
        "interaction_component": True,
        "baseline_abs_error": False,
        "corrected_abs_error": False,
    }
    for csv_path in sorted(viz_dir.glob("series_*.csv")):
        if csv_path.name == "series_summary.csv":
            continue
        frame = pd.read_csv(csv_path)
        for col, signed in heatmap_cols.items():
            if col not in frame:
                continue
            path = out_dir / f"{csv_path.stem}_{col}.svg"
            title = f"{viz_dir.parent.name} / {csv_path.stem} / {col}"
            _write_text(path, _heatmap_svg(frame, col, title=title, signed=signed))
            written.append(path)
    links = "\n".join(f'<li><a href="{p.name}">{html.escape(p.name)}</a></li>' for p in written if p.parent == out_dir)
    _write_text(out_dir / "index.html", f"<!doctype html><meta charset='utf-8'><title>{html.escape(str(viz_dir.parent))}</title><h1>{html.escape(str(viz_dir.parent))}</h1><ul>{links}</ul>")
    written.append(out_dir / "index.html")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SVG heatmaps from 2-Exp-21 visualization CSV files.")
    parser.add_argument("--root", default="runs/2-Exp-21_freshretailnet_visualization", help="2-Exp-21 run root.")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"run root not found: {root}")
    viz_dirs = sorted(path for path in root.rglob("visualization") if path.is_dir())
    if not viz_dirs:
        raise SystemExit(f"no visualization directories found under: {root}")
    written: list[Path] = []
    for viz_dir in viz_dirs:
        written.extend(plot_visualization_dir(viz_dir))
    print(f"wrote {len(written)} files")
    for path in written[:20]:
        print(path)
    if len(written) > 20:
        print(f"... {len(written) - 20} more")


if __name__ == "__main__":
    main()
