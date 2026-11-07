"""Static PNG charts for the README (light surface, validated categorical palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
COLORS = {"equal_weight": "#2a78d6", "min_variance": "#eb6834", "momentum_tilt": "#1baf7a",
          "benchmark": "#52514e"}
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
LABEL = {"equal_weight": "Equal-weight", "min_variance": "Minimum-variance",
         "momentum_tilt": "Momentum-tilted", "benchmark": "Nifty 50"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _fig(w=10, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    _style(ax)
    return fig, ax


def _gap_band(ax, res, bottom=False):
    g = res["meta"]["gap_return_dates"]
    if not g:
        return
    end = pd.Timestamp(g[0]["trade_date"])
    start = end - pd.Timedelta(days=int(g[0]["gap_days"]))
    ax.axvspan(start, end, color="#e6e5e1", alpha=0.7, lw=0)
    lo, hi = ax.get_ylim()
    y = (lo + hi) / 2 if bottom else hi
    ax.text(start + (end - start) / 2, y, "Tata Investment\ndata gap\n(no daily prices)",
            ha="center", va="center" if bottom else "top", fontsize=7.5, color=INK2)


def _lines(res, key, series_keys):
    idx = pd.to_datetime(res["dates"])
    return idx, {k: np.array(res[key][k], dtype=float) for k in series_keys if k in res[key]}


def nav_chart(res, out):
    keys = ["equal_weight", "min_variance", "momentum_tilt", "benchmark"]
    idx, d = _lines(res, "nav", keys)
    fig, ax = _fig()
    for k, v in d.items():
        ax.plot(idx, v * 100, color=COLORS[k], lw=2, label=LABEL[k],
                ls="--" if k == "benchmark" else "-")
    ax.set_xlim(idx[0], idx[-1] + pd.Timedelta(days=170))
    _gap_band(ax, res)
    ends = sorted(((v[-1] * 100, k) for k, v in d.items()))
    last_y = None
    for y, k in ends:
        yy = y if last_y is None or y - last_y > 5 else last_y + 5
        ax.text(idx[-1] + pd.Timedelta(days=10), yy, f"{LABEL[k]}  {y:,.0f}", color=INK,
                fontsize=9, va="center")
        last_y = yy
    ax.set_ylabel("Value of 100 invested (net of costs)", color=INK2, fontsize=9)
    ax.set_title("Walk-forward backtest: growth of 100", loc="left", color=INK, fontsize=12,
                 fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "nav.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def drawdown_chart(res, out):
    keys = ["equal_weight", "min_variance", "momentum_tilt", "benchmark"]
    idx, d = _lines(res, "drawdown", keys)
    fig, ax = _fig(10, 4.2)
    g = res["meta"]["gap_return_dates"]
    brk = pd.Timestamp(g[0]["trade_date"]) - pd.Timedelta(days=1) if g else None
    for k, v in d.items():
        ser = pd.Series(v * 100, index=idx)
        if brk is not None:                       # break the line: drawdown inside the gap is unobservable
            ser.loc[brk] = np.nan
            ser = ser.sort_index()
        ax.plot(ser.index, ser.values, color=COLORS[k], lw=1.8, label=LABEL[k],
                ls="--" if k == "benchmark" else "-")
    ax.axhline(0, color=GRID, lw=1)
    ax.set_ylabel("Drawdown from peak (%)", color=INK2, fontsize=9)
    ax.set_title("Drawdowns", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK2, ncol=4)
    _gap_band(ax, res, bottom=True)
    fig.tight_layout()
    fig.savefig(out / "drawdown.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def corr_heatmap(res, out):
    t = res["corr"]["tickers"]
    sec = res["sectors"]
    order = sorted(t, key=lambda x: (sec[x], x))
    pos = [t.index(o) for o in order]
    m = np.array(res["corr"]["matrix"])[np.ix_(pos, pos)]
    fig, ax = plt.subplots(figsize=(8.6, 7.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    off = m.copy()
    np.fill_diagonal(off, np.nan)
    im = ax.imshow(off, cmap=SEQ, vmin=0, vmax=0.7)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(order)), order, fontsize=8.5, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                continue
            v = m[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#fcfcfb" if v > 0.35 else INK)
    ax.set_title("Daily-return correlation (105 unique pairs)", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    fig.tight_layout()
    fig.savefig(out / "correlation.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def risk_return(res, out):
    s = pd.DataFrame(res["stocks"])
    fr = pd.DataFrame(res["frontier"])
    fig, ax = _fig(9.5, 5.6)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.plot(fr["vol"] * 100, fr["ret"] * 100, color=COLORS["min_variance"], lw=2,
            label="Long-only efficient frontier (in-sample)")
    ax.scatter(s["ann_vol"] * 100, s["ann_mean_return"] * 100, s=60, color=COLORS["equal_weight"],
               edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Individual stocks")
    mv = res["insample_minvar"]
    ax.scatter([mv["vol"] * 100], [mv["ret"] * 100], s=110, marker="D", color=COLORS["min_variance"],
               edgecolor=SURFACE, linewidth=1.5, zorder=4, label="Min-variance (in-sample)")
    # crowded clusters get leader lines so every label points at exactly one dot
    offsets = {"TATAINVEST": (-10, 18), "BHEL": (12, -16), "SHRIRAMFIN": (-8, 18), "VEDL": (12, 16),
                "SUNPHARMA": (-14, 14), "LT": (6, 32), "SBIN": (24, 16), "BAJAJ-AUTO": (-32, -18),
                "UNITDSPR": (4, -26), "HDFCBANK": (-14, 14), "HINDUNILVR": (-10, -16)}
    for r in s.itertuples():
        dx, dy = offsets.get(r.ticker, (7, 4))
        ax.annotate(r.ticker, (r.ann_vol * 100, r.ann_mean_return * 100), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=INK2,
                    ha="right" if dx < 0 else "left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=4)
                    if abs(dx) > 10 or abs(dy) > 10 else None)
    ax.set_xlabel("Annualised volatility (%)", color=INK2, fontsize=9)
    ax.set_ylabel("Annualised mean return (%)", color=INK2, fontsize=9)
    ax.set_title("Risk vs return, Apr 2021 to Mar 2026 (full sample, reference only)", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "risk_return.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def latest_weights(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), facecolor=SURFACE, sharex=True)
    for ax, key in zip(axes, ("min_variance", "momentum_tilt")):
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        w = res["weights"][key]
        s = pd.Series(w["w"][-1], index=w["tickers"]).sort_values()
        ax.barh(s.index, s.values * 100, color=COLORS[key], height=0.62)
        for y, v in enumerate(s.values):
            ax.text(v * 100 + 0.4, y, f"{v * 100:.1f}%", va="center", fontsize=8, color=INK2)
        ax.set_title(f"{LABEL[key]}\nweights at {w['dates'][-1]}", loc="left", color=INK, fontsize=11,
                     fontweight="bold")
        ax.tick_params(axis="y", labelsize=8.5)
    axes[0].set_xlim(0, 32)
    fig.tight_layout()
    fig.savefig(out / "weights.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def make_all(res, out_dir: Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (nav_chart, drawdown_chart, corr_heatmap, risk_return, latest_weights):
        fn(res, out)
