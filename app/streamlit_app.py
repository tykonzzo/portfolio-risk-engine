"""Interactive dashboard. Run locally:  streamlit run app/streamlit_app.py

Reads the files written by run_pipeline.py (outputs/results.json). If they are missing,
the pipeline is run once on first load (takes a few seconds).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Portfolio Risk Engine", layout="wide")

COLORS = {"equal_weight": "#2a78d6", "min_variance": "#eb6834", "momentum_tilt": "#1baf7a",
          "benchmark": "#52514e"}


@st.cache_data(show_spinner="Building the database and running the backtest...")
def load() -> dict:
    path = ROOT / "outputs" / "results.json"
    if not path.exists():
        import run_pipeline
        run_pipeline.main(["--no-charts"])
    return json.loads(path.read_text())


R = load()
labels = R["meta"]["labels"]
keys = [k for k in ("equal_weight", "min_variance", "momentum_tilt", "benchmark") if k in R["nav"]]
dates = pd.to_datetime(R["dates"])

st.title("Portfolio Risk Engine")
st.caption("15 NSE stocks, Apr 2021 to Mar 2026. SQL pipeline, then a walk-forward backtest of three long-only portfolios.")

cfg = R["meta"]["config"]
with st.sidebar:
    st.header("Window")
    window = st.radio("Chart window", ["Full period", "3Y", "2Y", "1Y"], index=0)
    st.markdown(
        f"**Assumptions**  \nLookback {cfg['lookback']} days  \nMonthly rebalance  \n"
        f"Cost {cfg['cost_bps']:.0f} bps one-way  \nRisk-free {cfg['rf']:.1%}  \n"
        f"Min-variance cap {cfg['max_weight']:.0%} per stock")
    st.markdown("Change them with `python run_pipeline.py --help`.")

years = {"Full period": None, "3Y": 3, "2Y": 2, "1Y": 1}[window]
i0 = 0
if years:
    cut = dates[-1] - pd.DateOffset(years=years)
    i0 = int((dates >= cut).argmax())

# ---- scorecard
m = pd.DataFrame([r for r in R["metrics"] if r["key"] in keys]).set_index("strategy")
show = m[["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "avg_annual_turnover"]]
st.subheader("Scorecard")
st.caption(f"Out-of-sample {R['dates'][0]} to {R['dates'][-1]}, net of costs.")
st.dataframe(
    show.rename(columns={"total_return": "Total return", "cagr": "CAGR", "ann_vol": "Volatility", "sharpe": "Sharpe",
                         "max_drawdown": "Max drawdown", "calmar": "Calmar", "avg_annual_turnover": "Turnover / yr"})
    .style.format({"Total return": "{:.1%}", "CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}",
                   "Max drawdown": "{:.1%}", "Calmar": "{:.2f}", "Turnover / yr": "{:.0%}"}),
    width="stretch")

# ---- growth and drawdown
fig = go.Figure()
dd = go.Figure()
for k in keys:
    nav = pd.Series(R["nav"][k][i0:], index=dates[i0:])
    nav = nav / nav.iloc[0] * 100
    style = dict(color=COLORS[k], width=2, dash="dash" if k == "benchmark" else "solid")
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name=labels[k], line=style))
    dd.add_trace(go.Scatter(x=nav.index, y=(nav / nav.cummax() - 1) * 100, name=labels[k], line=style))
gap = R["meta"]["gap_return_dates"]
for f in (fig, dd):
    if gap:
        end = pd.Timestamp(gap[0]["trade_date"])
        f.add_vrect(x0=end - pd.Timedelta(days=int(gap[0]["gap_days"])), x1=end, fillcolor="grey", opacity=0.15,
                    line_width=0, annotation_text="Data gap", annotation_position="top left")
    f.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
fig.update_yaxes(title="Value of 100 invested")
dd.update_yaxes(title="Below peak (%)")
st.subheader("Growth of 100")
st.plotly_chart(fig, width="stretch")
st.subheader("Drawdowns")
st.plotly_chart(dd, width="stretch")

# ---- correlation and risk-return
c1, c2 = st.columns(2)
tick = R["corr"]["tickers"]
order = sorted(tick, key=lambda t: (R["sectors"][t], t))
mat = pd.DataFrame(R["corr"]["matrix"], index=tick, columns=tick).loc[order, order]
for i in range(len(order)):
    mat.iat[i, i] = None
with c1:
    st.subheader("Diversification")
    h = go.Figure(go.Heatmap(z=mat.values, x=order, y=order, zmin=0, zmax=0.7, colorscale="Blues",
                             hovertemplate="%{y} and %{x}: %{z:.2f}<extra></extra>"))
    h.update_yaxes(autorange="reversed")
    h.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(h, width="stretch")
with c2:
    st.subheader("Risk vs return")
    S = pd.DataFrame(R["stocks"])
    fr = pd.DataFrame(R["frontier"])
    s = go.Figure()
    s.add_trace(go.Scatter(x=fr.vol * 100, y=fr.ret * 100, mode="lines", name="Efficient frontier (in-sample)",
                           line=dict(color=COLORS["min_variance"])))
    s.add_trace(go.Scatter(x=S.ann_vol * 100, y=S.ann_mean_return * 100, mode="markers+text", text=S.ticker,
                           textposition="top right", name="Stocks", marker=dict(color=COLORS["equal_weight"], size=9)))
    s.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Annualised volatility (%)",
                    yaxis_title="Annualised mean return (%)", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(s, width="stretch")

# ---- weights
st.subheader("Weights at each rebalance")
strat = st.radio("Strategy", ["min_variance", "momentum_tilt"], format_func=lambda k: labels[k], horizontal=True)
W = R["weights"][strat]
idx = st.slider("Rebalance", 0, len(W["dates"]) - 1, len(W["dates"]) - 1, format="")
st.caption(f"Weights set at the close of {W['dates'][idx]}")
ws = pd.Series(W["w"][idx], index=W["tickers"]).sort_values()
b = go.Figure(go.Bar(x=ws * 100, y=ws.index, orientation="h", marker_color=COLORS[strat],
                     text=[f"{v:.1f}%" for v in ws * 100], textposition="outside"))
b.update_layout(height=460, margin=dict(l=10, r=40, t=10, b=10), xaxis=dict(range=[0, 32], ticksuffix="%"))
st.plotly_chart(b, width="stretch")

# ---- notes
with st.expander("Data and method notes", expanded=False):
    q = R["meta"]
    st.markdown(
        + (f"- **Gap:** {', '.join(q['tickers_with_missing_days']) or 'One stock'} is missing {q['excluded_non_common_days']} trading days "
           f"(around {q['gap_return_dates'][0]['trade_date']}). Those days are dropped for every stock; the one return spanning the gap "
           "stays in portfolio value but is excluded from volatility, correlation and Sharpe.\n" if q["gap_return_dates"] else "- **Gap:** none. Every stock has every trading day.\n") +
        "- **Prices** are back-adjusted for six splits and bonuses (table below). Dividends are not included.\n"
        "- **Sharpe** = (CAGR - risk-free) / annualised volatility of daily returns.")
    st.dataframe(pd.DataFrame(q["corporate_actions"]), width="stretch", hide_index=True)
