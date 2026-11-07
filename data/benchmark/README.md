# Optional: Nifty 50 benchmark

Drop one CSV of Nifty 50 index history in this folder and re-run `python run_pipeline.py`.
The engine then adds a Nifty 50 line to every chart and computes beta, tracking error and
information ratio for each portfolio.

Download: niftyindices.com > Reports > Historical Data > Index: NIFTY 50, from 01-Apr-2021 to
30-Mar-2026. Any file with a `Date` column and a `Close` column works.
