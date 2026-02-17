import polars as pl
import numpy as np

csv_path = "output/all_trials_timeseries.csv"
sub = "강비은"
vel = 30.0

df = pl.read_csv(csv_path, infer_schema_length=10000, encoding="utf8-lossy")
g = df.filter((pl.col("subject")==sub) & (pl.col("velocity")==vel)).sort(["trial","MocapFrame"])

if g.height == 0:
    print("NO ROWS for", sub, vel)
    raise SystemExit

trials = sorted(set(g.get_column("trial").to_list()))
print("trials:", trials)

for trial in trials:
    tdf = g.filter(pl.col("trial")==trial).sort("MocapFrame")
    for col in ["Ankle_L_Z_deg","Ankle_R_Z_deg"]:
        if col not in tdf.columns:
            continue
        arr = np.asarray(tdf.get_column(col).to_list(), dtype=float)
        fin = np.isfinite(arr)
        d = np.abs(np.diff(arr[fin]))
        mx = float(np.nanmax(d)) if d.size else float("nan")
        print(f"trial={trial:>2} {col}: max|diff|={mx:.1f} deg  min={np.nanmin(arr):.1f} max={np.nanmax(arr):.1f}")
