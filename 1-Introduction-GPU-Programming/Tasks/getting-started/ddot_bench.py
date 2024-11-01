import matplotlib as mpl
mpl.use('Agg')
import pandas as pd


df = pd.read_csv("ddot_bench.dat", header=None, names=["Vector Length", "Implementation", "Runtime / s"])

df["Performance / MFlOp/s"] = (2 * df["Vector Length"] - 1) * 1e-6 / df["Runtime / s"]

ax = df.pivot_table(columns="Implementation", values="Performance / MFlOp/s", index="Vector Length").plot(title="DDot Benchmark", loglog=True, style="o-")
ax.set_ylabel("Performance / MFlOp/s ")
fig = ax.get_figure()
fig.savefig("ddot_bench.png")
fig.savefig("ddot_bench.svg")
fig.savefig("ddot_bench.pdf")
