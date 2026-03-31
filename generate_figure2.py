import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe

# Load data
# Expected columns include: waveform, Ti_s, pause_fraction, R2_over_R1, C2_over_C1, PI
# Data file should be present in the working directory when the script is run.
df = pd.read_csv("sweep_PI_MP_patterns.csv")

# Baseline scenario: square flow, Ti = 1.0 s, no end-inspiratory pause
df_base = df[
    (df["waveform"] == "square")
    & (df["Ti_s"] == 1.0)
    & (df["pause_fraction"] == 0.0)
].copy()

# Pivot table for heatmap
heatmap_data = (
    df_base.pivot(index="R2_over_R1", columns="C2_over_C1", values="PI")
    .sort_index(axis=0)
    .sort_index(axis=1)
)

x_vals = heatmap_data.columns.to_numpy()
y_vals = heatmap_data.index.to_numpy()
z_vals = heatmap_data.values

# Plot
fig, ax = plt.subplots(figsize=(8.2, 6.4))

# Use a diverging colormap centred at PI = 0.5 to emphasise preferential loading
norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
im = ax.imshow(
    z_vals,
    origin="lower",
    aspect="auto",
    extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
    cmap="coolwarm",
    norm=norm,
    interpolation="nearest",
)

# PI = 0.5 contour
cont = ax.contour(
    x_vals,
    y_vals,
    z_vals,
    levels=[0.5],
    colors="black",
    linewidths=2.0,
)

# Improve contour visibility on mixed background
cont.set_path_effects([pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])

clabels = ax.clabel(cont, fmt={0.5: "PI = 0.5"}, inline=True, fontsize=9)
for txt in clabels:
    txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])

# Mechanical symmetry point
ax.scatter(1.0, 1.0, s=70, marker="o", facecolors="none", edgecolors="black", linewidths=1.1, zorder=5)
ax.annotate(
    "Mechanical symmetry\n(C₂/C₁ = 1, R₂/R₁ = 1)",
    xy=(1.0, 1.0),
    xytext=(1.35, 1.12),
    arrowprops=dict(arrowstyle="->", lw=1.0),
    fontsize=9,
    ha="left",
    va="bottom",
)

# Axes
ax.set_xlabel("Compliance ratio (C₂/C₁)")
ax.set_ylabel("Resistance ratio (R₂/R₁)")

# Colorbar with explicit midpoint tick
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Partition Index (PI)")
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

# Clean layout: no title, no caption embedded in the figure
plt.tight_layout()
plt.savefig("Figure_2.png", dpi=600, bbox_inches="tight")
plt.savefig("Figure_2.svg", bbox_inches="tight")
plt.savefig("Figure_2.pdf", bbox_inches="tight")
plt.show()
