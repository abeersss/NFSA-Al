"""Figures for the IJ-ICT manuscript (300 dpi PNG, print-safe greys + one blue)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif"})
BLUE = "#1f5fbf"
GREY = "#555555"
LIGHT = "#e8eef8"


# --------------------------------------------------------------------------
# Figure 1: one Turbo-NAFS round
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc="white", ec="black", lw=1.0, fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight)


def arrow(ax, p, q, color="black", lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(6.3, 3.9))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 6.6)
ax.axis("off")

# state halves
box(ax, 1.3, 6.0, 1.6, 0.5, r"$L$ (64 bits)", weight="bold")
box(ax, 9.0, 6.0, 1.6, 0.5, r"$R$ (64 bits)", weight="bold")

# superposition block
ax.add_patch(FancyBboxPatch((2.25, 2.7), 5.6, 2.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc=LIGHT, ec=BLUE, lw=1.3))
ax.text(5.05, 5.05, r"Superposition $\Sigma_r(L)$ of three key-derived paths",
        ha="center", va="center", fontsize=7.6, color=BLUE)
ys = [4.5, 3.75, 3.0]
labels = [["MUL", "SIG", "ROT", "ARX"], ["SIG", "ARX", "XOR", "ARX"], ["XSR", "MUL", "SUB", "ROT"]]
for j, (y, lab) in enumerate(zip(ys, labels), start=1):
    xs = [3.45, 4.3, 5.15, 6.0]
    for k, (x, t) in enumerate(zip(xs, lab)):
        box(ax, x, y, 0.72, 0.42, t, fs=7.5)
        if k:
            arrow(ax, (xs[k - 1] + 0.36, y), (x - 0.36, y), color=GREY, lw=0.8)
    box(ax, 7.1, y, 0.78, 0.42, rf"$\times\alpha_{{r,{j}}}$", fs=8)
    arrow(ax, (xs[-1] + 0.36, y), (7.1 - 0.39, y), color=GREY, lw=0.8)
    ax.text(2.38, y + 0.02, rf"$P_{{r,{j}}}$", ha="left", va="center", fontsize=8)
# L feeds the paths
ax.plot([1.3, 1.3], [5.75, 2.2], color="black", lw=1.0)
for y in ys:
    ax.plot([1.3, 2.25], [y, y], color=GREY, lw=0.8)
    arrow(ax, (2.82, y), (3.45 - 0.36, y), color=GREY, lw=0.8)
# sum and modular addition into R
ax.add_patch(Circle((8.15, 3.75), 0.2, fc="white", ec="black"))
ax.text(8.15, 3.75, r"$\Sigma$", ha="center", va="center", fontsize=8)
for y in ys:
    arrow(ax, (7.1 + 0.39, y), (8.15 - 0.17, 3.75 + (y - 3.75) * 0.45), color=GREY, lw=0.8)
ax.add_patch(Circle((9.0, 3.75), 0.22, fc="white", ec="black"))
ax.text(9.0, 3.75, r"$\boxplus$", ha="center", va="center", fontsize=11)
arrow(ax, (8.35, 3.75), (8.78, 3.75))
arrow(ax, (9.0, 5.75), (9.0, 3.97))
ax.text(9.15, 4.75, r"mod $2^{64}$", fontsize=7.5, color=GREY)

# full-state layer
arrow(ax, (1.3, 2.2), (1.3, 1.95))
arrow(ax, (9.0, 3.53), (9.0, 1.95))
box(ax, 5.15, 1.7, 8.6, 0.5, r"$\oplus\,K_r$ (128-bit round key)", fs=8.5)
box(ax, 5.15, 1.05, 8.6, 0.5, r"SubBytes: AES S-box on all 16 bytes", fs=8.5)
box(ax, 5.15, 0.4, 8.6, 0.5, r"$\pi_r$: key-derived byte permutation, L-bytes $\leftrightarrow$ R-bytes", fs=8.5)
arrow(ax, (5.15, 1.45), (5.15, 1.3))
arrow(ax, (5.15, 0.8), (5.15, 0.65))
ax.text(9.55, 1.7, "full\nstate", fontsize=7.5, color=GREY, va="center")
fig.savefig("fig1_round.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 2: strict avalanche statistics vs rounds
# --------------------------------------------------------------------------
r = json.load(open("../../results/results.json"))
curve = r["diffusion"]["round_curve"]
rounds = [c["rounds"] for c in curve]
mean = [c["mean_flip"] for c in curve]
lo = [c["min_cell"] for c in curve]
hi = [c["max_cell"] for c in curve]
first = next(c["rounds"] for c in curve if c["sac_p"] > 0.001)

fig, ax = plt.subplots(figsize=(6.3, 2.9))
ax.vlines(rounds, lo, hi, color="#9a9a9a", lw=3.2, label="range over all 16,384 bit pairs")
ax.plot(rounds, mean, "-o", color=BLUE, ms=4, lw=1.6, label="mean flip probability")
ax.hlines(0.5, 0.5, 16.4, color=GREY, ls="--", lw=0.8)
ax.text(16.55, 0.5, "ideal\n0.50", va="center", fontsize=8, color=GREY)
ax.axvline(first, color="black", ls=":", lw=0.8)
ax.text(first + 0.2, 0.67, f"SAC test passes from round {first}", fontsize=8)
for x, y in zip(rounds[:first - 1], mean[:first - 1]):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(7, -3), fontsize=8)
ax.set_xlim(0.5, 17.6)
ax.set_ylim(0, 0.72)
ax.set_xticks(rounds)
ax.set_xlabel("Number of turbo rounds")
ax.set_ylabel("Output-bit flip probability")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.savefig("fig2_diffusion.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("ok", first)
