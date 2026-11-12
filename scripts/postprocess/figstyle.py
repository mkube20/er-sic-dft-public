"""
figstyle.py: shared matplotlib style for the manuscript figures.

Arial or Helvetica at 8 pt, figures drawn at their printed size, vector PDF
with embedded fonts and a colorblind-safe (Wong) palette, following the
npj Computational Materials figure guidelines.
"""

import matplotlib as mpl

FULL_W = 6.85
COL_W = 3.31

BASE = 8

PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "red":    "#D55E00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
    "gray":   "#7F7F7F",
    "black":  "#000000",
}


def apply():
    """Apply the shared matplotlib style.
    Used by: all plotting scripts.
    No inputs or return value.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": BASE,
        "axes.titlesize": BASE,
        "axes.labelsize": BASE,
        "xtick.labelsize": BASE,
        "ytick.labelsize": BASE,
        "legend.fontsize": BASE,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "axes.labelpad": 3.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.4,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 6,
    })


def panel_label(ax, letter, *, x=0.0, y=1.0, **kw):
    """Draw a bold panel letter in the corner of an axes.
    Used by: plot_structure.
    Inputs: axes, letter.
    """
    ax.annotate(f"({letter})", xy=(x, y), xycoords="axes fraction",
                xytext=(0, 4), textcoords="offset points",
                ha="left", va="bottom", fontweight="bold",
                fontsize=BASE + 1, **kw)


def save(fig, output):
    """Save a figure as PDF and PNG.
    Used by: all plotting scripts.
    Inputs: figure, output path.
    """
    from pathlib import Path
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(str(output).replace(".pdf", ".png"), dpi=400)
    print(f"Written: {output}")
