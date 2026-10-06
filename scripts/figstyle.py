"""Shared paper figure style: scienceplots (serif, CM math) + large fonts,
vector-PDF friendly. Import and call apply_style() before plotting."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass
    # Figures are drawn at roughly twice their printed width, so these
    # sizes correspond to ~9-10 pt on the page.
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "figure.titlesize": 19,
        "axes.linewidth": 1.1,
        "lines.linewidth": 2.4,
        "lines.markersize": 8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
