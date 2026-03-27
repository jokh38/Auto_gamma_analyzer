"""Shared plotting styles for the main app and generated reports."""

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def get_gamma_colormap():
    """Return the report gamma colormap for consistent app/report rendering."""
    cmap = LinearSegmentedColormap.from_list(
        "gamma_report_map",
        [
            (0.0, "#3b4cc0"),
            (0.5, "#f0f0f0"),
            (1.0, "#b40426"),
        ],
    )
    cmap.set_bad(color="#ffffff")
    return cmap


def get_gamma_norm(max_gamma=None):
    """Center gamma colouring at the pass/fail boundary of 1.0."""
    vmax = 2.0 if max_gamma is None else max(2.0, float(max_gamma))
    return TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
