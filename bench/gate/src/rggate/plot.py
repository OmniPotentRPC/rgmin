"""Optional SVG figures. Matplotlib is imported only when a plot is asked for."""


def write_svg(report, path):
    """Write Kaplan-Meier curves and the performance profile to ``path``."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("matplotlib is not installed") from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for method, curve in report.km.items():
        points = curve["curve"]
        if not points:
            continue
        times = [point[0] for point in points]
        survival = [point[1] for point in points]
        low = [point[2] for point in points]
        high = [point[3] for point in points]
        axes[0].step(times, survival, where="post", label=method)
        axes[0].fill_between(times, low, high, step="post", alpha=0.15)
    axes[0].set_xlabel("calls")
    axes[0].set_ylabel("survival")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=7, loc="best")

    taus = report.perf_taus
    for index, method in enumerate(report.method_names):
        values = [row[index] for row in report.perf]
        axes[1].step(taus, values, where="post", label=method)
    axes[1].set_xlabel("tau")
    axes[1].set_ylabel("fraction")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(path, format="svg")
    plt.close(figure)
