from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PEAKS_TO_HIGHLIGHT = 10


def load_capture(csv_path):
    """Load the ciphertext column and 200 trace samples from the capture."""
    frame = pd.read_csv(csv_path, header=None, dtype={0: str})
    ciphertexts = np.frombuffer(
        b"".join(bytes.fromhex(value.strip()) for value in frame.iloc[:, 0]),
        dtype=np.uint8,
    ).reshape(-1, 16)
    traces = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    return ciphertexts, traces


def main():
    root = Path(__file__).resolve().parents[1]
    _, traces = load_capture(root / "Round10_traces.csv")

    centered = traces - traces.mean(axis=0)
    variance = np.mean(centered * centered, axis=0)
    samples = np.arange(traces.shape[1])

    # Same local-maximum rule as third_task.py. A plateau is represented by
    # its rightmost point because the right neighbor must be strictly lower.
    local_peaks = (
        np.flatnonzero(
            (variance[1:-1] >= variance[:-2]) & (variance[1:-1] > variance[2:])
        )
        + 1
    )
    if len(local_peaks) < PEAKS_TO_HIGHLIGHT:
        raise ValueError(
            f"found only {len(local_peaks)} local maxima; "
            f"need {PEAKS_TO_HIGHLIGHT} to highlight"
        )

    strongest = local_peaks[np.argsort(variance[local_peaks])[-PEAKS_TO_HIGHLIGHT:]]
    detected = np.sort(strongest)

    print(f"Traces used: {len(traces)} (all traces in the capture)")
    print(f"All local maxima across samples 0–{traces.shape[1] - 1}:")
    print("sample   variance")
    for sample in local_peaks:
        print(f"{sample:>6}   {variance[sample]:.12g}")

    print("\nTen strongest local maxima from all traces:")
    print("sample   variance          left-neighbor   right-neighbor")
    for sample in detected:
        print(
            f"{sample:>6}   {variance[sample]:.12g}   "
            f"{variance[sample - 1]:.12g}   {variance[sample + 1]:.12g}"
        )

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(
        samples,
        variance,
        color="steelblue",
        linewidth=1.5,
        label="Across-trace variance",
    )
    selected_mask = np.isin(local_peaks, detected)
    other_peaks = local_peaks[~selected_mask]
    ax.scatter(
        other_peaks,
        variance[other_peaks],
        s=20,
        color="0.65",
        zorder=3,
        label="Other local maxima (full window)",
    )
    ax.scatter(
        detected,
        variance[detected],
        s=58,
        color="crimson",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label=f"{PEAKS_TO_HIGHLIGHT} strongest local maxima",
    )

    for sample in detected:
        ax.annotate(
            f"{sample}",
            (sample, variance[sample]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="darkred",
        )

    ax.set(
        title="Across-trace variance across the 200-sample capture",
        xlabel="Time sample (zero-based index)",
        ylabel="Across-trace variance",
    )
    ax.set_xlim(0, traces.shape[1] - 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.text(
        0.5,
        0.015,
        "Marked points are local variance maxima selected as activity candidates; "
        "variance alone does not prove AES round transitions.\n"
        "Their repeated pattern suggests register activity associated with state transitions.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    output = root / "variance_plots" / "variance_plot.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Plot saved to: {output}")


if __name__ == "__main__":
    main()
