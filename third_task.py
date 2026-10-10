#!/usr/bin/env python3
"""Task 3: derive timing from trace activity and scan round-10 key candidates.

First, it detects activity samples from across-trace variance without key
guesses or a preset sample spacing. It then scans all missing-endpoint key
candidates using transition HD, HW(S[r-1] XOR S[r]), at those detected samples.
The result assumes the ten strongest local variance peaks correspond to AES
round events and depends on the shared missing-endpoint and leakage hypotheses.
The recovered RK10 is used directly by fourth_task.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SBOX = np.array(
    [
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        1,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,
        4,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        5,
        0x9A,
        7,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,
        9,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,
        0x53,
        0xD1,
        0,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        2,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        6,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        8,
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        3,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,
    ],
    dtype=np.uint8,
)
RCON = (1, 2, 4, 8, 16, 32, 64, 128, 0x1B, 0x36)


def expand_aes128(key):
    words = [int.from_bytes(key[i : i + 4], "big") for i in range(0, 16, 4)]
    for i in range(4, 44):
        t = words[i - 1]
        if i % 4 == 0:
            t = ((t << 8) & 0xFFFFFFFF) | (t >> 24)
            t = sum(
                int(SBOX[(t >> (24 - 8 * j)) & 0xFF]) << (24 - 8 * j) for j in range(4)
            ) ^ (RCON[i // 4 - 1] << 24)
        words.append(words[i - 4] ^ t)
    return [
        b"".join(words[4 * r + j].to_bytes(4, "big") for j in range(4))
        for r in range(11)
    ]


def invert_aes128_rk10(round_key):
    words = [None] * 44
    for j in range(4):
        words[40 + j] = int.from_bytes(round_key[4 * j : 4 * j + 4], "big")
    for i in range(43, 3, -1):
        t = words[i - 1]
        if i % 4 == 0:
            t = ((t << 8) & 0xFFFFFFFF) | (t >> 24)
            t = sum(
                int(SBOX[(t >> (24 - 8 * j)) & 0xFF]) << (24 - 8 * j) for j in range(4)
            ) ^ (RCON[i // 4 - 1] << 24)
        words[i - 4] = words[i] ^ t
    return b"".join(words[i].to_bytes(4, "big") for i in range(4))


INV_SBOX = np.argsort(SBOX).astype(np.uint8)
HW = np.array([x.bit_count() for x in range(256)], dtype=np.float64)
INV_MC = ((14, 11, 13, 9), (9, 14, 11, 13), (13, 9, 14, 11), (11, 13, 9, 14))
OUT_FOR_PRE_SHIFT = np.array(
    [4 * ((c - r) % 4) + r for c in range(4) for r in range(4)]
)


def gf_mul(a, b):
    out = 0
    for _ in range(8):
        if b & 1:
            out ^= a
        a = ((a << 1) ^ (0x11B if a & 0x80 else 0)) & 0xFF
        b >>= 1
    return out


def make_gf_tables():
    tables = {}
    for factor in {x for row in INV_MC for x in row}:
        tables[factor] = np.array(
            [gf_mul(x, factor) for x in range(256)], dtype=np.uint8
        )
    return tables


GF = make_gf_tables()


def invert_to_round8(ciphertexts, rk10, rk9):
    """Return S9 and S8 for column-major AES state bytes."""
    s9 = np.empty_like(ciphertexts)
    # C = ShiftRows(SubBytes(S9)) XOR RK10.
    for state_pos, cipher_pos in enumerate(OUT_FOR_PRE_SHIFT):
        s9[:, state_pos] = INV_SBOX[ciphertexts[:, cipher_pos] ^ rk10[cipher_pos]]

    # Undo round 9 AddRoundKey and MixColumns.
    u = s9 ^ rk9
    inv_mix = np.empty_like(u)
    for col in range(4):
        for row in range(4):
            value = np.zeros(len(ciphertexts), dtype=np.uint8)
            for k in range(4):
                value ^= GF[INV_MC[row][k]][u[:, 4 * col + k]]
            inv_mix[:, 4 * col + row] = value

    # Undo ShiftRows and SubBytes to recover the old register S8.
    shifted = np.empty_like(inv_mix)
    for col in range(4):
        for row in range(4):
            shifted[:, 4 * col + row] = inv_mix[:, 4 * ((col - row) % 4) + row]
    s8 = INV_SBOX[shifted]
    return s9, s8


def activity_profile(traces):
    """Measure across-trace variance at each sample."""
    centered = traces - traces.mean(axis=0)
    return np.mean(centered * centered, axis=0)


def find_round_activity_samples(traces, rounds=10):
    """Choose the `rounds` strongest local variance peaks, in time order."""
    variance = activity_profile(traces)
    peaks = (
        np.flatnonzero(
            (variance[1:-1] >= variance[:-2]) & (variance[1:-1] > variance[2:])
        )
        + 1
    )
    if len(peaks) < rounds:
        raise ValueError(f"found {len(peaks)} local peaks; need {rounds}")
    strongest = peaks[np.argsort(variance[peaks])[-rounds:]]
    samples = np.sort(strongest)
    return samples, variance


def inverse_round(state, rk):
    """Undo a standard AES round 1..9: S_r -> S_(r-1)."""
    u = state ^ rk
    inv_mix = np.empty_like(u)
    for col in range(4):
        for row in range(4):
            val = np.zeros(len(u), dtype=np.uint8)
            for k in range(4):
                val ^= GF[INV_MC[row][k]][u[:, 4 * col + k]]
            inv_mix[:, 4 * col + row] = val
    shifted = np.empty_like(inv_mix)
    for col in range(4):
        for row in range(4):
            shifted[:, 4 * col + row] = inv_mix[:, 4 * ((col - row) % 4) + row]
    return INV_SBOX[shifted]


def fit_gain(htr, ytr, hte, yte):
    """Fit centered multivariate leakage on train; score it on held-out rows."""
    hm, ym = htr.mean(axis=0), ytr.mean(axis=0)
    htrc, ytrc = htr - hm, ytr - ym
    htec, ytec = hte - hm, yte - ym
    gram = htrc.T @ htrc
    cross = htrc.T @ ytrc
    beta = np.linalg.pinv(gram) @ cross
    ptr, pte = htrc @ beta, htec @ beta
    gtr = (2 * (ytrc * ptr).sum() - (ptr * ptr).sum()) / (ytrc * ytrc).sum()
    gte = (2 * (ytec * pte).sum() - (pte * pte).sum()) / (ytec * ytec).sum()
    return (
        float(gtr),
        float(gte),
        float((ytrc * ytrc).sum()),
        float((ytec * ytec).sum()),
    )


def save_endpoint_correlation_plots(
    root, correlations, rk10, event_samples, selected_z
):
    """Plot all candidate correlations over every sample, grouped by byte/round."""
    output_dir = root / "actual_round10_correlation_plots"
    output_dir.mkdir(exist_ok=True)
    sample_indices = np.arange(correlations.shape[-1])
    for byte in range(16):
        fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True, sharey=True)
        for round_index, axis in enumerate(axes.flat):
            for z in range(256):
                if z == selected_z:
                    continue
                axis.plot(
                    sample_indices,
                    correlations[z, round_index, byte],
                    color="gray",
                    alpha=0.12,
                    linewidth=0.45,
                )
            axis.plot(
                sample_indices,
                correlations[selected_z, round_index, byte],
                color="crimson",
                linewidth=1.5,
                label=f"z=0x{selected_z:02x} (selected)",
            )
            axis.axvline(
                event_samples[round_index],
                color="black",
                alpha=0.45,
                linewidth=0.8,
                linestyle="--",
            )
            axis.axhline(0, color="black", linewidth=0.5)
            axis.set_title(
                f"Round {round_index + 1}, event sample {event_samples[round_index]}"
            )
            axis.set_ylabel("Pearson r")
            axis.grid(alpha=0.2)
        for axis in axes[-1, :]:
            axis.set_xlabel("Trace sample")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"Endpoint candidates across rounds 1–8 — byte {byte:02d} "
            f"(RK10=0x{rk10[byte]:02x})\n"
            "Gray: other z candidates; red: selected global regression candidate"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        output_path = output_dir / f"byte_{byte:02d}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    print(f"Saved all-candidate round correlation plots to {output_dir}")


def load_capture(root):
    """Load the supplied CSV directly, without requiring Task 1 cache files."""
    csv_path = root / "Round10_traces.csv"
    frame = pd.read_csv(csv_path, header=None, dtype={0: str})
    ciphertexts = np.array(
        [np.frombuffer(bytes.fromhex(value), dtype=np.uint8) for value in frame[0]],
        dtype=np.uint8,
    )
    traces = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    if ciphertexts.shape != (len(frame), 16) or traces.shape != (len(frame), 200):
        raise ValueError(
            f"expected N x (ciphertext + 200 samples); got "
            f"ciphertexts {ciphertexts.shape}, traces {traces.shape}"
        )
    return ciphertexts, traces


def main():
    root = Path(__file__).resolve().parent
    ct_all, tr_all = load_capture(root)
    counts = np.stack([np.bincount(ct_all[:, b], minlength=256) for b in range(16)])
    missing = counts.argmin(axis=1).astype(np.uint8)
    doubled = counts.argmax(axis=1).astype(np.uint8)
    print(f"ciphertext rows: {len(ct_all)}")
    print("byte  missing-value  doubled-value  counts(missing/doubled)")
    for b in range(16):
        print(
            f"{b:>4}  0x{missing[b]:02x}          0x{doubled[b]:02x}          "
            f"{counts[b, missing[b]]}/{counts[b, doubled[b]]}"
        )
    print(
        "The histogram identifies absent ciphertext outputs; the 256-way "
        "transition scan resolves the endpoint under its leakage model."
    )
    keep = np.all(ct_all != doubled[None, :], axis=1)
    row_ids = np.flatnonzero(keep)
    ct, traces = ct_all[keep], tr_all[keep]
    training_winners = []
    event_samples = []
    endpoint_correlations = np.empty((256, 8, 16, traces.shape[1]), dtype=np.float32)
    centered_traces = traces - traces.mean(axis=0)
    trace_sumsquares = (centered_traces * centered_traces).sum(axis=0)
    for fold, train_first in enumerate((True, False), 1):
        train = np.flatnonzero((row_ids < 50_000) == train_first)
        test = np.flatnonzero((row_ids < 50_000) != train_first)
        activity_samples, activity_variance = find_round_activity_samples(
            traces[train], rounds=10
        )
        print(
            f"fold {fold} round activity samples discovered from training "
            f"trace variation: {activity_samples.tolist()}"
        )
        print(
            "  across-trace variance at events:",
            [round(float(activity_variance[t]), 10) for t in activity_samples],
        )
        # The first nine detected events are ordinary rounds 1..9. The tenth
        # is the protected final round, excluded from the ordinary-round HD fit.
        round_samples = activity_samples[:9]
        event_samples.append(round_samples[:8])
        candidate_results = []
        for m in range(256):
            rk10 = missing ^ SBOX[m]
            master = invert_aes128_rk10(rk10.tobytes())
            rks = [np.frombuffer(k, dtype=np.uint8) for k in expand_aes128(master)]
            s9, s8 = invert_to_round8(ct, rks[10], rks[9])
            states = {9: s9, 8: s8}
            for r in range(8, 0, -1):
                states[r - 1] = inverse_round(states[r], rks[r])

            round_scores = []
            for r in range(1, 10):
                h = HW[states[r - 1] ^ states[r]]
                sample = int(round_samples[r - 1])
                y = traces[:, sample]
                gt, ge, et, ee = fit_gain(h[train], y[train], h[test], y[test])
                round_scores.append((gt, ge, et, ee, sample))
                if fold == 1 and r <= 8:
                    h_centered = h - h.mean(axis=0)
                    denominator = np.sqrt(
                        (h_centered * h_centered).sum(axis=0)[:, None]
                        * trace_sumsquares[None, :]
                    )
                    endpoint_correlations[m, r - 1] = (
                        h_centered.T @ centered_traces
                    ) / np.maximum(denominator, 1e-20)
            train8 = sum(a * et for a, _, et, _, _ in round_scores[:8]) / sum(
                et for _, _, et, _, _ in round_scores[:8]
            )
            test8 = sum(b * ee for _, b, _, ee, _ in round_scores[:8]) / sum(
                ee for _, _, _, ee, _ in round_scores[:8]
            )
            candidate_results.append((m, round_scores, train8, test8))

        by_train = sorted(candidate_results, key=lambda x: x[2], reverse=True)
        by_test = sorted(candidate_results, key=lambda x: x[3], reverse=True)
        ff = next(r for r in candidate_results if r[0] == 0xFF)
        ff_train_rank = next(i + 1 for i, r in enumerate(by_train) if r[0] == 0xFF)
        ff_test_rank = next(i + 1 for i, r in enumerate(by_test) if r[0] == 0xFF)
        train_winner = by_train[0]
        training_winners.append(train_winner[0])
        print(f"fold {fold}: train/test={len(train)}/{len(test)}")
        print(
            f"  training-selected endpoint (rounds 1-8): "
            f"z=0x{train_winner[0]:02x}, gain={train_winner[2]:+.6f}; "
            f"held-out gain={train_winner[3]:+.6f}"
        )
        print(
            f"  z=ff rank={ff_train_rank}/256 on training, "
            f"{ff_test_rank}/256 held out; "
            f"gain={ff[2]:+.6f}/{ff[3]:+.6f}"
        )
        print(
            "  z=ff per-round sample, training gain, held-out gain:",
            [
                (r, sample, round(train_gain, 6), round(test_gain, 6))
                for r, (train_gain, test_gain, _, _, sample) in enumerate(ff[1], 1)
            ],
        )
        print(
            "  top held-out candidates:",
            [(f"0x{x[0]:02x}", round(x[3], 6)) for x in by_test[:8]],
        )

    if training_winners[0] == training_winners[1]:
        selected_z = training_winners[0]
        selected_rk10 = missing ^ SBOX[selected_z]
        print("Task 3 candidate agreed across both training halves:")
        print(f"  z_missing=0x{selected_z:02x}")
        print(f"  actual RK10 candidate={selected_rk10.tobytes().hex()}")
        save_endpoint_correlation_plots(
            root, endpoint_correlations, selected_rk10, event_samples[0], selected_z
        )
    else:
        print("its over")


if __name__ == "__main__":
    main()
