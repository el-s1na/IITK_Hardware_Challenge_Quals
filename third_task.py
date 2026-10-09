#!/usr/bin/env python3
"""Task 3: derive timing from trace activity and scan round-10 key candidates.

First, it detects activity samples from across-trace variance without key
guesses or a preset sample spacing. It then scans all missing-endpoint key
candidates using transition HD, HW(S[r-1] XOR S[r]), at those detected samples.
The result assumes the ten strongest local variance peaks correspond to AES
round events and depends on the shared missing-endpoint and leakage hypotheses.
Run fourth_task.py with the printed RK10 to derive the master key.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,1,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    4,0xc7,0x23,0xc3,0x18,0x96,5,0x9a,7,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    9,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,2,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,6,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,8,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,3,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)
RCON = (1, 2, 4, 8, 16, 32, 64, 128, 0x1B, 0x36)


def expand_aes128(key):
    words = [int.from_bytes(key[i:i + 4], "big") for i in range(0, 16, 4)]
    for i in range(4, 44):
        t = words[i - 1]
        if i % 4 == 0:
            t = ((t << 8) & 0xFFFFFFFF) | (t >> 24)
            t = sum(int(SBOX[(t >> (24 - 8*j)) & 0xFF]) << (24 - 8*j)
                    for j in range(4)) ^ (RCON[i // 4 - 1] << 24)
        words.append(words[i - 4] ^ t)
    return [b"".join(words[4*r + j].to_bytes(4, "big") for j in range(4))
            for r in range(11)]


def invert_aes128_rk10(round_key):
    words = [None] * 44
    for j in range(4):
        words[40 + j] = int.from_bytes(round_key[4*j:4*j + 4], "big")
    for i in range(43, 3, -1):
        t = words[i - 1]
        if i % 4 == 0:
            t = ((t << 8) & 0xFFFFFFFF) | (t >> 24)
            t = sum(int(SBOX[(t >> (24 - 8*j)) & 0xFF]) << (24 - 8*j)
                    for j in range(4)) ^ (RCON[i // 4 - 1] << 24)
        words[i - 4] = words[i] ^ t
    return b"".join(words[i].to_bytes(4, "big") for i in range(4))


INV_SBOX = np.argsort(SBOX).astype(np.uint8)
HW = np.array([x.bit_count() for x in range(256)], dtype=np.float64)
INV_MC = ((14, 11, 13, 9), (9, 14, 11, 13),
          (13, 9, 14, 11), (11, 13, 9, 14))
OUT_FOR_PRE_SHIFT = np.array([
    4 * ((c - r) % 4) + r for c in range(4) for r in range(4)
])
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
        tables[factor] = np.array([gf_mul(x, factor) for x in range(256)], dtype=np.uint8)
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
    peaks = np.flatnonzero(
        (variance[1:-1] >= variance[:-2])
        & (variance[1:-1] > variance[2:])
    ) + 1
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
    return (float(gtr), float(gte), float((ytrc * ytrc).sum()),
            float((ytec * ytec).sum()))


def correlation_by_sample(features, traces):
    """Pearson correlation of each byte feature with every trace sample."""
    features = features - features.mean(axis=0)
    traces = traces - traces.mean(axis=0)
    numerator = features.T @ traces
    denominator = np.sqrt(
        (features * features).sum(axis=0)[:, None]
        * (traces * traces).sum(axis=0)[None, :]
    )
    return numerator / np.maximum(denominator, 1e-20)


def save_round_key_correlation_plots(root, rk10, ciphertexts, traces,
                                     heldout_folds, event_samples):
    """Save one actual-key transition-correlation plot per RK10 byte."""
    master = invert_aes128_rk10(rk10.tobytes())
    round_keys = [np.frombuffer(key, dtype=np.uint8)
                  for key in expand_aes128(master)]
    s9, s8 = invert_to_round8(ciphertexts, round_keys[10], round_keys[9])
    states = {9: s9, 8: s8}
    for round_number in range(8, 0, -1):
        states[round_number - 1] = inverse_round(
            states[round_number], round_keys[round_number]
        )

    correlations = {}
    for fold_number, heldout in enumerate(heldout_folds, 1):
        fold_correlations = []
        for round_number in range(1, 9):
            feature = HW[states[round_number - 1][heldout]
                         ^ states[round_number][heldout]]
            fold_correlations.append(
                correlation_by_sample(feature, traces[heldout])
            )
        correlations[fold_number] = np.stack(fold_correlations)

    output_dir = root / "actual_round10_correlation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = np.arange(traces.shape[1])
    for byte in range(16):
        fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True, sharey=True)
        for round_number, axis in enumerate(axes.flat, 1):
            for fold_number, style in ((1, "-"), (2, "--")):
                axis.plot(
                    sample_indices,
                    correlations[fold_number][round_number - 1, byte],
                    linestyle=style,
                    linewidth=0.9,
                    label=f"held-out fold {fold_number}",
                )
            axis.axvline(event_samples[0][round_number - 1],
                         color="black", alpha=0.35, linewidth=0.8)
            axis.axvline(event_samples[1][round_number - 1],
                         color="black", alpha=0.35, linewidth=0.8,
                         linestyle=":")
            axis.axhline(0, color="gray", linewidth=0.6)
            axis.set_title(f"Round {round_number}, event sample "
                           f"{event_samples[0][round_number - 1]}")
            axis.set_ylabel("Pearson r")
            axis.grid(alpha=0.2)
        for axis in axes[-1, :]:
            axis.set_xlabel("Trace sample")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"Actual RK10 transition-HW correlations — byte {byte:02d} "
            f"(RK10 byte 0x{rk10[byte]:02x})"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_dir / f"byte_{byte:02d}.png", dpi=150)
        plt.close(fig)
    print(f"Saved 16 actual-RK10 correlation plots to {output_dir}")


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
        print(f"{b:>4}  0x{missing[b]:02x}          0x{doubled[b]:02x}          "
              f"{counts[b, missing[b]]}/{counts[b, doubled[b]]}")
    print("The histogram identifies absent ciphertext outputs; the 256-way "
          "transition scan resolves the endpoint under its leakage model.")
    keep = np.all(ct_all != doubled[None, :], axis=1)
    row_ids = np.flatnonzero(keep)
    ct, traces = ct_all[keep], tr_all[keep]
    training_winners = []
    heldout_folds = []
    event_samples = []
    for fold, train_first in enumerate((True, False), 1):
        train = np.flatnonzero((row_ids < 50_000) == train_first)
        test = np.flatnonzero((row_ids < 50_000) != train_first)
        activity_samples, activity_variance = find_round_activity_samples(
            traces[train], rounds=10
        )
        print(f"fold {fold} round activity samples discovered from training "
              f"trace variation: {activity_samples.tolist()}")
        print("  across-trace variance at events:",
              [round(float(activity_variance[t]), 10) for t in activity_samples])
        # The first nine detected events are ordinary rounds 1..9. The tenth
        # is the protected final round, excluded from the ordinary-round HD fit.
        round_samples = activity_samples[:9]
        heldout_folds.append(test)
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
                gt, ge, et, ee = fit_gain(
                    h[train], y[train], h[test], y[test]
                )
                round_scores.append((gt, ge, et, ee, sample))
            train8 = sum(a * et for a, _, et, _, _ in round_scores[:8]) / sum(
                et for _, _, et, _, _ in round_scores[:8])
            test8 = sum(b * ee for _, b, _, ee, _ in round_scores[:8]) / sum(
                ee for _, _, _, ee, _ in round_scores[:8])
            candidate_results.append((m, round_scores, train8, test8))

        by_train = sorted(candidate_results, key=lambda x: x[2], reverse=True)
        by_test = sorted(candidate_results, key=lambda x: x[3], reverse=True)
        ff = next(r for r in candidate_results if r[0] == 0xFF)
        ff_train_rank = next(i + 1 for i, r in enumerate(by_train) if r[0] == 0xFF)
        ff_test_rank = next(i + 1 for i, r in enumerate(by_test) if r[0] == 0xFF)
        train_winner = by_train[0]
        training_winners.append(train_winner[0])
        print(f"fold {fold}: train/test={len(train)}/{len(test)}")
        print(f"  training-selected endpoint (rounds 1-8): "
              f"z=0x{train_winner[0]:02x}, gain={train_winner[2]:+.6f}; "
              f"held-out gain={train_winner[3]:+.6f}")
        print(f"  z=ff rank={ff_train_rank}/256 on training, "
              f"{ff_test_rank}/256 held out; "
              f"gain={ff[2]:+.6f}/{ff[3]:+.6f}")
        print("  z=ff per-round sample, training gain, held-out gain:", [
            (r, sample, round(train_gain, 6), round(test_gain, 6))
            for r, (train_gain, test_gain, _, _, sample) in enumerate(ff[1], 1)
        ])
        print("  top held-out candidates:", [
            (f"0x{x[0]:02x}", round(x[3], 6)) for x in by_test[:8]
        ])

    if training_winners[0] == training_winners[1]:
        selected_z = training_winners[0]
        selected_rk10 = missing ^ SBOX[selected_z]
        print("Task 3 candidate agreed across both training halves:")
        print(f"  z_missing=0x{selected_z:02x}")
        print(f"  actual RK10 candidate={selected_rk10.tobytes().hex()}")
        save_round_key_correlation_plots(
            root, selected_rk10, ct, traces, heldout_folds, event_samples
        )
        print("Task 4 next step:")
        print(f"  python fourth_task.py {selected_rk10.tobytes().hex()}")
    else:
        print("Task 3 has no single candidate agreed by both training halves; "
              "do not promote a round-10 key to Task 4 yet.")


if __name__ == "__main__":
    main()
