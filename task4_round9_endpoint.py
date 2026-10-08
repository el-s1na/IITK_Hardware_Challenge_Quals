#!/usr/bin/env python3
"""Exploratory HW(S8) scan over all 256 missing-z hypotheses.

For each common missing index z, derive RK10[b] = missing[b] XOR SBOX[z],
invert the AES-128 key schedule to RK9, and reconstruct the round-8 register
state S8 from ciphertexts that are not the doubled echo. The expected old
register value during the round-9 update is S8. This script tests HW(S8)
against samples 45-47, while regressing out a fake-key nuisance basis.

The state inversion is checked against a forward AES implementation on a
known test block. This script tests an HW(S8) association only; it does not
establish that the samples physically leak HW(S8), and its endpoint result is
not proof. See task4_round9_hd_endpoint.py for the direct state-register HD
model HW(S8 XOR S9).
"""
from pathlib import Path

import numpy as np

from task4_verify import SBOX, expand_aes128, invert_aes128_rk10


INV_SBOX = np.argsort(SBOX).astype(np.uint8)
HW = np.array([x.bit_count() for x in range(256)], dtype=np.float64)
INV_MC = ((14, 11, 13, 9), (9, 14, 11, 13),
          (13, 9, 14, 11), (11, 13, 9, 14))
OUT_FOR_PRE_SHIFT = np.array([
    4 * ((c - r) % 4) + r for c in range(4) for r in range(4)
])
INV_SR = np.array([4 * ((c + r) % 4) + r for c in range(4) for r in range(4)])


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


def forward_states(plaintext, master):
    """Small independent forward AES path used only to validate inversion."""
    keys = [np.frombuffer(k, dtype=np.uint8) for k in expand_aes128(master)]
    state = np.frombuffer(plaintext, dtype=np.uint8).copy() ^ keys[0]
    s8 = s9 = None
    for rnd in range(1, 10):
        state = SBOX[state]
        shifted = np.empty_like(state)
        for col in range(4):
            for row in range(4):
                shifted[4 * col + row] = state[4 * ((col + row) % 4) + row]
        state = shifted.copy()
        for col in range(4):
            a = state[4 * col:4 * col + 4].copy()
            state[4 * col + 0] = gf_mul(int(a[0]), 2) ^ gf_mul(int(a[1]), 3) ^ int(a[2]) ^ int(a[3])
            state[4 * col + 1] = int(a[0]) ^ gf_mul(int(a[1]), 2) ^ gf_mul(int(a[2]), 3) ^ int(a[3])
            state[4 * col + 2] = int(a[0]) ^ int(a[1]) ^ gf_mul(int(a[2]), 2) ^ gf_mul(int(a[3]), 3)
            state[4 * col + 3] = gf_mul(int(a[0]), 3) ^ int(a[1]) ^ int(a[2]) ^ gf_mul(int(a[3]), 2)
        state ^= keys[rnd]
        if rnd == 8:
            s8 = state.copy()
        if rnd == 9:
            s9 = state.copy()
    state = SBOX[state]
    shifted = np.empty_like(state)
    for col in range(4):
        for row in range(4):
            shifted[4 * col + row] = state[4 * ((col + row) % 4) + row]
    ciphertext = shifted ^ keys[10]
    return s8, s9, ciphertext


def score_gain(y, prediction):
    return float((2 * (y * prediction).sum() - (prediction * prediction).sum())
                 / (y * y).sum())


def main():
    root = Path(__file__).resolve().parent
    ct_all = np.load(root / "results" / "ct.npy").astype(np.uint8)
    traces_all = np.load(root / "results" / "traces.npy").astype(np.float64)
    fake = np.load(root / "results" / "stage1.npz")["fake_key"].astype(np.uint8)
    counts = np.stack([np.bincount(ct_all[:, b], minlength=256) for b in range(16)])
    missing = counts.argmin(axis=1).astype(np.uint8)
    doubled = counts.argmax(axis=1).astype(np.uint8)
    keep = np.all(ct_all != doubled[None, :], axis=1)
    row_ids = np.flatnonzero(keep)
    ct = ct_all[keep]
    traces = traces_all[keep]

    # Validate the inverse equations independently on a known test block.
    # An unrelated test-only key keeps the implementation check independent
    # of the endpoint candidates and any well-known AES example key.
    test_key = bytes.fromhex("00102030405060708090a0b0c0d0e0f0")
    test_plain = bytes.fromhex("ffeeddccbbaa99887766554433221100")
    test_s8, test_s9, test_ct = forward_states(test_plain, test_key)
    test_rks = [np.frombuffer(k, dtype=np.uint8) for k in expand_aes128(test_key)]
    check_s9, check_s8 = invert_to_round8(
        np.frombuffer(test_ct, dtype=np.uint8).reshape(1, 16), test_rks[10], test_rks[9]
    )
    assert np.array_equal(check_s9[0], test_s9)
    assert np.array_equal(check_s8[0], test_s8)

    # Regress a broad fake-key basis on the training half, then hold its
    # coefficients fixed on test rows.
    nuisance = []
    for b in range(16):
        q = INV_SBOX[ct[:, b] ^ fake[b]]
        nuisance.extend((
            HW[q],
            HW[ct[:, b] ^ fake[b]],
            HW[q ^ ct[:, INV_SR[b]]],
            HW[q ^ ct[:, b]],
        ))
    X = np.column_stack(nuisance + [np.ones(len(ct))])
    all_results = []

    for train_first in (True, False):
        train = np.flatnonzero((row_ids < 50_000) == train_first)
        test = np.flatnonzero((row_ids < 50_000) != train_first)
        x_pinv = np.linalg.pinv(X[train])
        y_train_raw = traces[train, 45:48]
        y_test_raw = traces[test, 45:48]
        beta_y = x_pinv @ y_train_raw
        y_train = y_train_raw - X[train] @ beta_y
        y_test = y_test_raw - X[test] @ beta_y
        denominator = float((y_test * y_test).sum())

        fold_results = []
        byte_scores = np.empty((256, 16), dtype=np.float64)
        test_predictions = {}
        for z_missing in range(256):
            rk10 = missing ^ SBOX[z_missing]
            master = invert_aes128_rk10(rk10.tobytes())
            round_keys = [np.frombuffer(k, dtype=np.uint8) for k in expand_aes128(master)]
            _, s8 = invert_to_round8(ct, round_keys[10], round_keys[9])
            H = HW[s8]
            beta_nuisance = x_pinv @ H[train]
            h_train = H[train] - X[train] @ beta_nuisance
            h_test = H[test] - X[test] @ beta_nuisance
            beta = np.linalg.lstsq(h_train, y_train, rcond=None)[0]
            pred_train = h_train @ beta
            pred_test = h_test @ beta
            train_score = score_gain(y_train, pred_train)
            test_score = score_gain(y_test, pred_test)
            fold_results.append((train_score, z_missing, test_score, rk10, master))
            test_predictions[z_missing] = pred_test

            # Independent single-byte held-out support for the selected
            # shared endpoint.
            numerator = (h_train[:, :, None] * y_train[:, None, :]).sum(axis=0)
            norm = (h_train * h_train).sum(axis=0)
            beta_byte = numerator / np.maximum(norm[:, None], 1e-20)
            pred_byte = h_test[:, :, None] * beta_byte[None, :, :]
            byte_scores[z_missing] = (
                2 * (y_test[:, None, :] * pred_byte).sum(axis=(0, 2))
                - (pred_byte * pred_byte).sum(axis=(0, 2))
            ) / denominator

        fold_results.sort(reverse=True, key=lambda x: x[0])
        all_results.append((fold_results, byte_scores, y_test, test_predictions))

    print(f"AES inversion self-check: PASS; rows used {len(ct)}/{len(ct_all)} "
          "(doubled-output rows excluded)")
    for fold_index, (results, byte_scores, y_test, test_predictions) in enumerate(all_results):
        selected = results[0]
        ff_rank = next(i + 1 for i, row in enumerate(results) if row[1] == 0xFF)
        zero_rank = next(i + 1 for i, row in enumerate(results) if row[1] == 0)
        ff = next(row for row in results if row[1] == 0xFF)
        zero = next(row for row in results if row[1] == 0)
        fd = next(row for row in results if row[1] == 0xFD)

        # Test the preselected z=255 prediction by shuffling held-out rows.
        pred_ff = test_predictions[0xFF]
        pred_zero = test_predictions[0]
        observed_delta = ff[2] - zero[2]
        rng = np.random.default_rng(991 + fold_index)
        null_ff = []
        null_delta = []
        for _ in range(1_000):
            y_perm = y_test[rng.permutation(len(y_test))]
            null_ff.append(score_gain(y_perm, pred_ff))
            null_delta.append(score_gain(y_perm, pred_ff) - score_gain(y_perm, pred_zero))
        p_ff = (1 + sum(x >= ff[2] for x in null_ff)) / (1 + len(null_ff))
        p_delta = (1 + sum(x >= observed_delta for x in null_delta)) / (1 + len(null_delta))

        ff_byte_ranks = [
            int(np.where(np.argsort(byte_scores[:, b])[::-1] == 0xFF)[0][0]) + 1
            for b in range(16)
        ]
        support = [b for b, rank in enumerate(ff_byte_ranks) if rank == 1]
        heldout_sorted = sorted(results, key=lambda row: row[2], reverse=True)
        ff_test_rank = next(i + 1 for i, row in enumerate(heldout_sorted)
                            if row[1] == 0xFF)
        zero_test_rank = next(i + 1 for i, row in enumerate(heldout_sorted)
                              if row[1] == 0)
        fd_test_rank = next(i + 1 for i, row in enumerate(heldout_sorted)
                            if row[1] == 0xFD)
        print(f"fold {fold_index + 1}: training selected z=0x{selected[1]:02x}; "
              f"z=ff train rank {ff_rank}/256, heldout rank {ff_test_rank}/256, "
              f"heldout gain={ff[2]:+.6f}, p={p_ff:.4f}; "
              f"z=0 train rank {zero_rank}/256, heldout rank {zero_test_rank}/256, "
              f"heldout gain={zero[2]:+.6f}; "
              f"paired zff-over-z0 p={p_delta:.4f}")
        print(f"  z=fd train rank {next(i + 1 for i, row in enumerate(results) if row[1] == 0xFD)}/256, "
              f"heldout rank {fd_test_rank}/256, train gain={fd[0]:+.6f}, "
              f"heldout gain={fd[2]:+.6f}")
        top = ", ".join(
            f"0x{row[1]:02x} ({row[0]:+.6f} train, {row[2]:+.6f} heldout)"
            for row in results[:5]
        )
        print(f"  top five endpoint candidates: {top}")
        top_test = ", ".join(
            f"0x{row[1]:02x} ({row[2]:+.6f})"
            for row in heldout_sorted[:5]
        )
        print(f"  top five held-out gains: {top_test}")
        print(f"  z=ff per-byte top-rank support on heldout: {support} "
              f"({len(support)}/16); ranks={ff_byte_ranks}")
        print(f"  z=ff RK10={ff[3].tobytes().hex()} master={ff[4].hex()}")

    print("Interpretation: this is an exploratory HW(S8) association only; "
          "it does not prove sample timing, physical leakage, or the missing endpoint.")


if __name__ == "__main__":
    main()
