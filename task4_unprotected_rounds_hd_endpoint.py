#!/usr/bin/env python3
"""Cross-validated key-endpoint scan using unprotected AES register transitions.

The README says the countermeasure affects only the final round. The supplied
hint gives a register Hamming-distance model for the final update. This script
extends that same old/new-register model to ordinary AES rounds 1..9:
HW(S[r-1] XOR S[r]). The event windows are spaced four trace samples apart,
anchored to the measured final-round CPA peak at samples 49-50.

This is still conditional on the event cadence corresponding to one round per
four trace samples; scores are reported per round and jointly so that assumption
can be inspected rather than hidden.
"""
from pathlib import Path
import numpy as np
from Crypto.Cipher import AES

from task4_round9_endpoint import (
    GF, HW, INV_MC, INV_SBOX, gf_mul, invert_to_round8,
)
from task4_verify import SBOX, expand_aes128, invert_aes128_rk10


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
    """Fit centered multivariate linear leakage on train; return gains on both."""
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


def forward_states(plaintext, master):
    """Independent forward AES path returning states S0..S9 and ciphertext."""
    keys = [np.frombuffer(k, dtype=np.uint8) for k in expand_aes128(master)]
    state = np.frombuffer(plaintext, dtype=np.uint8).copy() ^ keys[0]
    states = [state.copy()]
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
        states.append(state.copy())
    state = SBOX[state]
    shifted = np.empty_like(state)
    for col in range(4):
        for row in range(4):
            shifted[4 * col + row] = state[4 * ((col + row) % 4) + row]
    ciphertext = shifted ^ keys[10]
    return states, ciphertext


def main():
    root = Path(__file__).resolve().parent
    # Independently validate reconstruction of every round state.
    test_key = bytes.fromhex("00102030405060708090a0b0c0d0e0f0")
    test_pt = bytes.fromhex("ffeeddccbbaa99887766554433221100")
    test_states, test_ct = forward_states(test_pt, test_key)
    test_rks = [np.frombuffer(k, dtype=np.uint8) for k in expand_aes128(test_key)]
    test_s9, test_s8 = invert_to_round8(test_ct[None, :], test_rks[10], test_rks[9])
    recovered = {9: test_s9[0], 8: test_s8[0]}
    for r in range(8, 0, -1):
        recovered[r - 1] = inverse_round(recovered[r][None, :], test_rks[r])[0]
    for r in range(10):
        assert np.array_equal(recovered[r], test_states[r]), f"round state S{r} mismatch"
    assert test_ct.tobytes() == AES.new(test_key, AES.MODE_ECB).encrypt(test_pt)
    print("AES round-state inversion self-check: PASS (S0..S9 and ciphertext)")

    ct_all = np.load(root / "results" / "ct.npy").astype(np.uint8)
    tr_all = np.load(root / "results" / "traces.npy").astype(np.float64)
    counts = np.stack([np.bincount(ct_all[:, b], minlength=256) for b in range(16)])
    missing = counts.argmin(axis=1).astype(np.uint8)
    doubled = counts.argmax(axis=1).astype(np.uint8)
    keep = np.all(ct_all != doubled[None, :], axis=1)
    row_ids = np.flatnonzero(keep)
    ct, traces = ct_all[keep], tr_all[keep]
    # Round 10's measured peak anchors the event grid. Round r is four samples
    # earlier for every decrement in round number; use the three-sample burst.
    windows = {r: slice(49 - 4 * (10 - r), 52 - 4 * (10 - r))
               for r in range(1, 10)}

    for fold, train_first in enumerate((True, False), 1):
        train = np.flatnonzero((row_ids < 50_000) == train_first)
        test = np.flatnonzero((row_ids < 50_000) != train_first)
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
                y = traces[:, windows[r]]
                gt, ge, et, ee = fit_gain(h[train], y[train], h[test], y[test])
                round_scores.append((gt, ge, et, ee))
            # Sum weighted reductions in SSE across independently fitted events.
            train_gain = sum(a * et for a, _, et, _ in round_scores) / sum(
                et for _, _, et, _ in round_scores)
            test_gain = sum(b * ee for _, b, _, ee in round_scores) / sum(
                ee for _, _, _, ee in round_scores)
            train8 = sum(a * et for a, _, et, _ in round_scores[:8]) / sum(
                et for _, _, et, _ in round_scores[:8])
            test8 = sum(b * ee for _, b, _, ee in round_scores[:8]) / sum(
                ee for _, _, _, ee in round_scores[:8])
            candidate_results.append((train_gain, m, test_gain, round_scores,
                                      train8, test8))

        by_train = sorted(candidate_results, key=lambda x: x[0], reverse=True)
        by_test = sorted(candidate_results, key=lambda x: x[2], reverse=True)
        ff = next(r for r in candidate_results if r[1] == 0xFF)
        ff_train_rank = next(i + 1 for i, r in enumerate(by_train) if r[1] == 0xFF)
        ff_test_rank = next(i + 1 for i, r in enumerate(by_test) if r[1] == 0xFF)
        by_train8 = sorted(candidate_results, key=lambda x: x[4], reverse=True)
        by_test8 = sorted(candidate_results, key=lambda x: x[5], reverse=True)
        ff_train8_rank = next(i + 1 for i, r in enumerate(by_train8) if r[1] == 0xFF)
        ff_test8_rank = next(i + 1 for i, r in enumerate(by_test8) if r[1] == 0xFF)
        print(f"fold {fold}: train/test={len(train)}/{len(test)}")
        print(f"  z=ff train rank={ff_train_rank}/256, heldout rank={ff_test_rank}/256, "
              f"joint gain={ff[0]:+.6f}/{ff[2]:+.6f}")
        print(f"  rounds 1-8 only: train rank={ff_train8_rank}/256, "
              f"heldout rank={ff_test8_rank}/256, "
              f"gain={ff[4]:+.6f}/{ff[5]:+.6f}")
        print("  ff per-round train/heldout gains:", [
            (r, round(a, 6), round(b, 6)) for r, (a, b, _, _) in enumerate(ff[3], 1)
        ])
        print("  top train candidates:", [(f"0x{x[1]:02x}", round(x[0], 6), round(x[2], 6))
                                        for x in by_train[:8]])
        print("  top heldout candidates:", [(f"0x{x[1]:02x}", round(x[2], 6))
                                           for x in by_test[:8]])
        print("  top heldout candidates rounds 1-8:", [
            (f"0x{x[1]:02x}", round(x[5], 6)) for x in by_test8[:8]
        ])


if __name__ == "__main__":
    main()
