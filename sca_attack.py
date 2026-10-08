#!/usr/bin/env python3
"""Side-channel analysis of Round10_traces.csv (AES-128 fake-key countermeasure).

Stage 1: CPA on final-round state register with HD model from the readme
         -> converges on the FAKE round-10 key.
Stage 2: CPA sweep over candidates for the TRUE round-10 key using
         fake-key-aware leakage models -> actual round-10 key.
Outputs: results/*.npz, results/plots/*.png, keys printed to stdout.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/s1na/testing/Round10_traces.csv"
OUT = "/home/s1na/testing/results"
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

SBOX = np.array([
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16,
], dtype=np.uint8)
INV_SBOX = np.argsort(SBOX).astype(np.uint8)
HW = np.array([bin(x).count("1") for x in range(256)], dtype=np.int8)

# flat column-major index i = 4*col + row ; InvShiftRows: out(r,c) = in(r,(c+r)%4)
INV_SR = np.array([4 * ((c + r) % 4) + r for c in range(4) for r in range(4)])
SR = np.array([4 * ((c - r) % 4) + r for c in range(4) for r in range(4)])
assert np.array_equal(SR[INV_SR], np.arange(16))


def load_data():
    cache_tr = os.path.join(OUT, "traces.npy")
    cache_ct = os.path.join(OUT, "ct.npy")
    if os.path.exists(cache_tr) and os.path.exists(cache_ct):
        return np.load(cache_ct), np.load(cache_tr)
    print("Loading CSV ...")
    df = pd.read_csv(DATA, header=None, dtype={0: str})
    print("raw shape:", df.shape)
    ct = np.array([list(bytes.fromhex(h)) for h in df[0]], dtype=np.uint8)
    traces = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    np.save(cache_tr, traces)
    np.save(cache_ct, ct)
    return ct, traces


def cpa(hyp, traces, chunk=20000):
    """Pearson corr of each hypothesis column with each trace sample.
    hyp: (N, K) float64, traces: (N, S) float64 -> (S, K) corr."""
    N = traces.shape[0]
    hc = hyp - hyp.mean(axis=0)
    tm = traces.mean(axis=0)
    tc = traces - tm
    num = np.zeros((traces.shape[1], hyp.shape[1]))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        num += tc[s:e].T @ hc[s:e]
    den = np.sqrt((tc * tc).sum(axis=0))[:, None] * np.sqrt((hc * hc).sum(axis=0))[None, :]
    return num / den


def hd_hint_model(ct, byte, k):
    """HW(InvSBox[C[i]^k] ^ C[INV_SR[i]])  -- readme hint model, k: scalar or array."""
    k = np.atleast_1d(k)
    prev = INV_SBOX[ct[:, byte][:, None] ^ k[None, :]]      # (N, K)
    cur = ct[:, INV_SR[byte]][:, None]                       # (N, 1)
    return HW[prev ^ cur].astype(np.float64)


def winners_for_n(ct, traces, n, ks):
    """Winning candidate per byte using first n traces."""
    win = np.empty(16, dtype=np.int64)
    peaks = np.empty(16)
    for b in range(16):
        hyp = hd_hint_model(ct[:n], b, ks)
        corr = cpa(hyp, traces[:n])
        m = np.abs(corr).max(axis=0)          # (K,)
        win[b] = int(np.argmax(m))
        peaks[b] = m[win[b]]
    return win, peaks


def main():
    ct, traces = load_data()
    N, S = traces.shape
    print(f"traces: {N} x {S}, ciphertexts: {ct.shape}")
    ks = np.arange(256, dtype=np.uint8)

    # ---------------- Stage 1: hint-model CPA on all traces (fake key) ----------
    corr_all = np.empty((16, S, 256))
    fake_key = np.empty(16, dtype=np.uint8)
    fake_peak = np.empty(16)
    fake_peak_sample = np.empty(16, dtype=np.int64)
    for b in range(16):
        hyp = hd_hint_model(ct, b, ks)
        corr = cpa(hyp, traces)
        corr_all[b] = corr                      # (S, 256)
        m = np.abs(corr).max(axis=0)
        k = int(np.argmax(m))
        fake_key[b] = k
        fake_peak[b] = m[k]
        fake_peak_sample[b] = int(np.argmax(np.abs(corr[:, k])))
    print("\n[Stage 1] CPA (hint HD model) -> recovered key (per readme: FAKE key):")
    print("Fake RK10 =", fake_key.tobytes().hex())

    # convergence: min traces for full 16-byte recovery (first-N, then random subsets)
    Ns = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000,
          7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000]
    conv_first = []
    for n in Ns:
        win, _ = winners_for_n(ct, traces, n, ks)
        ok = np.array_equal(win, fake_key)
        conv_first.append((n, ok))
        print(f"  first-N n={n:>6}: {'ALL 16 CORRECT' if ok else f'{(win==fake_key).sum()}/16 bytes'}")
    min_first = next((n for n, ok in conv_first if ok), None)

    rng = np.random.default_rng(1)
    conv_rand = []
    for n in [n for n in Ns if n <= 20000]:
        oks = []
        for _ in range(5):
            idx = rng.choice(N, size=n, replace=False)
            idx.sort()
            win, _ = winners_for_n(ct[idx], traces[idx], n, ks)
            oks.append(np.array_equal(win, fake_key))
        conv_rand.append((n, sum(oks)))
        print(f"  random-N n={n:>6}: {sum(oks)}/5 subsets fully correct")
    min_rand = next((n for n, okc in conv_rand if okc == 5), None)

    np.savez(os.path.join(OUT, "stage1.npz"),
             fake_key=fake_key, fake_peak=fake_peak,
             fake_peak_sample=fake_peak_sample, corr_all=corr_all.astype(np.float32),
             Ns=np.array(Ns),
             conv_first=np.array([ok for _, ok in conv_first]),
             conv_rand=np.array([c for _, c in conv_rand]),
             min_first=min_first if min_first else -1,
             min_rand=min_rand if min_rand else -1)

    # 4x4 grid plot: all 256 candidates, winner highlighted
    fig, axes = plt.subplots(4, 4, figsize=(20, 12), sharex=True)
    for b, ax in enumerate(axes.flat):
        k = fake_key[b]
        others = np.delete(np.arange(256), k)
        ax.plot(np.abs(corr_all[b][:, others]), color="0.85", lw=0.3, zorder=1)
        ax.plot(np.abs(corr_all[b][:, k]), color="crimson", lw=1.0, zorder=2,
                label=f"k=0x{k:02X} peak={fake_peak[b]:.3f} @s{fake_peak_sample[b]}")
        ax.set_title(f"byte {b}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1.0)
    fig.suptitle("Stage 1 CPA — hint HD model — |corr| vs sample, winner per byte (fake key)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "stage1_fake_key_grid.png"), dpi=110)
    plt.close(fig)

    # leakage-location histogram
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(16), fake_peak_sample + 1, color="steelblue")
    ax.set_xticks(range(16)); ax.set_xlabel("key byte"); ax.set_ylabel("peak sample")
    ax.set_title("Stage 1: sample index of the correlation peak per key byte")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "stage1_peak_samples.png"), dpi=110)
    plt.close(fig)

    print(f"min traces (first-N): {min_first}, min traces (all 5 random subsets): {min_rand}")
    print("Stage 1 done.\n", flush=True)
    return ct, traces, fake_key


if __name__ == "__main__":
    main()
