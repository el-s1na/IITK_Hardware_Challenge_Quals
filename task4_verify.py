#!/usr/bin/env python3
"""Verify both histogram endpoint candidates against the raw CSV.

The FIPS key is an external anchor, not something uniquely selected by the
histogram: the identity-echo histogram also permits the alternate z=0 branch.
This script makes that dependency explicit and checks the z=255 result against
the raw ciphertext column, an independent key expansion, and the AES known-answer
vector.
"""
import csv
from pathlib import Path

import numpy as np
from Crypto.Cipher import AES


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
MASTER = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")


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


def main():
    root = Path(__file__).resolve().parent
    counts = np.zeros((16, 256), dtype=np.int64)
    rows = 0
    with (root / "Round10_traces.csv").open(newline="") as f:
        for row in csv.reader(f):
            block = bytes.fromhex(row[0].strip().strip('"'))
            if len(block) != 16:
                raise ValueError(f"row {rows} has {len(block)} ciphertext bytes")
            for pos, value in enumerate(block):
                counts[pos, value] += 1
            rows += 1

    missing = counts.argmin(axis=1).astype(np.uint8)
    doubled = counts.argmax(axis=1).astype(np.uint8)
    fake = np.load(root / "results" / "stage1.npz")["fake_key"].astype(np.uint8)
    rk10 = expand_aes128(MASTER)[10]
    rk10_arr = np.frombuffer(rk10, dtype=np.uint8)
    inv_sbox = np.argsort(SBOX).astype(np.uint8)
    inferred_z = inv_sbox[missing ^ rk10_arr]
    alternate_rk10 = (missing ^ 0x63).tobytes()
    alternate_z = inv_sbox[missing ^ np.frombuffer(alternate_rk10, dtype=np.uint8)]
    alternate_master = invert_aes128_rk10(alternate_rk10)
    shift_rk = np.frombuffer(rk10, dtype=np.uint8)
    skip_rk = np.frombuffer(alternate_rk10, dtype=np.uint8)
    shift_tables, skip_tables = [], []
    for b in range(16):
        # Same identity default and same untouched address slot(0); different
        # address/value pairing rules leave opposite output endpoints absent.
        shifted = np.arange(256, dtype=np.uint8)
        for z in range(255):
            shifted[int(SBOX[z + 1] ^ fake[b])] = SBOX[z] ^ shift_rk[b]
        skipped = np.arange(256, dtype=np.uint8)
        for z in range(1, 256):
            skipped[int(SBOX[z] ^ fake[b])] = SBOX[z] ^ skip_rk[b]
        shift_tables.append(np.bincount(shifted, minlength=256))
        skip_tables.append(np.bincount(skipped, minlength=256))
    shift_tables = np.stack(shift_tables)
    skip_tables = np.stack(skip_tables)

    assert rows == 100_000
    assert np.all(counts[np.arange(16), missing] == 0)
    assert np.all(doubled ^ fake == 0x63)
    assert np.all(inferred_z == 255)
    assert np.all(alternate_z == 0)
    assert expand_aes128(alternate_master)[10] == alternate_rk10
    expected_occupancy = np.ones((16, 256), dtype=np.int64)
    expected_occupancy[np.arange(16), doubled] = 2
    expected_occupancy[np.arange(16), missing] = 0
    assert np.array_equal(shift_tables, expected_occupancy)
    assert np.array_equal(skip_tables, expected_occupancy)
    assert AES.new(MASTER, AES.MODE_ECB).encrypt(
        bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    ).hex() == "3925841d02dc09fbdc118597196a0b32"

    print(f"raw CSV rows: {rows}")
    print(f"FIPS AES-128 example master key: {MASTER.hex()}")
    print(f"FIPS example candidate RK10: {rk10.hex()}")
    print(f"missing xor FIPS candidate RK10: {(missing ^ rk10_arr).tobytes().hex()}")
    print(f"InvSBox(missing xor FIPS candidate RK10): {set(map(int, inferred_z))}")
    print(f"doubled xor CPA fake key: {(doubled ^ fake).tobytes().hex()}")
    print(f"alternate z=0 candidate RK10: {alternate_rk10.hex()}")
    print(f"alternate candidate master: {alternate_master.hex()}")
    print("identity-init shifted-slot model predicts observed hole/echo: PASS")
    print("identity-init skip-first model predicts same hole/echo: PASS")
    print("FIPS AES known-answer vector: PASS")
    print("Evidence boundary: FIPS candidate has z=255, but histogram also permits z=0.")


if __name__ == "__main__":
    main()
