import numpy as np
import pandas as pd
from pathlib import Path


CSV_PATH = Path("Round10_traces.csv")

N_TRACES = 100000

INV_SBOX = np.array(
  [
    0x52,
    0x09,
    0x6A,
    0xD5,
    0x30,
    0x36,
    0xA5,
    0x38,
    0xBF,
    0x40,
    0xA3,
    0x9E,
    0x81,
    0xF3,
    0xD7,
    0xFB,
    0x7C,
    0xE3,
    0x39,
    0x82,
    0x9B,
    0x2F,
    0xFF,
    0x87,
    0x34,
    0x8E,
    0x43,
    0x44,
    0xC4,
    0xDE,
    0xE9,
    0xCB,
    0x54,
    0x7B,
    0x94,
    0x32,
    0xA6,
    0xC2,
    0x23,
    0x3D,
    0xEE,
    0x4C,
    0x95,
    0x0B,
    0x42,
    0xFA,
    0xC3,
    0x4E,
    0x08,
    0x2E,
    0xA1,
    0x66,
    0x28,
    0xD9,
    0x24,
    0xB2,
    0x76,
    0x5B,
    0xA2,
    0x49,
    0x6D,
    0x8B,
    0xD1,
    0x25,
    0x72,
    0xF8,
    0xF6,
    0x64,
    0x86,
    0x68,
    0x98,
    0x16,
    0xD4,
    0xA4,
    0x5C,
    0xCC,
    0x5D,
    0x65,
    0xB6,
    0x92,
    0x6C,
    0x70,
    0x48,
    0x50,
    0xFD,
    0xED,
    0xB9,
    0xDA,
    0x5E,
    0x15,
    0x46,
    0x57,
    0xA7,
    0x8D,
    0x9D,
    0x84,
    0x90,
    0xD8,
    0xAB,
    0x00,
    0x8C,
    0xBC,
    0xD3,
    0x0A,
    0xF7,
    0xE4,
    0x58,
    0x05,
    0xB8,
    0xB3,
    0x45,
    0x06,
    0xD0,
    0x2C,
    0x1E,
    0x8F,
    0xCA,
    0x3F,
    0x0F,
    0x02,
    0xC1,
    0xAF,
    0xBD,
    0x03,
    0x01,
    0x13,
    0x8A,
    0x6B,
    0x3A,
    0x91,
    0x11,
    0x41,
    0x4F,
    0x67,
    0xDC,
    0xEA,
    0x97,
    0xF2,
    0xCF,
    0xCE,
    0xF0,
    0xB4,
    0xE6,
    0x73,
    0x96,
    0xAC,
    0x74,
    0x22,
    0xE7,
    0xAD,
    0x35,
    0x85,
    0xE2,
    0xF9,
    0x37,
    0xE8,
    0x1C,
    0x75,
    0xDF,
    0x6E,
    0x47,
    0xF1,
    0x1A,
    0x71,
    0x1D,
    0x29,
    0xC5,
    0x89,
    0x6F,
    0xB7,
    0x62,
    0x0E,
    0xAA,
    0x18,
    0xBE,
    0x1B,
    0xFC,
    0x56,
    0x3E,
    0x4B,
    0xC6,
    0xD2,
    0x79,
    0x20,
    0x9A,
    0xDB,
    0xC0,
    0xFE,
    0x78,
    0xCD,
    0x5A,
    0xF4,
    0x1F,
    0xDD,
    0xA8,
    0x33,
    0x88,
    0x07,
    0xC7,
    0x31,
    0xB1,
    0x12,
    0x10,
    0x59,
    0x27,
    0x80,
    0xEC,
    0x5F,
    0x60,
    0x51,
    0x7F,
    0xA9,
    0x19,
    0xB5,
    0x4A,
    0x0D,
    0x2D,
    0xE5,
    0x7A,
    0x9F,
    0x93,
    0xC9,
    0x9C,
    0xEF,
    0xA0,
    0xE0,
    0x3B,
    0x4D,
    0xAE,
    0x2A,
    0xF5,
    0xB0,
    0xC8,
    0xEB,
    0xBB,
    0x3C,
    0x83,
    0x53,
    0x99,
    0x61,
    0x17,
    0x2B,
    0x04,
    0x7E,
    0xBA,
    0x77,
    0xD6,
    0x26,
    0xE1,
    0x69,
    0x14,
    0x63,
    0x55,
    0x21,
    0x0C,
    0x7D,
  ],
  dtype=np.uint8,
)


# ============================================================
# AES ShiftRows inverse mapping
#
# Standard AES column-major byte numbering:
#
#       0  4  8 12
#       1  5  9 13
#       2  6 10 14
#       3  7 11 15
#
# C[INV_SR[i]] corresponds to the state byte before ShiftRows.
#
# IMPORTANT:
# The README gives INV_SR symbolically but does not print
# the array itself, so this assumes standard AES byte ordering.
# ============================================================

INV_SR = np.array(
  [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11], dtype=np.int64
)


# ============================================================
# Hamming-weight lookup table
# ============================================================

HW = np.array([x.bit_count() for x in range(256)], dtype=np.uint8)


# ============================================================
# Load CSV
# ============================================================


def load_dataset(csv_path: Path, n_traces=None):
  if not csv_path.exists():
    raise FileNotFoundError(
      f"Could not find {csv_path.resolve()}\n"
      f"Put Round10_traces.csv in the same directory as this script."
    )

  print(f"[+] Loading {csv_path} ...")

  df = pd.read_csv(csv_path, header=None)

  print(f"[+] Raw CSV shape: {df.shape}")

  if df.shape[1] != 201:
    raise ValueError(f"Expected 201 columns, got {df.shape[1]}")

  if n_traces is not None:
    df = df.iloc[:n_traces]

  # Column 0 = 32 hex characters = 16 bytes
  ciphertext_hex = df.iloc[:, 0].astype(str)

  ciphertexts = np.fromiter(
    (byte for value in ciphertext_hex for byte in bytes.fromhex(value)), dtype=np.uint8
  )

  ciphertexts = ciphertexts.reshape(len(df), 16)

  # Columns 1..200 = power samples
  traces = df.iloc[:, 1:].to_numpy(dtype=np.float64)

  print(f"[+] Ciphertexts shape: {ciphertexts.shape}")
  print(f"[+] Traces shape:      {traces.shape}")

  if traces.shape[1] != 200:
    raise ValueError(f"Expected 200 trace samples, got {traces.shape[1]}")

  return ciphertexts, traces


# ============================================================
# CPA for one byte
# ============================================================


def cpa_one_byte(ciphertexts, traces, byte_index):
  """
  Recover one fake-key byte using the challenge's model:

      HD_i[k] =
          HW(
              InvSBox[C[i] XOR k]
              XOR
              C[INV_SR[i]]
          )

  Returns:
      correlation : (256, 200)
      scores      : (256,)
      winner      : integer 0..255
      peak_sample : integer 0..199
  """

  n = len(ciphertexts)

  c_i = ciphertexts[:, byte_index]
  c_other = ciphertexts[:, INV_SR[byte_index]]

  # --------------------------------------------------------
  # Center traces once.
  #
  # traces_centered[:, t] =
  #       P[:,t] - mean(P[:,t])
  # --------------------------------------------------------

  traces_centered = traces - traces.mean(axis=0)

  # ||P_t - mean(P_t)|| for each sample t
  trace_norms = np.sqrt(np.sum(traces_centered * traces_centered, axis=0))

  correlation = np.zeros((256, traces.shape[1]), dtype=np.float64)

  for k in range(256):
    # ----------------------------------------------------
    # Step 1:
    #
    # C[i] XOR k
    # ----------------------------------------------------

    x = c_i ^ np.uint8(k)

    # ----------------------------------------------------
    # Step 2:
    #
    # InvSBox[C[i] XOR k]
    # ----------------------------------------------------

    inv_sbox_value = INV_SBOX[x]

    # ----------------------------------------------------
    # Step 3:
    #
    # InvSBox[...] XOR C[INV_SR[i]]
    # ----------------------------------------------------

    hd_input = inv_sbox_value ^ c_other

    # ----------------------------------------------------
    # Step 4:
    #
    # Hamming weight
    #
    # This gives one predicted leakage value per trace.
    # ----------------------------------------------------

    h = HW[hd_input].astype(np.float64)

    # Center hypothesis
    h_centered = h - h.mean()

    h_norm = np.sqrt(np.sum(h_centered * h_centered))

    if h_norm == 0:
      continue

    # ----------------------------------------------------
    # Step 5:
    #
    # Pearson correlation against ALL 200 samples.
    #
    # Numerator:
    #
    #       sum_j H_j * P_j(t)
    #
    # after centering.
    # ----------------------------------------------------

    numerator = h_centered @ traces_centered

    denominator = h_norm * trace_norms

    correlation[k] = numerator / denominator

  # The challenge explicitly says to use the highest
  # absolute correlation peak.
  scores = np.max(np.abs(correlation), axis=1)

  winner = int(np.argmax(scores))

  # Sample where winning key reaches maximum |correlation|
  peak_sample = int(np.argmax(np.abs(correlation[winner])))

  return correlation, scores, winner, peak_sample


# ============================================================
# Main attack
# ============================================================


def main():

  ciphertexts, traces = load_dataset(CSV_PATH, N_TRACES)

  fake_key = []

  print()
  print("=" * 70)
  print("Starting final-round CPA")
  print("=" * 70)

  for byte_index in range(16):
    print()
    print(f"[+] Attacking byte {byte_index}")

    correlation, scores, winner, peak_sample = cpa_one_byte(
      ciphertexts, traces, byte_index
    )

    fake_key.append(winner)

    print(f"    Winner       : 0x{winner:02X}")
    print(f"    Peak sample  : {peak_sample}")
    print(f"    Peak |corr|  : {scores[winner]:.6f}")

    # Show top 5 candidates
    best = np.argsort(scores)[::-1][:5]

    print("    Top candidates:")
    for rank, k in enumerate(best, start=1):
      peak = scores[k]
      sample = np.argmax(np.abs(correlation[k]))

      print(f"      #{rank}: 0x{k:02X} |corr|={peak:.6f} sample={sample}")

  # --------------------------------------------------------
  # Print recovered 16-byte fake key
  # --------------------------------------------------------

  fake_key_hex = "".join(f"{byte:02X}" for byte in fake_key)

  print()
  print("=" * 70)
  print("RECOVERED FAKE ROUND-10 KEY")
  print("=" * 70)
  print(fake_key_hex)
  print()

  print("Byte-wise:")
  for i, byte in enumerate(fake_key):
    print(f"  byte {i:2d}: 0x{byte:02X}")

  print()
  print("=" * 70)
  print("Done.")
  print("=" * 70)


if __name__ == "__main__":
  main()
