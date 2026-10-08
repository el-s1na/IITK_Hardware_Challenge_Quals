import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "Round10_traces.csv"

# Trace counts used for characterization.
# IMPORTANT:
# Every one of these N values is now evaluated over ALL 200
# samples, not just samples 49..50.
TRACE_COUNTS = [
  250,
  500,
  750,
  1000,
  1250,
  1500,
  1750,
  2000,
  2500,
  3000,
  3500,
  4000,
  5000,
  7500,
  10000,
  15000,
  20000,
  50000,
  100000,
]

# Randomization makes the trace-count experiment reproducible.
# N traces are selected as the first N traces of this fixed
# random permutation.
RANDOM_SEED = 0x5CA

# Output directories/files
FULL_PLOT_DIR = "correlation_plots_full"
TRACE_CHAR_CSV = "trace_characterisation.csv"
RANK_PLOT = "rank_plot_all_bytes.png"
WORST_RANK_PLOT = "worst_rank_plot.png"

os.makedirs(FULL_PLOT_DIR, exist_ok=True)


# ============================================================
# AES INVERSE S-BOX
# ============================================================

AES_INV_SBOX = np.array(
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
# HAMMING WEIGHT TABLE
# ============================================================

HW = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


# ============================================================
# AES BYTE MAPPING
# ============================================================
#
# This is the same mapping used in the previous successful
# fake-key recovery script.
#
# The challenge defines:
#
#   HD_i[k] =
#       HW(
#          InvSBox[C[i] xor k]
#          xor C[INV_SR[i]]
#       )
#
# IMPORTANT:
# Keep this mapping unchanged because the earlier attack
# converged to:
#
#   91DEC068E3FACE4599CE38F420863A30
#
# ============================================================

INV_SR = np.array(
  [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11], dtype=np.int64
)


def load_data(filename):

  print(f"[+] Loading {filename} ...")

  df = pd.read_csv(filename, header=None)

  print(f"[+] Raw CSV shape: {df.shape}")

  if df.shape[1] != 201:
    raise ValueError(f"Expected 201 columns, got {df.shape[1]}")

  # ========================================================
  # Column 0 = ciphertext
  #
  # The CSV may contain ciphertext as:
  #
  #   1. hexadecimal strings
  #   2. bytes objects
  #
  # Handle both cases.
  # ========================================================

  ciphertext_column = df.iloc[:, 0].to_numpy()

  ciphertext_list = []

  for x in ciphertext_column:
    # -----------------------------------------------
    # Case 1: already bytes
    # -----------------------------------------------

    if isinstance(x, bytes):
      raw = x

    # -----------------------------------------------
    # Case 2: Python bytearray
    # -----------------------------------------------

    elif isinstance(x, bytearray):
      raw = bytes(x)

    # -----------------------------------------------
    # Case 3: hexadecimal string
    # -----------------------------------------------

    elif isinstance(x, str):
      x = x.strip()

      try:
        raw = bytes.fromhex(x)

      except ValueError as e:
        raise ValueError(f"Could not parse ciphertext value: {repr(x)}") from e

    else:
      raise TypeError(f"Unexpected ciphertext type: {type(x)} value={repr(x)}")

    # -----------------------------------------------
    # AES ciphertext must be exactly 16 bytes
    # -----------------------------------------------

    if len(raw) != 16:
      raise ValueError(
        f"Expected 16-byte ciphertext, got {len(raw)} bytes: {repr(raw)}"
      )

    ciphertext_list.append(raw)

  ciphertexts = np.frombuffer(b"".join(ciphertext_list), dtype=np.uint8).reshape(-1, 16)

  # ========================================================
  # Columns 1..200 = power traces
  # ========================================================

  traces = df.iloc[:, 1:].to_numpy(dtype=np.float32)

  print(f"[+] Ciphertexts shape: {ciphertexts.shape}")

  print(f"[+] Traces shape:      {traces.shape}")

  if ciphertexts.shape[1] != 16:
    raise ValueError(f"Expected 16 ciphertext bytes, got {ciphertexts.shape[1]}")

  if traces.shape[1] != 200:
    raise ValueError(f"Expected 200 trace samples, got {traces.shape[1]}")

  print()
  print(f"[+] Total traces loaded: {len(traces)}")
  print()

  return ciphertexts, traces


# ============================================================
# BUILD HYPOTHESIS MATRIX
# ============================================================


def build_hypothesis(ciphertexts, byte_index):
  """
  Build:

      H[j,k] =
          HW(
              InvSBox[C_j[i] xor k]
              xor C_j[INV_SR[i]]
          )

  Shape:

      N x 256
  """

  n = ciphertexts.shape[0]

  candidates = np.arange(256, dtype=np.uint8)

  # --------------------------------------------------------
  # C[i] for every trace
  # --------------------------------------------------------

  c_i = ciphertexts[:, byte_index]

  # Shape:
  #
  #   (N,1) xor (256,)
  #
  # gives:
  #
  #   (N,256)
  #
  x = np.bitwise_xor(c_i[:, None], candidates[None, :])

  # --------------------------------------------------------
  # InvSBox[C[i] xor k]
  # --------------------------------------------------------

  inv_sbox_value = AES_INV_SBOX[x]

  # --------------------------------------------------------
  # C[INV_SR[i]]
  # --------------------------------------------------------

  reference_byte = ciphertexts[:, INV_SR[byte_index]]

  # --------------------------------------------------------
  # HD hypothesis
  # --------------------------------------------------------

  h = HW[np.bitwise_xor(inv_sbox_value, reference_byte[:, None])]

  return h.astype(np.float32)


# ============================================================
# CPA
# ============================================================


def calculate_cpa_scores(ciphertexts, traces, byte_index):
  """
  Calculate Pearson correlation for:

      256 key candidates x 200 samples

  Returns:

      correlations : (256,200)
      scores       : (256,)
      peak_samples  : (256,)
  """

  n = len(traces)

  if n < 2:
    raise ValueError("Need at least two traces.")

  # --------------------------------------------------------
  # Build hypothesis
  # --------------------------------------------------------

  hypothesis = build_hypothesis(ciphertexts, byte_index)

  # --------------------------------------------------------
  # Trace statistics
  # --------------------------------------------------------

  traces = np.asarray(traces, dtype=np.float32)

  sum_t = traces.sum(axis=0, dtype=np.float64)

  sum_t2 = np.sum(traces * traces, axis=0, dtype=np.float64)

  # --------------------------------------------------------
  # Hypothesis statistics
  # --------------------------------------------------------

  sum_h = hypothesis.sum(axis=0, dtype=np.float64)

  sum_h2 = np.sum(hypothesis * hypothesis, axis=0, dtype=np.float64)

  # --------------------------------------------------------
  # Cross-product:
  #
  #      H^T T
  #
  # Result:
  #
  #      256 x 200
  # --------------------------------------------------------

  cross = hypothesis.T @ traces

  cross = cross.astype(np.float64, copy=False)

  # --------------------------------------------------------
  # Covariance numerator:
  #
  # sum(H*T) - sum(H)sum(T)/N
  # --------------------------------------------------------

  covariance = cross - np.outer(sum_h, sum_t) / n

  # --------------------------------------------------------
  # Variance terms
  # --------------------------------------------------------

  var_h = sum_h2 - (sum_h * sum_h) / n

  var_t = sum_t2 - (sum_t * sum_t) / n

  # Numerical safety
  var_h = np.maximum(var_h, 1e-15)

  var_t = np.maximum(var_t, 1e-15)

  # --------------------------------------------------------
  # Pearson correlation
  # --------------------------------------------------------

  denominator = np.sqrt(var_h[:, None] * var_t[None, :])

  correlations = covariance / denominator

  # --------------------------------------------------------
  # Score for candidate k:
  #
  #      max_t |rho(k,t)|
  # --------------------------------------------------------

  absolute_corr = np.abs(correlations)

  scores = absolute_corr.max(axis=1)

  peak_samples = absolute_corr.argmax(axis=1)

  return (correlations, scores, peak_samples)


# ============================================================
# PRINT TOP CANDIDATES
# ============================================================


def print_top_candidates(scores, peak_samples, number=5):

  order = np.argsort(scores)[::-1]

  print("    Top candidates:")

  for rank, candidate in enumerate(order[:number], start=1):
    print(
      f"      #{rank}: "
      f"0x{candidate:02X} "
      f"|corr|={scores[candidate]:.6f} "
      f"sample={peak_samples[candidate]}"
    )


# ============================================================
# FULL-DATA CPA
# ============================================================


def full_data_attack(ciphertexts, traces):

  print("=" * 70)
  print("FULL-DATA FINAL-ROUND CPA")
  print("=" * 70)
  print()

  recovered_key = []

  all_correlations = []
  all_scores = []
  all_peaks = []

  for byte_index in range(16):
    print(f"[+] Attacking byte {byte_index}")

    (correlations, scores, peak_samples) = calculate_cpa_scores(
      ciphertexts, traces, byte_index
    )

    winner = int(np.argmax(scores))

    peak_sample = int(peak_samples[winner])

    peak_corr = float(scores[winner])

    print(f"    Winner       : 0x{winner:02X}")

    print(f"    Peak sample  : {peak_sample}")

    print(f"    Peak |corr|  : {peak_corr:.6f}")

    print_top_candidates(scores, peak_samples)

    # ----------------------------------------------------
    # Save full correlation plot
    # ----------------------------------------------------

    plt.figure(figsize=(12, 6))

    for candidate in range(256):
      if candidate == winner:
        continue

      plt.plot(np.abs(correlations[candidate]), linewidth=0.7, alpha=0.25)

    # Highlight winning candidate
    plt.plot(
      np.abs(correlations[winner]), linewidth=2.2, label=f"Winner 0x{winner:02X}"
    )

    plt.xlabel("Sample")

    plt.ylabel("|Pearson correlation|")

    plt.title(f"Byte {byte_index:02d} - 256 CPA candidates")

    plt.legend()

    plt.grid(True, alpha=0.25)

    plt.tight_layout()

    filename = os.path.join(FULL_PLOT_DIR, f"byte_{byte_index:02d}.png")

    plt.savefig(filename, dpi=150)

    plt.close()

    print(f"    saved {filename}")

    print()

    recovered_key.append(winner)

    all_correlations.append(correlations)

    all_scores.append(scores)

    all_peaks.append(peak_samples)

  recovered_key = np.array(recovered_key, dtype=np.uint8)

  return (recovered_key, all_correlations, np.array(all_scores), np.array(all_peaks))


# ============================================================
# TRACE-COUNT ATTACK
# ============================================================


def characterize_trace_count(ciphertexts, traces, reference_key):

  print("=" * 70)
  print("TRACE-COUNT CHARACTERISATION")
  print("=" * 70)
  print()

  print("[+] IMPORTANT: Each trace count below uses ALL 200 samples.")

  print()

  n_total = len(traces)

  # --------------------------------------------------------
  # Deterministic random permutation
  # --------------------------------------------------------

  rng = np.random.default_rng(RANDOM_SEED)

  permutation = rng.permutation(n_total)

  shuffled_ciphertexts = ciphertexts[permutation]

  shuffled_traces = traces[permutation]

  results = []

  rank_matrix = []

  winner_matrix = []

  first_full_recovery = None

  # --------------------------------------------------------
  # Check only N values which exist
  # --------------------------------------------------------

  valid_trace_counts = [n for n in TRACE_COUNTS if n <= n_total]

  for N in valid_trace_counts:
    print("=" * 50)
    print(f"[+] Characterising N = {N}")

    ct_N = shuffled_ciphertexts[:N]

    traces_N = shuffled_traces[:N]

    winners = np.zeros(16, dtype=np.uint8)

    ranks = np.zeros(16, dtype=np.int32)

    candidate_scores = np.zeros((16, 256), dtype=np.float64)

    # ----------------------------------------------------
    # Attack all 16 bytes
    # ----------------------------------------------------

    for byte_index in range(16):
      (_correlations, scores, _peak_samples) = calculate_cpa_scores(
        ct_N, traces_N, byte_index
      )

      candidate_scores[byte_index] = scores

      winner = int(np.argmax(scores))

      winners[byte_index] = winner

      # ------------------------------------------------
      # Rank of the reference key byte
      #
      # Rank 1 = best
      #
      # Rank =
      #
      #   1 + number of candidates
      #       with a strictly larger score
      # ------------------------------------------------

      reference_byte = int(reference_key[byte_index])

      reference_score = scores[reference_byte]

      rank = 1 + int(np.count_nonzero(scores > reference_score))

      ranks[byte_index] = rank

    # ----------------------------------------------------
    # Full recovery criterion
    # ----------------------------------------------------

    worst_rank = int(np.max(ranks))

    full_recovery = worst_rank == 1

    winner_hex = "".join(f"{x:02X}" for x in winners)

    print(f"    Winner = {winner_hex}")

    print(f"    Worst rank = {worst_rank}")

    print(f"    Full recovery = {full_recovery}")

    # ----------------------------------------------------
    # Save result
    # ----------------------------------------------------

    results.append({"N": N, "worst_rank": worst_rank, "full_recovery": full_recovery})

    rank_matrix.append(ranks.copy())

    winner_matrix.append(winners.copy())

    # ----------------------------------------------------
    # First successful point
    # ----------------------------------------------------

    if full_recovery and first_full_recovery is None:
      first_full_recovery = N

  # ========================================================
  # Convert results
  # ========================================================

  results_df = pd.DataFrame(results)

  rank_matrix = np.array(rank_matrix)

  winner_matrix = np.array(winner_matrix)

  # ========================================================
  # STABLE FULL RECOVERY
  # ========================================================
  #
  # Stable means:
  #
  #   from this tested N onward,
  #   every tested N also has WorstRank == 1
  #
  # This is deliberately based only on the tested
  # trace-count values.
  # ========================================================

  stable_full_recovery = None

  for i in range(len(results_df)):
    remaining = results_df.iloc[i:]

    if np.all(remaining["full_recovery"].to_numpy()):
      stable_full_recovery = int(results_df.iloc[i]["N"])
      break

  # ========================================================
  # PRINT SUMMARY TABLE
  # ========================================================

  print()
  print("=" * 70)
  print("TRACE COUNT / RANK RESULTS")
  print("=" * 70)

  print(results_df.to_string(index=False))

  print()

  if first_full_recovery is not None:
    print(f"[+] First full-recovery among tested N: {first_full_recovery} traces")

  else:
    print("[!] No full recovery among tested trace counts.")

  if stable_full_recovery is not None:
    print(f"[+] Stable full-recovery among tested N: {stable_full_recovery} traces")

  else:
    print("[!] No stable full recovery among tested trace counts.")

  # ========================================================
  # SAVE CSV
  # ========================================================

  results_df.to_csv(TRACE_CHAR_CSV, index=False)

  print(f"[+] Saved {TRACE_CHAR_CSV}")

  # ========================================================
  # RANK PLOT: ALL BYTES
  # ========================================================

  plt.figure(figsize=(12, 7))

  Ns = results_df["N"].to_numpy()

  for byte_index in range(16):
    plt.plot(
      Ns,
      rank_matrix[:, byte_index],
      marker="o",
      linewidth=1.2,
      markersize=4,
      label=f"Byte {byte_index}",
    )

  plt.xlabel("Number of traces")

  plt.ylabel("Rank of reference key byte")

  plt.title("Fake-key CPA rank vs trace count")

  plt.yscale("log")

  plt.yticks([1, 2, 4, 8, 16, 32, 64, 128, 256], [1, 2, 4, 8, 16, 32, 64, 128, 256])

  plt.grid(True, which="both", alpha=0.25)

  plt.legend(ncol=2, fontsize=8)

  plt.tight_layout()

  plt.savefig(RANK_PLOT, dpi=150)

  plt.close()

  print(f"[+] Saved {RANK_PLOT}")

  # ========================================================
  # WORST-RANK PLOT
  # ========================================================

  plt.figure(figsize=(10, 6))

  worst_ranks = results_df["worst_rank"].to_numpy()

  plt.plot(Ns, worst_ranks, marker="o", linewidth=2)

  plt.axhline(1, linestyle="--", linewidth=1.5, label="Full recovery (rank = 1)")

  plt.xlabel("Number of traces")

  plt.ylabel("Worst rank across 16 bytes")

  plt.title("Worst key-byte rank vs trace count")

  plt.yscale("log")

  plt.yticks([1, 2, 4, 8, 16, 32, 64, 128, 256], [1, 2, 4, 8, 16, 32, 64, 128, 256])

  plt.grid(True, which="both", alpha=0.25)

  plt.legend()

  plt.tight_layout()

  plt.savefig(WORST_RANK_PLOT, dpi=150)

  plt.close()

  print(f"[+] Saved {WORST_RANK_PLOT}")

  return (
    results_df,
    rank_matrix,
    winner_matrix,
    first_full_recovery,
    stable_full_recovery,
  )


# ============================================================
# FINAL VERIFICATION
# ============================================================


def full_200_sample_verification(ciphertexts, traces, reference_key, verification_N):

  print()
  print("=" * 70)
  print(f"FULL 200-SAMPLE VERIFICATION AT N = {verification_N}")
  print("=" * 70)

  # --------------------------------------------------------
  # Use the SAME deterministic shuffled traces used in
  # characterization.
  # --------------------------------------------------------

  rng = np.random.default_rng(RANDOM_SEED)

  permutation = rng.permutation(len(traces))

  ct_N = ciphertexts[permutation[:verification_N]]

  traces_N = traces[permutation[:verification_N]]

  verified_key = []

  mismatches = []

  for byte_index in range(16):
    (_correlations, scores, peak_samples) = calculate_cpa_scores(
      ct_N, traces_N, byte_index
    )

    winner = int(np.argmax(scores))

    reference = int(reference_key[byte_index])

    peak = float(scores[winner])

    sample = int(peak_samples[winner])

    verified_key.append(winner)

    status = "OK" if winner == reference else "MISMATCH"

    if winner != reference:
      mismatches.append(byte_index)

    print(
      f"byte {byte_index:02d}: "
      f"reference=0x{reference:02X}, "
      f"winner=0x{winner:02X}, "
      f"peak={peak:.6f}, "
      f"sample={sample}, "
      f"{status}"
    )

  verified_key = np.array(verified_key, dtype=np.uint8)

  reference_hex = "".join(f"{x:02X}" for x in reference_key)

  verified_hex = "".join(f"{x:02X}" for x in verified_key)

  print()
  print(f"Reference key : {reference_hex}")

  print(f"Verified key  : {verified_hex}")

  if len(mismatches) == 0:
    print()
    print("[+] GLOBAL VERIFICATION PASSED")

    return True

  else:
    print()
    print("[!] GLOBAL VERIFICATION FAILED")

    print(f"[!] Mismatching bytes: {mismatches}")

    return False


# ============================================================
# MAIN
# ============================================================


def main():

  # --------------------------------------------------------
  # Load traces
  # --------------------------------------------------------

  ciphertexts, traces = load_data(CSV_FILE)

  # --------------------------------------------------------
  # FULL DATA CPA
  # --------------------------------------------------------

  (reference_key, all_correlations, all_scores, all_peaks) = full_data_attack(
    ciphertexts, traces
  )

  reference_hex = "".join(f"{x:02X}" for x in reference_key)

  print("=" * 70)
  print("REFERENCE FAKE ROUND-10 KEY")
  print("=" * 70)

  print(reference_hex)

  print("=" * 70)
  print()

  # --------------------------------------------------------
  # TRACE COUNT CHARACTERIZATION
  # --------------------------------------------------------

  (
    results_df,
    rank_matrix,
    winner_matrix,
    first_full_recovery,
    stable_full_recovery,
  ) = characterize_trace_count(ciphertexts, traces, reference_key)

  # --------------------------------------------------------
  # Choose verification N
  #
  # We verify at the first full-recovery point if one exists.
  # This directly tests whether that reported threshold also
  # works under the exact same ALL-200-sample CPA.
  # --------------------------------------------------------

  verification_N = first_full_recovery

  global_verification = False

  if verification_N is not None:
    global_verification = full_200_sample_verification(
      ciphertexts, traces, reference_key, verification_N
    )

  # --------------------------------------------------------
  # FINAL SUMMARY
  # --------------------------------------------------------

  print()
  print("=" * 70)
  print("FINAL SUMMARY")
  print("=" * 70)

  print()
  print("Reference fake key:")

  print(f"    {reference_hex}")

  if first_full_recovery is not None:
    print(f"First full-recovery N (all 200 samples):")

    print(f"    {first_full_recovery}")

  else:
    print("First full-recovery N:")

    print("    NONE")

  if stable_full_recovery is not None:
    print(f"Stable full-recovery N (among tested counts):")

    print(f"    {stable_full_recovery}")

  else:
    print("Stable full-recovery N:")

    print("    NONE")

  if verification_N is not None:
    print(f"Verification N:")

    print(f"    {verification_N}")

    print("Global verification:")

    print(f"    {'PASSED' if global_verification else 'FAILED'}")

  else:
    print("Verification N:")

    print("    NONE")

    print("Global verification:")

    print("    NOT RUN")

  print()
  print("Output files:")

  print(f"    {FULL_PLOT_DIR}/")

  print(f"    {TRACE_CHAR_CSV}")

  print(f"    {RANK_PLOT}")

  print(f"    {WORST_RANK_PLOT}")

  print()
  print("=" * 70)


if __name__ == "__main__":
  main()
