### Overview

### Task1

The challenge gives us leakage hypothesis around final-state register, means when the transition is from S9 to Ciphertext. We can calculate the Hamming Distance between these transition, and in this transition fake key is used. We can use CPA and then choose the candidate with highest absolute correlation peak to be the winner and thus obtaining the fake key byte.

In solve.py code, we use the value of k from 0x00 to 0xff, and calculate H(k), suppose Hj(k) is predicted leakage for the byte k of trace j. And the power at trace j at time t is Pj(t). So for candidate k and t, we have two vectors H(k) and P(t).

We calculate pearson correlation between these two which gives us a number between -1 to 1. As the readme stated we take the absolute correlation value.
and then we calculate Score = max(over time)absolute(rho(k,t))
for one byte k and time sample from 0 to 200.

TODO: mathematical proof and clean the comments from code

### Task 2

In this we have to find out, the minimum number of traces required to calculate the fake key. In second_task.py, we first calculate the refernce key which is the fake key, and for different number of traces N, we calculate the score(according to absolute correlation) for a byte position and then for the byte position we find what is expected position of the reference key in this trace, and thus finding rank. If the rank is 1 for all the byte position then that means we found the minimum Number of traces.

### Task 3 — actual round-10 key

The Task 1 CPA returns the fake key, so it does not solve actual-key recovery. We first counted ciphertext values at each byte position and found one absent value and one doubled value. The relation between the doubled value and fake key is consistent with the countermeasure’s identity-initialized lookup table, but the histogram allows multiple missing-endpoint explanations.

To distinguish those explanations, we formed all 256 round-10-key candidates from the absent ciphertext values and each possible missing S-box input. We reconstructed AES states for ordinary ciphertext rows and compared predicted round-transition Hamming distances with trace samples. The candidate scan fits on one half of the capture and scores on the other, then swaps halves. `z=0xff` ranked first in both directions, producing `d014f9a8c9ee2589e13f0cc8b6630ca6` as the actual round-10-key candidate.

This conclusion depends on the shared-endpoint, standard-AES-row, and transition-leakage assumptions. Timing is derived from across-trace sample variance: `third_task.py` finds the ten strongest local activity peaks, and both training halves independently find samples 13, 17, 21, 25, 29, 33, 37, 41, 45, and 49. The four-sample spacing is observed after peak detection, not supplied to it. The histogram alone also supports `z=0`; the transition scan is the evidence that favors `z=0xff`. The script saves 16 per-byte correlation plots for the selected actual-key transition model in `actual_round10_correlation_plots/`. Details and evidence limits are in [TASK3_TASK4_METHOD.md](TASK3_TASK4_METHOD.md).

### Task 4 — actual master key

AES-128 key expansion can be inverted from the round-10 key. Inverting the Task 3 candidate gives `2b7e151628aed2a6abf7158809cf4f3c`, and expanding it reproduces the selected round-10 key. Task 4 is therefore an exact key-schedule step conditional on Task 3’s candidate; it does not independently resolve the missing-endpoint ambiguity.
