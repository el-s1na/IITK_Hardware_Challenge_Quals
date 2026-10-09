TASK 1 -> Fake key
TASK 2 -> Minimum number of traces required to get fake key
TASK 3 -> Actual Round10 key
TASK 4 -> Actual Master key

## Task 3 and Task 4 progress

- `third_task.py` runs the Task 3 analysis end to end: it counts ciphertext
  byte histograms, derives round activity samples from trace variance, scans
  all 256 missing-endpoint candidates with a train/held-out transition-HW
  model, prints the supported round-10 key when both training directions
  agree, and saves 16 correlation plots for its actual-key transition model.
  It reads `Round10_traces.csv` directly.
- `fourth_task.py` takes that round-10 key, reverses the AES-128 key schedule,
  and checks that expanding the recovered master key returns the input key.

Run them with:

```sh
python third_task.py
python fourth_task.py <32-hex-character-round10-key>
```

The analysis history, results, and assumptions are in
[`TASK3_TASK4_METHOD.md`](TASK3_TASK4_METHOD.md).
