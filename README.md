# IITK Hardware Challenge

This repository contains our solution to the IITK Innovation Challenge 2026 qualification problem on side-channel analysis of AES-128. The objective is to study a final-round faking countermeasure, recover the fake round-10 key, estimate the measurements needed for full recovery, infer the actual round-10 key, and derive the corresponding master key.

The completed work includes four task solutions, per-byte correlation plots for both key recovery stages, rank and variance plots, and a report with the full plot appendices.

## File structure

```text
.
├── report/
│   ├── report.tex                    # Report source
│   ├── appendix_plots.tex            # Additional correlation plots
│   └── report.pdf                    # Compiled report
├── task_solution/
│   ├── first_task.py                 # Fake round-10 key recovery
│   ├── second_task.py                # Trace count analysis and rank plots
│   ├── third_task.py                 # Actual round-10 key recovery
│   ├── fourth_task.py                # Master key recovery
│   └── variance_plot.py              # Variance plot generation
├── correlation_plots_full/           # Fake-key correlation plots, bytes 00–15
├── actual_round10_correlation_plots/  # Actual-key correlation plots, bytes 00–15
├── rank_plots/
│   └── rank_plot_all_bytes.png
├── worst_rank_plots/
│   └── worst_rank_plot.png
├── variance_plots/
│   └── variance_plot.png
├── trace_characterisation.csv        # Trace count results
├── readme.docx                       # Challenge instructions
├── writeup.md                        # Task writeup
└── README.md
```
