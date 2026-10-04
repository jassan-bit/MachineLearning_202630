Frozen SVR delivery audit, computed on 2026-10-04.
Reproduce: .venv-repro/Scripts/python.exe src/audit_current_model.py
Source predictions: results/optimized_minute_2023_2025/predictions.csv.gz
No saved models, source data, original metrics or forecasts are changed.
metric_confidence_intervals.csv: per-configuration and macro percentile 95% intervals.
shared_bootstrap_indices.npz: one date draw per replication shared by both models and all series.
paired_loss_bootstrap.csv: SVR minus persistence MSE, recentered two-sided bootstrap, Holm over 16 configurations.
residual_diagnostics.csv: 112 separate series, JB nominal and dependence-adjusted HAC diagnostics.
residual_acf_pacf.csv / residual_ljung_box.csv: no concatenation of series; Holm families stated in provenance.
learning_curve.csv: 384 auxiliary 2023--2024 fits, fixed delivery configuration, same validation per cut.
coefficients_engineered_units.csv: inverse-scaled coefficients and changes in relative correction per training SD.
Use provenance.json for assumptions, limitations, parameters and SHA-256 hashes.
2025 was already explored. These are conditional retrospective diagnostics, not a pristine test or causal analysis.
