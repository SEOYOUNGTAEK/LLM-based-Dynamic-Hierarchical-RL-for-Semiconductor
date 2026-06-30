===================================================================
SUPPLEMENTARY MATERIAL
LLM-Governed Hierarchical DRL for Dependable Scheduling
in Nondeterministic Digital Twins
HICSS-58 Submission
===================================================================

FILES
-----
1. supplementary_results_summary.csv
   Per-seed evaluation summary for all three conditions.
   One row per (condition, seed) combination (15 rows total).

   Columns:
     condition            : PureDRL | RuleBased | LLM-HRL
     seed                 : s42 | s123 | s777 | s2024 | s7
     eval_CT_mean_s       : Mean urgent-lot cycle time over eval episodes (s)
     eval_CT_std_s        : Std dev of CT within eval window (s)
     eval_CT_min_s        : Min CT observed during eval (s)
     eval_CT_max_s        : Max (worst-case) CT during eval (s)
     eval_TMR_mean_pct    : Mean Target Meet Rate during eval (%)
     fail_CT_gt_3500s     : 1 = FAIL (mean CT > tau), 0 = PASS
     failure_threshold_tau: 3500 (s) — fixed a priori threshold
     n_eval_episodes      : 30 (episodes 71-100 per run)

2. supplementary_results_all_episodes.csv
   Full episode-level data for all 15 runs (100 episodes each = 1,500 rows).

   Columns:
     condition         : PureDRL | RuleBased | LLM-HRL
     seed              : s42 | s123 | s777 | s2024 | s7
     episode           : 1-100
     phase             : train (ep 1-70) | eval (ep 71-100)
     urgent_lot_CT_s   : Urgent-lot cycle time for that episode (s)
     target_meet_rate_pct : Production attainment (%)
     total_reward      : Cumulative episode reward
     masking_rate      : Fraction of dispatch steps masked by L1 (LLM-HRL only)

EXPERIMENTAL SETUP
------------------
  Scenario      : 20 urgent lots / 258 total (7.7%), 100 episodes per run
  Failure def.  : mean eval CT > tau = 3,500 s  (episodes 71-100)
  Digital twin  : Siemens Tecnomatix Plant Simulation (semiconductor line)
  RL agent      : Rainbow DQN, hidden [1024, 512, 256], lr=3e-4, gamma=0.98
  LLM governor  : Mistral-7B via Ollama (local, no external API)
  Seeds         : 42, 123, 777, 2024, 7  (N=5 independent runs per condition)

AGGREGATE RESULTS (matches Table 2 in paper)
--------------------------------------------
  Condition   Fail   CT mean   CT std   Worst CT   TMR mean
  PureDRL     3/5    3819 s    914 s    5545 s     86.9%
  RuleBased   3/5    3347 s    543 s    8567 s     88.3%
  LLM-HRL     0/5    3071 s    249 s    3587 s     86.1%

  Statistical test: p = 0.010 (one-sided binomial, H0: rho_LLM = 0.60)
===================================================================
