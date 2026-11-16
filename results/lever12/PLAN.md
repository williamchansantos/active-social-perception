# Development test of two planner changes (tuning seed 11; written before running)

Final method (no coupling) at 60 deg / 4 m, loorot predictors, clean protocol, prune 1.0. Zara01, Zara02, ETH,
Hotel, 300 episodes each (1200 paired per variant). Planners: proposed (supermpc-active) and fixed (supermpc).

| variant | change |
|---|---|
| base | as in the paper (delta_unseen 0.02, CV state covariance) |
| du01 | delta_unseen 0.01: restores the intended 2 %/step risk, since the unseen model's P(>=1) is ~2x low (results/ppp_accuracy) |
| gcov | state_cov_mode "predictor": track covariance propagated through the predictor's sensitivity instead of CV |
| both | du01 + gcov |

Metric: success excluding spawn (primary), all collisions, never-seen / tracked collisions, steps.
Rule: a change is carried to a pre-registered fresh-seed confirmation only if the proposed method's success improves
over base with exact McNemar p < 0.05 on these 1200 episodes and does not lower success on either scene group.
Seed 11 is the tuning seed; nothing here is reported as a result.
