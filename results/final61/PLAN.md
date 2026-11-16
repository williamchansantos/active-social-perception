# Final pre-registered simulation study (written and committed before any results)

Frozen method: SupeR-MPC + PMB tracker + risk-aware active gaze (tracked and unseen terms) + open-loop
unseen-pedestrian chance constraint (no observation-aware coupling), existence weighting, stale-track pruning at
1 m position std, CV-propagated track state covariance, robot-uncertainty propagation. 60 deg FOV, 4 m range,
N = 8, dt = 0.4 s, delta = 0.05, delta_unseen = 0.02. Rotation-augmented leave-one-scene-out Social LSTM
predictors (checkpoints/loorot_*.pt, SHA-256 in results/final61/checksums.txt). Code as committed with this plan.
Baseline implementations were frozen after a development check on seed 11 (40 episodes each on Zara01 and ETH);
nothing was tuned on it.

Primary metric (both parts): success excluding spawn collisions. Secondary: success counting all collisions.
- 2D: success = goal reached (0.3 m) within 90 steps and no collision except with "spawn" pedestrians, i.e.
  those whose replay starts mid-episode within 3 m of the robot (run_nav.py, success_nospawn).
- SocNavBench: success = SocNavBench "Success" and no collision (robot-pedestrian centre distance < 0.44 m in
  the simulator states) with a pedestrian who had existed for < 2 s.

Failures: one automatic retry per episode (SocNavBench) as implemented in run_batch.sh; all remaining
failures are counted and reported; comparisons use the episodes completed by all methods; no manual reruns.

## Part A. 2D replays (fresh seed 61)
Clean protocol (PMB prior fitted on the first half of each recording, episodes from the second half), 450
episodes in each of zara01, zara02, eth, hotel (1800 paired episodes).

Methods (all with the same planner, tracker and predictor; they differ only in gaze unless noted):
| name | planner |
|---|---|
| proposed | supermpc-active (unseen_obs_aware False) |
| information gain | supermpc-infogain |
| TTC heuristic | supermpc-ttc: look at the most collision-relevant tracked pedestrian (relevance to the planned path summed over the horizon; this behaviour was called "look-at-nearest" in all earlier results) |
| look-at-nearest | supermpc-nearest: look at the closest tracked pedestrian (new) |
| decoupled | supermpc-decoupled: plan the motion first without the camera, then optimize only the camera with the proposed perception cost for that motion |
| scan | supermpc-scan: periodic pan sweep |
| fixed | supermpc: forward-facing camera |
| gaze oracle | supermpc-gazeoracle: ground-truth pointing, same camera (upper bound) |
| 360 camera | supermpc-360 (upper bound) |

Ablations of the proposed method (each removes one component): no unseen constraint (unseen_constraint=False),
no unseen gaze term (active-tracked-only), no tracked gaze term (active-unseen-only), plain Kalman tracker
(active-kf), no stale-track pruning (prune_pos_std 2.5), no existence weighting (existence_weighting=False), no
robot-uncertainty propagation (active-no-sens), no track state covariance (active-no-statecov). (No active
gaze = the fixed baseline.)

Primary tests (exact McNemar on the 1800 paired episodes, success excl. spawn), Holm over the six:
  proposed > TTC heuristic, > look-at-nearest, > decoupled, > scan, > fixed (one-sided);
  proposed vs information gain (two-sided).
Ablation family (two-sided exact McNemar, full method vs each ablation), Holm over the eight.
Secondary: per scene group (Zara, ETH+Hotel); all-collision metric; never-seen / tracked collisions; oracles.

## Part B. SocNavBench, RGB-D vision (fresh seeds 7-16)
Setup as results/socnav/confirm and final53 (walls in the MPC, 30 curated single-recording episodes,
rendered RGB-D, Faster R-CNN detector, depth ranging, Hungarian association; robot pose ground truth + noise).
Methods: proposed and the THREE best non-oracle alternatives of Part A, ranked by pooled 2D success excl. spawn
among {information gain, TTC heuristic, look-at-nearest, decoupled, scan, fixed}, selected automatically by
results/final61/analyze.py (part A) before Part B is launched. 10 seeds x 30 episodes = 300 paired runs per method.

Primary tests: proposed > each of the three (one-sided), Holm over the three. Unit of analysis: the episode
(seeds of an episode are correlated): paired permutation test on the 30 per-episode mean differences
(sign flips, 100,000 permutations). Secondary: pooled exact McNemar over the 300 runs, 95% cluster-bootstrap
CI of the difference (resampling episodes), all-collision metric, per dataset.
