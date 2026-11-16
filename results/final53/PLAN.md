# Final-method confirmation on fresh seeds (written before any results)

Method change (user decision, 2026-10-12): the observation-aware coupling of the unseen-pedestrian constraint is
dropped (unseen_obs_aware=False), because removing it had no measurable effect in 2D (p = 0.32) or SocNavBench
vision (14:13). Final method = SupeR-MPC + PMB tracker + risk-aware gaze (tracked and unseen terms) + open-loop
unseen chance constraint; baselines now differ from it only in gaze. Nothing else changed; nothing tuned.
Fresh seeds, never used before: 2D seed 53; SocNavBench seeds 4, 5, 6.

## A. 2D replays (seed 53)
60 deg / 4 m, rotation-augmented leave-one-scene-out predictors (loorot), clean protocol (PMB prior from the first
half, episodes from the second half), prune_pos_std 1.0, delta_unseen 0.02, N = 8, unseen_joint False.
Zara01+Zara02 (600 episodes) and ETH+Hotel (600). Planners: proposed (supermpc-active), information gain,
look-at-nearest, fixed, gaze oracle, 360.
Primary per scene group (success, exact McNemar, Holm over the 2 per group): proposed > look-at-nearest
(one-sided), proposed vs information gain (two-sided). Secondary: proposed > fixed; never-seen / tracked
collisions.

## B. SocNavBench (seeds 4-6)
As results/socnav/confirm/PLAN.md (walls, 60 deg / 4 m, 30 curated episodes, spawn-excluded success as primary),
with the final method (--camera active --mpc-set unseen_obs_aware=False). Methods: proposed, information gain,
fixed; perception: gt and vision. 90 paired episode-seeds per method.
Primary (one-sided exact McNemar, Holm over the two): H1 vision: proposed > fixed; H2 vision: proposed >
information gain. Secondary: the same with the simulated camera and with all collisions; per-episode check.
(H3/H4 of the earlier confirmation, smaller perception cost, are dropped: no method lost success from vision.)
