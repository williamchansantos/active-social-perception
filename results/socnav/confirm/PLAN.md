# SocNavBench confirmation run (written before any results)

Hypothesis (from the exploratory runs val30 / val30w, seed 0): with real RGB-D perception the proposed method
(risk-aware gaze + observation-aware unseen constraint) succeeds more often than a fixed camera and than
information-gain gaze, and loses less success when ideal observation is replaced by vision.

Setup exactly as results/socnav/val30w (walls in the MPC, --walls 8; 60 deg / 4 m; PMB prune 1.0; loorot
predictors; leakage-free prior; radii 0.24 / 0.2), nothing re-tuned. 30 curated episodes x NEW seeds 1, 2, 3
(server --seed: robot pose-estimate noise, simulated-camera noise) = 90 paired episode-seeds per method.
Methods: proposed, information gain, fixed. Perception: gt (simulated camera on true positions) and vision.

Primary outcome: success excluding spawn = SocNavBench "Success" and no collision (robot-pedestrian centre
distance < 0.44 m in the simulator states) with a pedestrian who had existed for >= 2 s.
Primary tests (one-sided, exact; Holm over the four):
  H1 vision: proposed > fixed (McNemar)
  H2 vision: proposed > information gain (McNemar)
  H3 perception cost (gt minus vision, per episode-seed, in {-1, 0, 1}): proposed < fixed (sign test)
  H4 perception cost: proposed < information gain (sign test)
Secondary: the same with all collisions; gt comparisons; collisions per method; per dataset.
The 90 episode-seeds are not independent across seeds of the same episode; a per-episode check (majority over
seeds) is reported alongside.
