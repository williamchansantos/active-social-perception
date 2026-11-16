# SocNavBench curated episodes, ground-truth observation vs RGB-D vision (written before any results)

Episodes: SocNavBench's 30 single-recording validation episodes (params/episode_params_val.ini; ETH 6, Zara1 6,
Zara2 5, Univ 8, Hotel 5; the 3 DoubleHotel ones left out; "HotelSmall" -> "Hotel" map). One run per episode
and method (seed 0). Planner as in the 2D evaluation: SupeR-MPC, PMB (prune_pos_std 1.0), rotation-augmented
leave-one-scene-out Social LSTM, 60 deg / 4 m camera, delta_unseen 0.02; radii matched to SocNavBench (robot 0.24,
pedestrian 0.2). PMB prior: the episode's recording with the episode's time window +-10 s left out.
Static obstacles: waypoint 2.5 m ahead on the geodesic path (no walls in the MPC).

Perception modes:
- gt: simulated camera (FOV, range, occlusion, p_detect 0.9, noise) on the true positions, known identities.
- vision: SocNavBench renders RGB-D at the robot pose + pan; COCO Faster R-CNN (MobileNetV3) persons, range from
  depth, calibrated noise, Hungarian association; robot pose still ground truth + noise.
Methods: proposed (risk-aware gaze + observation-aware constraint), information gain, look-at-nearest, fixed;
gt also 360 deg (upper bound).

Outcome per episode: success = SocNavBench "Success" (goal within 0.3 m in the time budget) AND 0 robot-pedestrian
collisions (episode log). Also: reached, collisions, time to goal, min clearance.
Comparisons (exact McNemar, paired over the 30 episodes), per perception mode: proposed vs information gain
(two-sided), proposed vs nearest, proposed vs fixed. And per method: gt vs vision (the cost of real perception).
30 episodes give little power (a pilot); differences below ~25 points will not be significant.
