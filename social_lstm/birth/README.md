# Pedestrian birth model (`social_lstm/birth`)

Model-based baseline for the appearance of **new** pedestrians: a Poisson birth
process with a per-scene arrival rate and a grid-based spatial birth intensity,
as used by the birth component of PMBM trackers. Everything here is numpy and
matplotlib; nothing is learned with gradient descent.

## Files, in the order to read them

| File | What it does |
|---|---|
| `appearance.py` | Turns an `obsmat.txt` into *appearance events*: each pedestrian id's first frame, position and initial velocity. Also scene duration and the scalar arrival rate. |
| `analyze_appearances.py` | Step 1 CLI. Summary table + `plots/birth/<scene>_appearance_map.png` and `<scene>_arrivals_time.png`. |
| `exercise_counts.py` | Learning exercise: counts arrivals in fixed time windows and compares the tally with a Poisson distribution (variance/mean test). `plots/birth/exercise_counts_T<window>.png`. |
| `poisson_birth.py` | Step 2 models: `GridBirthModel` (histogram intensity) and `KDEBirthModel` (Gaussian kernel intensity), both with `fit`, `expected_count`, `sample`, `log_likelihood`; `GridBirthModel.uniform` is the baseline. |
| `evaluate_birth.py` | Step 2 CLI. Fits on the first 80 % of each recording, scores uniform / grid / KDE on the last 20 %. `plots/birth/<scene>_birth_model.png` and `birth_eval_summary.csv`. |
| `fov.py` | A stationary synthetic robot (`RobotView`: pose, field of view, range) and the replay that records each pedestrian's first frame inside its cone. |
| `analyze_fov.py` | Field-of-view CLI. Compares FOV births with scene births, fits the three models on FOV births. `plots/birth/<scene>_fov_births.png`, `<scene>_fov_birth_model.png`, `fov_summary.csv`. |

Run everything from the repository root:

```bash
python -m social_lstm.birth.analyze_appearances
python -m social_lstm.birth.exercise_counts --window 10
python -m social_lstm.birth.evaluate_birth                   # 0.5 m cells, 0.5 m bandwidth, 2 m regions
python -m social_lstm.birth.evaluate_birth --cell 1.0        # other grid cell sizes
python -m social_lstm.birth.evaluate_birth --bandwidth 0.25  # other KDE bandwidths
python -m social_lstm.birth.analyze_fov                      # robot at scene centre, 60 deg, 8 m
python -m social_lstm.birth.analyze_fov --placement edge     # robot at the scene edge looking in
python -m social_lstm.birth.analyze_fov --data data/ucy/zara01 --robot 0 12 -90
```

## The model in five lines

* Arrivals in the scene are a Poisson process with rate `lam` [1/s]: the number
  of arrivals in any `T` seconds is Poisson with mean `lam * T`, and arrivals are
  independent of each other.
* Where they appear is described by a rate per grid cell, `rate[i, j]` [1/s],
  with `sum(rate) == lam`. The intensity is `rate / cell_area` [1/s/m²].
* **Fit** = counting: `rate[i, j] = arrivals that started in cell (i, j) / duration`.
  A fraction `uniform_mix` (5 %) of the rate is spread evenly over all cells so
  that an arrival in a never-seen cell has non-zero probability.
* **Expected births in a region R over T seconds** = `T * sum(rate * overlap(cell, R))`.
* **Sampling** = draw `N ~ Poisson(lam * T)`, pick `N` cells with probability
  `rate / lam`, place each birth uniformly inside its cell and at a uniform time.
* **KDE variant**: instead of cells, every training arrival contributes a 2D
  Gaussian bump of width `bandwidth`; the intensity is the sum of bumps divided
  by the duration (plus the same 5 % uniform floor). The bandwidth plays the role
  of the cell size. Expected counts in a rectangle have a closed form (erf), and
  sampling is "pick a training arrival, add Gaussian noise".

Scoring on held-out events uses the Poisson point-process log-likelihood
`sum_i log intensity(x_i) - T * lam`. The uniform baseline has the same `lam`,
so the difference between the two models is purely about *where*.

## Step 2 results (defaults: 0.5 m cells, 0.5 m bandwidth, 2 m regions, 80/20 time split)

Held-out log-likelihood of the appearance events (higher is better) and the mean
log spatial density per event ("where", independent of "how many"):

| scene | train | test | predicted | LL uniform | LL grid | LL KDE | spatial/event uniform | grid | KDE |
|---|---|---|---|---|---|---|---|---|---|
| seq_eth | 237 | 122 | 59.2 | -908.8 | -953.7 | **-779.9** | -6.00 | -6.37 | **-4.95** |
| seq_hotel | 303 | 77 | 75.7 | -508.1 | -464.7 | **-440.9** | -4.97 | -4.41 | **-4.10** |
| zara01 | 117 | 23 | 29.2 | -177.4 | -154.6 | **-139.3** | -5.54 | -4.55 | **-3.88** |
| zara02 | 170 | 33 | 42.5 | -250.0 | -189.0 | **-171.4** | -5.61 | -3.76 | **-3.22** |
| students03 | 318 | 68 | 79.5 | -438.3 | -349.9 | **-310.2** | -5.89 | -4.59 | **-4.00** |

Three findings to study:

1. **The 0.5 m grid beats uniform in four of five scenes, but loses on seq_eth.**
   The ETH corridor is large (about 1,500 cells) and only 237 training arrivals
   fall into it, so most cells are empty and the held-out arrivals that land in
   them only get the 5 % uniform floor. That is overfitting of the histogram.
   Sweeping the cell size (held-out LL of the grid model):

   | cell | seq_eth | seq_hotel | zara01 | zara02 | students03 |
   |---|---|---|---|---|---|
   | 0.5 m | -953.7 | -464.7 | -154.6 | -189.0 | -349.9 |
   | 1.0 m | -864.7 | **-449.9** | **-139.0** | **-168.4** | **-331.5** |
   | 2.0 m | **-799.6** | -452.8 | -150.8 | -180.8 | -334.4 |
   | 3.0 m | -833.5 | -468.1 | -163.0 | -202.5 | -357.5 |

2. **The KDE beats the best grid in every scene, at every bandwidth tried.**
   Smoothing with Gaussian bumps instead of hard cell edges removes the "empty
   cell cliff". Sweeping the bandwidth (held-out LL of the KDE model):

   | bandwidth | seq_eth | seq_hotel | zara01 | zara02 | students03 |
   |---|---|---|---|---|---|
   | 0.10 m | -934.6 | -467.3 | -147.1 | -172.7 | -314.6 |
   | 0.15 m | -875.9 | -441.1 | -138.4 | **-160.3** | -301.3 |
   | 0.25 m | -824.1 | **-435.0** | **-134.3** | -160.5 | **-295.6** |
   | 0.50 m | -779.9 | -440.9 | -139.3 | -171.4 | -310.2 |
   | 0.75 m | -768.5 | -447.2 | -144.1 | -179.7 | -326.5 |
   | 1.00 m | **-768.3** | -453.6 | -148.1 | -186.4 | -339.9 |
   | 1.50 m | -779.1 | -465.0 | -155.1 | -196.8 | -360.2 |
   | 2.00 m | -794.4 | -474.1 | -160.8 | -205.3 | -375.1 |

   The best bandwidth is 0.15 to 0.25 m for the four dense scenes and about 1 m for
   the sparse ETH corridor. The 0.5 m default is within a few points of the best
   everywhere; a per-scene bandwidth chosen on a validation split would be the
   next refinement.

3. **The scalar rate is wrong when the rate is not stationary.** seq_eth had 122
   arrivals in its last 155 s against 59 predicted, because that stretch contains
   the busiest wave of the recording (see `exercise_counts_T10.png`). Neither
   spatial model can fix this; it is the homogeneous-in-time assumption failing,
   and the argument for a time-varying (Cox / Markov-modulated) rate later.

## Births relative to a robot's field of view

The data say when a pedestrian is first *annotated in the scene*. A robot sees a
new pedestrian when they first enter its camera cone, which can happen because
they entered the scene inside the cone (**scene_entry**) or because they were in
the scene already and walked into the cone (**walked_in**). `fov.py` places a
stationary robot (default: centre of the annotated area, looking along the main
walking direction, 60 deg field of view, 8 m range), replays the recording and
records each pedestrian's first in-cone frame. The result is a `SceneAppearances`
object, so steps 1 and 2 run on it unchanged.

| scene | robot pose | scene births | FOV births | scene_entry | walked_in | scene /min | FOV /min |
|---|---|---|---|---|---|---|---|
| seq_eth | (3.2, 5.0), 3 deg | 359 | 317 | 26 | 291 | 27.9 | 24.6 |
| seq_hotel | (0.5, -3.0), -88 deg | 380 | 224 | 122 | 102 | 31.6 | 18.6 |
| zara01 | (-0.5, 12.9), -86 deg | 140 | 129 | 60 | 69 | 23.3 | 21.5 |
| zara02 | (-1.0, -2.7), -88 deg | 203 | 190 | 89 | 101 | 29.0 | 27.1 |
| students03 | (0.7, 0.6), -67 deg | 386 | 206 | 86 | 120 | 107.4 | 57.3 |

Findings:

1. **Half or more of what the robot sees as a birth is not new to the scene.**
   With the robot in the middle of the ETH corridor 92 % of its births are people
   who walked into the cone; in the other scenes it is about half. The scene-level
   model of step 2 describes only the other part. For the camera-panning goal
   this is the part that matters, because it is the part the robot could have
   seen earlier by looking elsewhere.
2. **FOV births concentrate on the cone boundary**, where people cross the arc at
   8 m or the two side rays. The grid model matches or slightly beats the KDE
   here (held-out LL, robot at the centre):

   | scene | uniform | grid | KDE |
   |---|---|---|---|
   | seq_eth | -587.3 | **-467.4** | -487.5 |
   | seq_hotel | -327.5 | **-262.9** | -269.3 |
   | zara01 | -126.0 | **-102.7** | -103.8 |
   | zara02 | -220.6 | -192.8 | **-180.2** |
   | students03 | -156.0 | **-119.2** | -121.4 |

   The cone has hard edges and Gaussian bumps smear intensity across them into
   the part of the bounding box the robot cannot see.
3. **The models fitted on FOV births are descriptive only.** They answer "what
   does a camera at this pose experience as a birth", and that quantity is zero
   outside the cone by definition. The quantity a planner needs is the *scene*
   intensity from step 2, which must be non-zero where the robot is not
   looking: that is the whole reason to look elsewhere. So the FOV-fitted
   models should not be truncated to the cone or extended further. In PMBM
   terms, "scene_entry" births are the scene birth intensity times a visibility
   indicator, and "walked_in" births are undetected targets (the tracker's
   Poisson component) crossing into the cone. The FOV replay is therefore best
   kept as ground truth for testing that component once it exists.

Assumptions specific to this step: the robot does not move or turn; no
occlusion; only the first entry into the cone counts (re-entries are reported
separately: 25 in students03, 1 to 4 elsewhere); pedestrians inside the cone in
the first frame are left-censored as before. `--placement edge` and `--robot X Y
HEADING` give other poses; the numbers above move noticeably with the pose,
which is itself a result: the birth intensity the robot should use depends on
where it stands and looks.

## Assumptions and caveats

* `dt = 0.4 s` per annotated frame for every scene (stated in the ETH `info.txt`,
  verified from the velocity columns in all five files). Frame-id steps differ
  (6 for seq_eth, 10 elsewhere) and are handled.
* Pedestrians present in the very first frame are not arrivals (left-censored)
  and are excluded from the rate by default. This removes 42 of 428 in students03.
* "Appearance" means *first annotated in the scene*, not *entered a robot's
  field of view*. The ETH annotators also avoided people at the image border.
  A field-of-view version would re-run the same extraction on a synthetically
  cropped scene.
* The grid covers the bounding box of all annotated positions, padded by one
  cell. That is scene geometry and does not leak held-out events into the fit.
* Arrivals are not independent: about 20 % of consecutive arrivals share a frame
  (groups) and the rate drifts over tens of seconds. The variance/mean ratio of
  10 s window counts is 4.1 (eth), 4.0 (hotel), 1.8 (zara01), 1.2 (zara02),
  2.0 (students03); Poisson would give 1. The baseline gets the mean right and
  is overconfident about the spread.

## Ideas for after the baseline

* Rate changes in time: Markov-modulated Poisson process or a Cox process; a
  learned, context-conditioned intensity is a Cox process in disguise.
* Clumping: negative binomial / generalized Poisson / COM-Poisson at the count
  level, or a cluster process (Neyman-Scott, Hawkes) at the event level. Note that
  the PMBM filter wants a Poisson *intensity* for births, so a time-varying
  intensity fits the tracker more naturally than a non-Poisson count model.
* Spatial smoothing is done (KDE); a per-scene bandwidth chosen on a validation
  split, or an anisotropic kernel aligned with walkways, are possible refinements.
* Field of view: a moving or turning robot; occlusion by nearer pedestrians.
  Use the FOV replay (scene_entry + walked_in counts per interval) as the
  target when the undetected-target intensity of a PMBM-style tracker is built.
