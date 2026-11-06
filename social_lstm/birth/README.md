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
| `poisson_birth.py` | Step 2 model: `GridBirthModel` with `fit`, `uniform`, `expected_count`, `sample`, `log_likelihood`. |
| `evaluate_birth.py` | Step 2 CLI. Fits on the first 80 % of each recording, scores on the last 20 %, against a uniform baseline. `plots/birth/<scene>_birth_model.png` and `birth_eval_summary.csv`. |

Run everything from the repository root:

```bash
python -m social_lstm.birth.analyze_appearances
python -m social_lstm.birth.exercise_counts --window 10
python -m social_lstm.birth.evaluate_birth              # 0.5 m cells, 2 m regions
python -m social_lstm.birth.evaluate_birth --cell 1.0   # try other cell sizes
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

Scoring on held-out events uses the Poisson point-process log-likelihood
`sum_i log intensity(x_i) - T * lam`. The uniform baseline has the same `lam`,
so the difference between the two models is purely about *where*.

## Step 2 results (default settings: 0.5 m cells, 2 m regions, 80/20 time split)

| scene | train | test | predicted | LL grid | LL uniform | spatial/event grid | spatial/event uniform |
|---|---|---|---|---|---|---|---|
| seq_eth | 237 | 122 | 59.2 | -953.7 | **-908.8** | -6.37 | **-6.00** |
| seq_hotel | 303 | 77 | 75.7 | **-464.7** | -508.1 | **-4.41** | -4.97 |
| zara01 | 117 | 23 | 29.2 | **-154.6** | -177.4 | **-4.55** | -5.54 |
| zara02 | 170 | 33 | 42.5 | **-189.0** | -250.0 | **-3.76** | -5.61 |
| students03 | 318 | 68 | 79.5 | **-349.9** | -438.3 | **-4.59** | -5.89 |

"spatial/event" is the mean log of the normalised spatial density at the
held-out arrivals, i.e. how well the model says *where*, independent of *how many*.
Higher is better everywhere.

Two findings to study:

1. **The grid beats uniform in four of five scenes, but loses on seq_eth at 0.5 m.**
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

   1 m cells are best for four scenes and 2 m for ETH. The same effect shows up
   when raising `--uniform-mix`. The principled fix is to smooth the histogram
   (a KDE, or a coarse-to-fine grid), which is the "KDE alternative" to discuss.

2. **The scalar rate is wrong when the rate is not stationary.** seq_eth had 122
   arrivals in its last 155 s against 59 predicted, because that stretch contains
   the busiest wave of the recording (see `exercise_counts_T10.png`). The grid
   model cannot fix this; it is the homogeneous-in-time assumption failing, and
   the argument for a time-varying (Cox / Markov-modulated) rate later.

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
* Spatial smoothing: kernel density estimate instead of a histogram, with the
  bandwidth chosen on the held-out log-likelihood as above.
