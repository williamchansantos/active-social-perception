# Narrow field-of-view confirmation (written before any seed-31 results existed)

Motivation: on held-out seed 23, at 60 deg the proposed method beat information gain (111:80, p=0.03) and
look-at-nearest (127:78, p=0.001); at 87-120 deg it tied them. 60 deg was found after looking at seed 23,
so it is confirmed here on fresh episodes.

- Episodes: seed 31 (never used before), Zara01 + Zara02, 300 each (600 paired), clean protocol,
  range 4 m, delta_unseen 0.02, prune_pos_std 1.0, N = 8, R as default.
- Planners: proposed (supermpc-active, unseen_obs_aware=True); supermpc-infogain, supermpc-nearest,
  supermpc (fixed), supermpc-gazeoracle (open-loop unseen constraint, as in all earlier baselines);
  supermpc-360 once (does not depend on the FOV setting).
- Fields of view: 60 (primary), 45 and 70 (secondary).
- Primary hypotheses (success, exact McNemar, Holm over the two): at 60 deg, proposed > information gain
  and proposed > look-at-nearest.
- Secondary: the same comparisons at 45 and 70 deg; episodes with collisions against never-seen
  pedestrians (proposed vs information gain); proposed vs fixed at each FOV.
- No parameter is changed after seeing these results.
