"""Option 1: start the crowd multiplier with a weak prior, so the undetected-pedestrian model adapts quickly to an
unknown crowd density (PMBConfig.rate_prior_strength, rate_tau).

Selection uses ONLY the calibration episodes (2nd quarter of each recording; results/ppp_calibration/logs.pkl), under
  A. no shift:          prior fitted on the 1st quarter (as deployed)
  B. controlled shift:  prior fitted on the 1st quarter with half the pedestrians removed (it expects ~half the
                        density, as the real 1st-half prior does for the denser 2nd halves)
Rule (fixed before running): pick the (strength, memory) pair with the lowest summed negative-binomial NLL over A and
B (untracked counts near the robot outside the view + collision disks 1-8 steps ahead). Then, once, the test episodes
(seed 61, a real density shift) are scored with the chosen setting against the uncalibrated tracker.

    python weak_prior_experiment.py
"""
import os

import numpy as np

import calibrate_ppp as c
from evaluate_nav import SCENES

STRENGTHS = [None, 10.0, 3.0, 1.0, 0.5]
TAUS = [None, 50.0, 200.0, 1e9]
CACHE = "results/ppp_calibration/thinned"
_THIN = {}


def thinned_prior(rec):
    """Prior fitted on the 1st quarter with half the pedestrians (by id, seeded) removed."""
    from robot_nav.pmb import fit_scene_prior
    sc = rec["scene"]
    if sc not in _THIN:
        os.makedirs(CACHE, exist_ok=True)
        d = np.loadtxt(SCENES[sc]["path"])
        ids = np.unique(d[:, 1])
        keep = np.random.default_rng(c.SCENE_NAMES.index(sc)).permutation(ids)[: len(ids) // 2]
        path = f"{CACHE}/{sc}_half.txt"
        if not os.path.exists(path):  # written once, atomically (parallel workers may race)
            tmp = f"{path}.{os.getpid()}.tmp"
            np.savetxt(tmp, d[np.isin(d[:, 1], keep)], fmt="%.4f")
            os.replace(tmp, path)
        first, step, q1, split = c.quarters(sc)
        _THIN[sc] = fit_scene_prior(path, frame_range=(first, q1))
    return _THIN[sc]


def evaluate(recs, prior_fn, strength, tau):
    now, fc = c.replay(recs, prior_fn=prior_fn, rate_prior_strength=strength, rate_tau=tau)
    lam = now[:, 2] / now[:, 3]
    p = 1 - (fc[:, 3] / (fc[:, 3] + fc[:, 1])) ** fc[:, 2]
    y = (fc[:, 4] > 0).astype(float)
    nll_now, nll_fc = c.score(now, fc)
    return dict(nll=nll_now + nll_fc, ratio_now=np.mean(lam * now[:, 0]) / now[:, 1].mean(), ratio_fc=p.mean() / y.mean(),
                brier=np.mean((p - y) ** 2))


def name(strength, tau):
    return f"strength {'fitted' if strength is None else f'{strength:g}':>6s}, memory {'fitted' if tau is None else ('none' if tau > 1e8 else f'{tau:g}'):>6s}"


def main():
    import pickle
    logs = pickle.load(open(f"{c.OUT}/logs.pkl", "rb"))
    train, test = logs["train"], logs["test"]
    rows = []
    out = ["Weak crowd-multiplier prior: selection on calibration episodes (A: no shift, B: prior expects half the density)",
           f"{'setting':34s} {'A: NLL':>8s} {'ratio now':>9s} {'ratio disk':>10s} {'B: NLL':>8s} {'ratio now':>9s} {'ratio disk':>10s} {'A+B NLL':>8s}"]
    for s_ in STRENGTHS:
        for t_ in TAUS:
            A = evaluate(train, None, s_, t_)
            B = evaluate(train, thinned_prior, s_, t_)
            rows.append((s_, t_, A, B))
            out.append(f"{name(s_, t_):34s} {A['nll']:8.4f} {A['ratio_now']:9.2f} {A['ratio_fc']:10.2f} {B['nll']:8.4f} "
                       f"{B['ratio_now']:9.2f} {B['ratio_fc']:10.2f} {A['nll'] + B['nll']:8.4f}")
            print(out[-1], flush=True)
    best = min(rows, key=lambda r: r[2]["nll"] + r[3]["nll"])
    out.append(f"-> chosen: {name(best[0], best[1])}")
    out.append("\nTest episodes (seed 61; real density shift; nothing selected on them):")
    out.append(c.report("uncalibrated (as in the paper)", *c.replay(test)))
    out.append(c.report(f"weak prior: {name(best[0], best[1])}",
                        *c.replay(test, rate_prior_strength=best[0], rate_tau=best[1])))
    txt = "\n".join(out)
    print(txt)
    with open(f"{c.OUT}/weak_prior_summary.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
