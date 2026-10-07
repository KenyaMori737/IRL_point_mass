# IRL point-mass experiments

This repository contains exploratory point-mass inverse reinforcement learning
notebooks related to the planar reaching experiments in
[arXiv:2505.08619](https://arxiv.org/abs/2505.08619).

## Setup

Python 3.12 is recommended. From the repository root:

```bash
git clone https://github.com/KenyaMori737/IRL_point_mass.git
python3 -m venv .venv_IRL_point_mass
cd IRL_point_mass
source ../.venv_IRL_point_mass/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start JupyterLab from the repository root so that the local modules can be
imported:

```bash
jupyter lab
```

For the five-obstacle, 2.5-second point-mass example, open
`PointMass_IRL_AutoReg_7.ipynb` and run the cells from the top. Cells 0--7
construct the environment and solve the expert and initial trajectories. Cells
8--9 define and execute the IRL loop. The later cells visualize and evaluate
the learned weights.

`PointMass_IRL_AutoReg_6.ipynb` is an older four-obstacle experiment and has
saved interrupted outputs; use `PointMass_IRL_AutoReg_7.ipynb` as the primary
smoke test.

## Minimal one-obstacle MO-IRL example

Run the standalone PM1-style example with:

```bash
python test_main1.py
```

It uses a 1.5-second trajectory with `dt=0.05`, one circular obstacle, 20
trajectory suffixes, elastic-net regularization, step acceptance, and a moving
window of one trajectory. Results are written to:

- `outputs/mo_irl_one_obstacle.png`
- `outputs/mo_irl_one_obstacle.npz`

For a short smoke test:

```bash
python test_main1.py --max-irl-iterations 2 --max-oc-iterations 30 --subsamples 5
```

Use `python test_main1.py --help` to see all options. Add `--show` to open the
result plot after training.

## Four-pole MO-IRL example

Run the PM2-style four-pole setup taken from
`PointMass_IRL_AutoReg_6.ipynb` with:

```bash
python test_main2.py
```

Results are written to `outputs/mo_irl_four_poles.png` and
`outputs/mo_irl_four_poles.npz`. To display the obstacle-cost activation
regions in addition to the physical poles, run:

```bash
python test_main2.py --show-activation-margin
```

## Notes

- The notebooks are research prototypes rather than the final MO-IRL
  implementation used for every result in the paper.
- The callback compatibility code in `PointMass_model.py` supports both older
  mim-solvers versions, where Crocoddyl callbacks were accepted, and current
  versions, which provide callbacks in the `mim_solvers` module.
