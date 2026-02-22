# EE23B085 — DA6400 Reinforcement Learning Project

A collaborative RL project repository for 3 team members.

---

## Repository Layout

```
├── src/
│   ├── agents/          # RL algorithm implementations (one file per algorithm)
│   ├── environments/    # Custom / wrapped Gym environments
│   ├── utils/           # Shared helper functions used by everyone
│   └── train.py         # Main training entry-point
├── experiments/
│   ├── configs/         # YAML config files (one per experiment)
│   └── results/         # Saved plots / metrics (git-ignored, keep locally)
├── notebooks/           # Jupyter notebooks for analysis
├── report/
│   └── report_template.md  # Final report template — fill this in together
├── tests/               # Unit tests
├── requirements.txt
└── CONTRIBUTING.md      # Branching & collaboration guide ← READ THIS FIRST
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Eswar-gupta/EE23B085_da6400_rl_master_repo.git
cd EE23B085_da6400_rl_master_repo

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the default training script
python src/train.py --config experiments/configs/default.yaml

# 4. Run tests
pytest tests/
```

---

## Team Workflow (summary)

> See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full guide.

| Step | What to do |
|------|-----------|
| Start new work | `git checkout -b <member>/<feature>` |
| Share progress | Push your branch and open a Pull Request to `main` |
| Integrate | Merge PRs into `main` only after a team review |
| Final submission | Squash-merge all feature branches → `main`, fill in `report/report_template.md` |

---

## Module Ownership (suggested split)

| Module | Primary Owner | Notes |
|--------|--------------|-------|
| `src/agents/` | Member 1 | Implements RL algorithms |
| `src/environments/` | Member 2 | Wraps / customises Gym envs |
| `src/utils/` | Member 3 | Shared helpers, plotting, logging |
| `src/train.py` | All | Coordinate changes via PRs |
| `report/` | All | Fill in together before submission |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Submitting the Final Project

1. Ensure all feature branches are merged into `main`.
2. Complete `report/report_template.md` (or convert it to the required format).
3. Tag the final commit: `git tag v1.0-final && git push origin v1.0-final`.
4. Share the repo link (and tag) with the instructor.