# Contributing Guide

Welcome! This document explains **how the three of us collaborate** so we
can work in parallel without stepping on each other's toes and produce a
clean, integrated final submission.

---

## 1. Branching Strategy

We use a simple **feature-branch workflow**:

```
main                ← stable, always works; final submission comes from here
│
├── member1/<topic>  ← e.g. member1/dqn-agent
├── member2/<topic>  ← e.g. member2/custom-env
└── member3/<topic>  ← e.g. member3/plotting-utils
```

### Rules

| Rule | Why |
|------|-----|
| **Never commit directly to `main`** | Keeps the integration branch stable |
| **One branch per feature / experiment** | Easy to review and revert |
| **Open a Pull Request to merge** | At least one team member reviews before merge |
| **Keep branches short-lived** | Merge often to avoid big conflicts |

---

## 2. Day-to-Day Workflow

```bash
# Start work on a new feature
git checkout main
git pull origin main
git checkout -b member1/dqn-agent

# ... make changes, commit often ...
git add src/agents/dqn.py
git commit -m "feat(agents): add DQN with replay buffer"

# Push and open a Pull Request
git push origin member1/dqn-agent
# → go to GitHub, open PR from member1/dqn-agent → main
```

---

## 3. Handling Files Edited by Multiple People

Some files (like `src/train.py` or `report/report_template.md`) will be
touched by everyone. Follow these steps to avoid merge conflicts:

1. **Pull `main` before starting** (`git pull origin main`).
2. **Make small, focused commits** so conflicts are easy to resolve.
3. **Communicate in the PR** — if you're about to change a shared file,
   drop a message in the group chat first.
4. **Resolve conflicts locally**:
   ```bash
   git fetch origin
   git merge origin/main          # or git rebase origin/main
   # fix conflicts, then:
   git add <conflicted-file>
   git commit
   git push
   ```

---

## 4. Directory Ownership

Each member "owns" a directory — meaning they are the primary author and
reviewer for that area:

| Directory | Primary Owner |
|-----------|--------------|
| `src/agents/` | Member 1 |
| `src/environments/` | Member 2 |
| `src/utils/` | Member 3 |
| `experiments/configs/` | All (each member adds their own YAML) |
| `notebooks/` | All (each member adds their own notebook) |
| `report/` | All (fill in together) |

If you need to change someone else's directory, mention it in the PR
description so they can review it.

---

## 5. Experiment Configs

Each member keeps their experiment isolated via YAML config files:

```bash
# Copy the default config and rename it
cp experiments/configs/default.yaml experiments/configs/member1_dqn.yaml
# edit the file, then run:
python src/train.py --config experiments/configs/member1_dqn.yaml
```

Results (`*.csv`, `*.pt`, plots) go in `experiments/results/` which is
**git-ignored** — save them locally or upload to a shared drive.

---

## 6. Commit Message Convention

```
<type>(<scope>): <short summary>

type  : feat | fix | docs | test | refactor | chore
scope : agents | environments | utils | train | report | notebooks
```

Examples:
```
feat(agents): add PPO with clipped objective
fix(utils):   correct moving_average edge case
docs(report): add experiment results section
```

---

## 7. Final Integration Checklist

Before submitting, the team lead should:

- [ ] All feature branches merged into `main` via reviewed PRs
- [ ] `pytest tests/ -v` passes with no failures
- [ ] `report/report_template.md` fully filled in
- [ ] All member names / roll numbers added to the report table
- [ ] Tag the final commit: `git tag v1.0-final && git push origin v1.0-final`
