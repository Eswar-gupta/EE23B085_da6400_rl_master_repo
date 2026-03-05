# EE23B085_da6400_rl_master_repo 

## For Grid_world Just fallow the notebook and run all cells in line

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Clone and navigate to directory
jupyter notebook acrobot_rl.ipynb
```

## Dependencies
- gymnasium
- numpy
- matplotlib
- tqdm

## Overview
Implementation and comparison of Q-Learning and SARSA algorithms on the Acrobot-v1 
environment from OpenAI Gymnasium using tabular methods with state space discretization.

## Environment
- **Environment:** Acrobot-v1 (Gymnasium)
- **State Space:** 6 continuous variables (joint angles and angular velocities)
- **Action Space:** 3 discrete actions (apply torque left, none, right)
- **Reward:** -1 per timestep until goal reached or 500 steps exceeded

## Algorithms Implemented
- **Q-Learning** (off-policy)
- **SARSA** (on-policy)