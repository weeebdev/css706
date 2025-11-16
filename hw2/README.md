# Homework 2: Markov Decision Processes (MDP)

This repository contains implementations of Markov Decision Processes and various algorithms for solving them.

## Contents

### Practical Task Implementation (`mdp_practical_task.ipynb`)

A complete Jupyter notebook implementation of the practical task requirements:
- **GridWorldMDP**: A 3×3 grid world environment with:
  - Goal state (reward: +10)
  - Danger zone (reward: -5)
  - Stochastic transitions (80% intended, 20% random)
  - Discount factor: 0.9
- **Value Iteration**: Optimal value function and policy computation
- **Policy Simulation**: Random and policy-driven episode simulation
- **Visualization**: Grid world visualization and trajectory plotting
- **Performance Comparison**: Comparison between random and optimal policies

### Advanced Implementation (`mdp_homework.ipynb`)

The notebook includes:

1. **Grid World Environment**: A customizable grid world class for testing MDP algorithms
2. **Value Iteration**: Model-based algorithm that iteratively updates value function
3. **Policy Iteration**: Model-based algorithm that alternates between policy evaluation and improvement
4. **Q-Learning**: Model-free reinforcement learning algorithm
5. **Examples and Visualizations**: Grid world problems with visualizations of policies and value functions
6. **Analysis**: Comparison of algorithms and convergence analysis

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Open the practical task notebook:
```bash
jupyter notebook mdp_practical_task.ipynb
```

Or use JupyterLab:
```bash
jupyter lab mdp_practical_task.ipynb
```

3. Or open the advanced notebook:
```bash
jupyter notebook mdp_homework.ipynb
```

Or use JupyterLab:
```bash
jupyter lab mdp_homework.ipynb
```

## Running the Practical Task

The `mdp_practical_task.ipynb` notebook demonstrates:
- Grid world environment setup
- Value iteration algorithm implementation
- Random episode simulation
- Policy-driven episode simulation
- Performance comparison between random and optimal policies
- Visualizations of the grid world, values, policies, and trajectories

Simply open the notebook and run all cells:
```bash
jupyter notebook mdp_practical_task.ipynb
```

## Running the Notebook

Execute cells sequentially from top to bottom. The notebook includes:
- Complete implementations of all algorithms
- Example grid world problems
- Visualizations of policies and value functions
- Comparison between different algorithms
- Analysis of convergence and hyperparameter effects

## Key Features

- **GridWorld Class**: Flexible environment for creating custom MDP problems
- **Value Iteration**: Finds optimal value function V*(s) and policy π*(s)
- **Policy Iteration**: Alternative optimal algorithm with policy evaluation/improvement
- **Q-Learning**: Model-free learning that works without transition probabilities
- **Visualizations**: Heatmaps showing values and policies with arrows
- **Convergence Analysis**: Plots showing algorithm convergence

## Algorithms Implemented

### Value Iteration
- Iteratively updates value function using Bellman equation
- Extracts optimal policy from converged values
- Guaranteed convergence to optimal solution

### Policy Iteration
- Alternates between policy evaluation and policy improvement
- Often converges in fewer iterations than value iteration
- Also guaranteed to find optimal policy

### Q-Learning
- Model-free algorithm that learns Q-values from experience
- Uses epsilon-greedy exploration
- Learns optimal policy without knowing transition probabilities

## Example Usage

```python
# Create grid world
env = GridWorld(
    grid_size=(4, 4),
    obstacles=[(1, 1)],
    terminal_states={(0, 3): 1.0, (3, 3): -1.0},
    discount_factor=0.9
)

# Run value iteration
values, policy, history = value_iteration(env)

# Visualize results
env.visualize_grid(values=values, policy=policy)
```

## Notes

- All algorithms are implemented from scratch
- The grid world uses deterministic transitions
- Default reward is -0.04 per step (encourages finding terminal states quickly)
- Discount factor γ = 0.9 by default

