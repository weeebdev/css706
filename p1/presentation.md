---
theme: seriph
background: https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1920
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Project 2: Reinforcement Learning
  Training RL agents for BipedalWalker-v3
drawings:
  persist: false
transition: slide-left
title: Reinforcement Learning - BipedalWalker
mdc: true
---

# Reinforcement Learning
## Training Agents for BipedalWalker-v3

<div class="pt-12">
  <span class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Project 2 - Stable Baselines3 Implementation
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <span class="text-sm opacity-50">SAC • DDPG • PPO • TD3</span>
</div>

---
transition: fade-out
---

# Project Overview

<div class="grid grid-cols-2 gap-8">

<div>

## Objectives

- ✅ Implement **4 RL algorithms** (SAC, DDPG, PPO, TD3)
- ✅ Train on **BipedalWalker-v3** environment
- ✅ Provide **unfolded algorithms** with pseudocode
- ✅ Create **visualizations** and graphical results
- ✅ Compare algorithm performance

</div>

<div>

## Tools & Libraries

- **Gymnasium** - RL environments
- **Stable Baselines3** - RL algorithms
- **PyTorch** - Deep learning backend
- **Matplotlib/Seaborn** - Visualization
- **TensorBoard** - Training monitoring

</div>

</div>

<div class="mt-8 p-4 bg-green-500 bg-opacity-20 rounded-lg">
  <strong>Hardware:</strong> 2× NVIDIA GeForce RTX 5090 GPUs
</div>

---
layout: two-cols
---

# Environment: BipedalWalker-v3

<div class="pr-4">

## Description

A bipedal robot learning to walk on rough terrain.

## State Space
- **24 dimensions**: hull angle, angular velocity, leg joints, ground contact, lidar

## Action Space
- **4 continuous actions**: joint motor controls
- Range: [-1, 1]

## Reward
- +300 for reaching the goal
- Penalties for falling, energy usage

</div>

::right::

<div class="pl-4">

```python
import gymnasium as gym

env = gym.make("BipedalWalker-v3")

# Observation: 24D vector
obs_shape = env.observation_space.shape  
# (24,)

# Action: 4 motor controls
action_shape = env.action_space.shape    
# (4,)

# Continuous actions
action_low = env.action_space.low        
# [-1, -1, -1, -1]
action_high = env.action_space.high      
# [1, 1, 1, 1]
```

<div class="mt-4 text-center">
  <strong class="text-green-400">Solved at reward ≥ 300</strong>
</div>

</div>

---
layout: center
class: text-center
---

# Algorithms Implemented

<div class="grid grid-cols-4 gap-4 mt-8">
  <div class="p-4 bg-red-500 bg-opacity-30 rounded-lg">
    <h3 class="text-xl font-bold">SAC</h3>
    <p class="text-sm opacity-75">Soft Actor-Critic</p>
  </div>
  <div class="p-4 bg-purple-500 bg-opacity-30 rounded-lg">
    <h3 class="text-xl font-bold">DDPG</h3>
    <p class="text-sm opacity-75">Deep Deterministic PG</p>
  </div>
  <div class="p-4 bg-cyan-500 bg-opacity-30 rounded-lg">
    <h3 class="text-xl font-bold">PPO</h3>
    <p class="text-sm opacity-75">Proximal Policy Opt.</p>
  </div>
  <div class="p-4 bg-yellow-500 bg-opacity-30 rounded-lg">
    <h3 class="text-xl font-bold">TD3</h3>
    <p class="text-sm opacity-75">Twin Delayed DDPG</p>
  </div>
</div>

---

# Algorithm 1: SAC (Soft Actor-Critic)

<div class="grid grid-cols-2 gap-8">

<div>

## Key Features
- **Off-policy** actor-critic
- **Maximum entropy** framework
- Automatic temperature tuning
- Twin Q-networks for stability

## Objective Function

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E} [r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))]$$

Where $\alpha$ controls the entropy bonus.

</div>

<div>

## Pseudocode

```
Initialize: π_θ, Q_φ1, Q_φ2, targets, buffer D

for each iteration:
  # Collect experience
  a_t ~ π_θ(a_t|s_t)
  Store (s, a, r, s') in D
  
  # Update critics
  y = r + γ(min(Q'_1, Q'_2) - α log π)
  φ_i ← φ_i - λ∇(Q_φi - y)²
  
  # Update actor
  θ ← θ - λ∇(α log π - min(Q_1, Q_2))
  
  # Soft update targets
  φ'_i ← τφ_i + (1-τ)φ'_i
```

</div>

</div>

---

# Algorithm 2: DDPG (Deep Deterministic Policy Gradient)

<div class="grid grid-cols-2 gap-8">

<div>

## Key Features
- **Deterministic** policy
- Off-policy with experience replay
- Target networks for stability
- Ornstein-Uhlenbeck noise for exploration

## Policy Gradient

$$\nabla_\theta J \approx \mathbb{E}[\nabla_a Q(s,a)|_{a=\mu(s)} \nabla_\theta \mu(s)]$$

</div>

<div>

## Pseudocode

```
Initialize: μ_θ, Q_φ, targets μ'_θ, Q'_φ, buffer D

for each episode:
  for each timestep:
    # Action with exploration noise
    a_t = μ_θ(s_t) + ε
    
    # Store transition
    D ← D ∪ (s, a, r, s')
    
    # Sample minibatch
    y = r + γ Q'_φ(s', μ'_θ(s'))
    
    # Update critic
    φ ← φ - λ∇(Q_φ - y)²
    
    # Update actor
    θ ← θ + λ∇_a Q_φ(s, μ_θ(s))
    
    # Soft update
    θ' ← τθ + (1-τ)θ'
    φ' ← τφ + (1-τ)φ'
```

</div>

</div>

---

# Algorithm 3: PPO (Proximal Policy Optimization)

<div class="grid grid-cols-2 gap-8">

<div>

## Key Features
- **On-policy** algorithm
- Clipped surrogate objective
- GAE for advantage estimation
- Multiple epochs per batch
- Simple and robust

## Clipped Objective

$$L^{CLIP} = \mathbb{E}_t[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

Where $r_t = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$

</div>

<div>

## Pseudocode

```
Initialize: π_θ, V_φ

for each iteration:
  # Collect trajectories
  Run π_θ for T timesteps
  
  # Compute advantages (GAE)
  δ_t = r_t + γV(s_{t+1}) - V(s_t)
  Â_t = Σ (γλ)^k δ_{t+k}
  
  for k epochs:
    for each minibatch:
      # Probability ratio
      r_t = π_θ(a|s) / π_old(a|s)
      
      # Clipped objective
      L = min(r_t Â_t, clip(r_t) Â_t)
      
      # Update policy
      θ ← θ + ∇L
      
      # Update value function
      φ ← φ - ∇(V_φ - V_target)²
```

</div>

</div>

---

# Algorithm 4: TD3 (Twin Delayed DDPG)

<div class="grid grid-cols-2 gap-8">

<div>

## Key Improvements over DDPG
1. **Twin Critics**: Two Q-networks, use minimum
2. **Delayed Updates**: Update policy less frequently
3. **Target Smoothing**: Add noise to target actions

## Target Computation

$$y = r + \gamma \min_{i=1,2} Q'_{\phi_i}(s', \tilde{a})$$

Where $\tilde{a} = \pi'(s') + \text{clip}(\epsilon, -c, c)$

</div>

<div>

## Pseudocode

```
Initialize: π_θ, Q_φ1, Q_φ2, targets, buffer D

for each timestep:
  # Exploration
  a = π_θ(s) + ε,  ε ~ N(0, σ)
  
  # Store and sample
  D ← (s, a, r, s')
  Sample batch from D
  
  # Target with clipped noise
  ã = π'(s') + clip(ε, -c, c)
  y = r + γ min(Q'_1(s',ã), Q'_2(s',ã))
  
  # Update both critics
  φ_i ← φ_i - ∇(Q_φi - y)²
  
  if t mod d = 0:  # Delayed update
    # Update actor
    θ ← θ + ∇Q_φ1(s, π_θ(s))
    
    # Soft update targets
    θ' ← τθ + (1-τ)θ'
    φ'_i ← τφ_i + (1-τ)φ'_i
```

</div>

</div>

---
layout: center
---

# Training Configuration

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

## Hyperparameters

| Parameter | SAC | DDPG | PPO | TD3 |
|-----------|-----|------|-----|-----|
| Learning Rate | 3e-4 | 1e-3 | 3e-4 | 3e-4 |
| Buffer Size | 100k | 100k | - | 100k |
| Batch Size | 256 | 256 | 64 | 256 |
| Gamma (γ) | 0.99 | 0.99 | 0.99 | 0.99 |
| Tau (τ) | 0.005 | 0.005 | - | 0.005 |

</div>

<div>

## Training Setup

```python
TOTAL_TIMESTEPS = 100_000

ALGORITHMS = {
    'SAC': SAC,
    'DDPG': DDPG,
    'PPO': PPO,
    'TD3': TD3
}

# Custom callback for tracking
callback = TrainingCallback(
    check_freq=5000,
    verbose=1
)
```

</div>

</div>

---

# Training Results

<div class="grid grid-cols-2 gap-8">

<div>

## Training Time

| Algorithm | Time (seconds) |
|-----------|---------------|
| **PPO** ⚡ | **143.4s** |
| TD3 | 314.8s |
| DDPG | 351.4s |
| SAC | 648.8s |

<div class="mt-4 p-3 bg-blue-500 bg-opacity-20 rounded">
  PPO is 4.5× faster than SAC!
</div>

</div>

<div>

## Evaluation Results (10 episodes)

| Algorithm | Mean Reward | Std Dev |
|-----------|-------------|---------|
| **PPO** 🏆 | **+116.29** | 5.14 |
| SAC | -102.00 | 9.22 |
| TD3 | -106.86 | 4.52 |
| DDPG | -169.15 | 19.30 |

<div class="mt-4 p-3 bg-green-500 bg-opacity-20 rounded">
  Only PPO achieved positive reward!
</div>

</div>

</div>

---
layout: image-right
image: training_results.png
---

# Learning Curves

## Observations

1. **PPO** shows steady improvement
2. **SAC** learns slower but more stable
3. **TD3** has high variance initially
4. **DDPG** struggles with exploration

## Key Insights

- On-policy (PPO) works well for this task
- Off-policy methods need more samples
- 100k timesteps is insufficient for convergence
- Recommend 500k-1M for better results

---
layout: image-right
image: evaluation_results.png
---

# Evaluation Comparison

## Best Performer: PPO 🏆

- **Mean Reward**: +116.29
- **Std Deviation**: 5.14
- **Fastest Training**: 143.4s

## Runner-up: SAC

- More sample efficient long-term
- Needs more training time

## Most Stable: TD3

- Lowest standard deviation (4.52)
- Consistent but negative reward

---
layout: image-right
image: agent_behavior.png
---

# Agent Behavior

## PPO Agent Visualization

- 500 steps per episode
- Frame captured every 25 steps
- Shows walking progression

## Observations

- Agent maintains balance
- Forward locomotion learned
- Some wobbling present
- Not yet optimal gait

---

# Key Findings

<div class="grid grid-cols-3 gap-6">

<div class="p-4 bg-green-500 bg-opacity-20 rounded-lg text-center">
  <div class="text-4xl mb-2">🏆</div>
  <h3 class="font-bold">Best Performer</h3>
  <p class="text-2xl text-green-400">PPO</p>
  <p class="text-sm opacity-75">Reward: +116.29</p>
</div>

<div class="p-4 bg-blue-500 bg-opacity-20 rounded-lg text-center">
  <div class="text-4xl mb-2">⚡</div>
  <h3 class="font-bold">Fastest Training</h3>
  <p class="text-2xl text-blue-400">PPO</p>
  <p class="text-sm opacity-75">143.4 seconds</p>
</div>

<div class="p-4 bg-purple-500 bg-opacity-20 rounded-lg text-center">
  <div class="text-4xl mb-2">🎯</div>
  <h3 class="font-bold">Most Stable</h3>
  <p class="text-2xl text-purple-400">TD3</p>
  <p class="text-sm opacity-75">Std: 4.52</p>
</div>

</div>

<div class="mt-8 p-4 bg-yellow-500 bg-opacity-20 rounded-lg">

### Algorithm Comparison Summary

| Metric | SAC | DDPG | PPO | TD3 |
|--------|-----|------|-----|-----|
| Final Reward | -102.54 | -132.42 | **-71.33** | -123.10 |
| Eval Reward | -102.00 | -169.15 | **+116.29** | -106.86 |
| Training Time | 648.8s | 351.4s | **143.4s** | 314.8s |

</div>

---

# Recommendations

<div class="grid grid-cols-2 gap-8">

<div>

## For This Task (BipedalWalker)

1. **Use PPO** for best results
   - Fastest training
   - Best performance
   - Simple to tune

2. **Increase training steps**
   - Current: 100k steps
   - Recommended: 500k-1M steps
   - Goal: Reach 300+ reward

3. **Hyperparameter tuning**
   - Learning rate: 1e-4 to 3e-4
   - Batch size: 64-256
   - GAE lambda: 0.95-0.99

</div>

<div>

## General Insights

### When to use each algorithm:

| Algorithm | Best For |
|-----------|----------|
| **SAC** | Sample efficiency, continuous control |
| **DDPG** | Simple deterministic policies |
| **PPO** | Stability, robustness, parallelization |
| **TD3** | Improved DDPG, better stability |

### Future Improvements
- Parallel environments (n_envs > 1)
- Curriculum learning
- Reward shaping
- Domain randomization

</div>

</div>

---
layout: center
class: text-center
---

# Conclusion

<div class="text-xl mt-8">

✅ Implemented **4 RL algorithms**: SAC, DDPG, PPO, TD3

✅ Trained on **BipedalWalker-v3** using GPU acceleration

✅ Provided **unfolded algorithms** with detailed pseudocode

✅ Created **visualizations**: learning curves, comparisons, agent behavior

✅ **PPO** emerged as the best performer (+116.29 reward)

</div>

<div class="mt-12 p-6 bg-green-500 bg-opacity-20 rounded-lg inline-block">
  <strong>Project Requirements: Fully Satisfied ✓</strong>
</div>

---
layout: center
class: text-center
---

# Thank You!

<div class="text-lg opacity-75 mt-4">
  Project 2 - Reinforcement Learning
</div>

<div class="mt-8 grid grid-cols-4 gap-4 text-sm">
  <div class="p-2 bg-white bg-opacity-10 rounded">
    <strong>SAC</strong>
    <br>648.8s
  </div>
  <div class="p-2 bg-white bg-opacity-10 rounded">
    <strong>DDPG</strong>
    <br>351.4s
  </div>
  <div class="p-2 bg-green-500 bg-opacity-30 rounded">
    <strong>PPO 🏆</strong>
    <br>143.4s
  </div>
  <div class="p-2 bg-white bg-opacity-10 rounded">
    <strong>TD3</strong>
    <br>314.8s
  </div>
</div>

<div class="abs-br m-6 text-sm opacity-50">
  Powered by Stable Baselines3 & PyTorch
</div>

---
layout: end
---

# Questions?

<div class="text-center mt-8">
  <p class="text-lg opacity-75">
    All code and results available in the Jupyter notebook
  </p>
  <code class="text-sm">p1/rl_training.ipynb</code>
</div>

