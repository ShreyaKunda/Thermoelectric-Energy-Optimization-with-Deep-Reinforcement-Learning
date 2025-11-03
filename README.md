# Thermoelectric-Energy-Optimization-with-Deep-Reinforcement-Learning

## Overview

This project explores how deep reinforcement learning (DRL) can improve energy management in thermoelectric generator (TEG) systems. A custom Gymnasium environment models the coupled thermal and electrical dynamics of a TEG, and multiple DRL agents learn to balance charging, buffer storage, and direct load delivery to maximize energy efficiency while preserving battery health.

## Key Features

- Custom continuous and discrete TEG environments compatible with Gymnasium v0.26+.
- Support for five Stable-Baselines3 algorithms: PPO, A2C, DDPG, SAC, and DQN.
- Automated hyperparameter tuning with Optuna, including pruning of invalid PPO configurations.
- End-to-end training, evaluation, and benchmarking with reproducible logging and plotting utilities.
- GPU-aware training loop that automatically shifts to CUDA when available.

## Repository Layout

- `RL.ipynb` – main interactive workflow for tuning, training, and evaluating all agents.
- `Output/` – generated artifacts; each algorithm folder contains optimized and benchmark checkpoints, evaluation logs, and plots.
- `test.py` – placeholder for future unit tests or scripted experimentation.

## Environment Setup

1. Install Python 3.9+ and create an isolated environment (recommended: `python -m venv .venv`).
2. Activate the environment and install dependencies:

   ```powershell
   pip install gymnasium stable-baselines3 optuna torch torchvision torchaudio numpy pandas matplotlib ipywidgets
   ```

   Adjust the Torch installation command if you require a GPU-enabled build; see the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
3. (Optional) Enable notebook widgets support:

   ```powershell
   jupyter nbextension enable --py widgetsnbextension
   ```

## Run Training and Evaluation

1. Launch Jupyter and open `RL.ipynb`.
2. Execute the notebook sequentially. The final cell orchestrates Optuna tuning (default 50 trials per algorithm), trains the best models, evaluates them, and runs benchmark baselines with minimal hyperparameters.
3. Progress bars (ipywidgets) display Optuna and training status for each agent. Training automatically falls back to CPU if CUDA is unavailable.

> Tip: For long experiments, reduce `n_trials`, `timesteps_per_trial`, or `evaluation_timesteps` near the bottom of the notebook to shorten runs.

## Reviewing Results

- Optimized checkpoints: `Output/<ALGO>/best_model.pt`
- Benchmark checkpoints: `Output/<ALGO>/Benchmark/benchmark_model.pt`
- Evaluation metrics: CSV files stored alongside each checkpoint (`evaluation_logs.csv`).
- Visualization assets: Efficiency, demand fulfillment, cumulative reward, and comparative plots saved as PNGs in each directory.
- Metrics helper: `calculate_metrics` in `RL.ipynb` summarizes reward, battery health, regret, and demand fulfillment rates.

## Customization Ideas

- Extend the observation/action spaces to reflect additional physical constraints or battery chemistry.
- Swap in vectorized environments or multi-environment training via `SubprocVecEnv` for faster experiments.
- Add scripted unit tests in `test.py` to validate environment dynamics and reward calculations.
- Integrate additional RL algorithms (e.g., TD3) or apply transfer learning from pretrained policies.

## References

- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Optuna](https://optuna.org/)
- [PyTorch](https://pytorch.org/)

If you use this work in academic or commercial projects, please cite the relevant upstream libraries and consider including a reference to this repository.
