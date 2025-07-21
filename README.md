# 🧠 What Matters for Maximizing Data Reuse in Value-based Deep Reinforcement Learning

This repository implements the paper:  
**_What Matters for Maximizing Data Reuse In Value-based Deep Reinforcement Learning_**  
by **Roger Creus Castanyer**, **Glen Berseth**, and **Pablo Samuel Castro**  
Conducted at **Université de Montréal**, **Mila – Quebec AI Institute**, and **Google DeepMind**.

---

## 🚀 Included Algorithms

- `dqn_td0.py`: Standard DQN with TD(0) updates  
- `dqn_rollout.py`: DQN using Q(λ)
- `pqn.py`: PQN baseline
- `ppo.py`: PPO baseline
- `r2d2.py`: R2D2 agent  

---

## 📦 Installation

Make sure you're using a recent nightly build of PyTorch:

```bash
python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 
```

Then, install all required dependencies:

```bash
pip install -r requirements.txt
```
