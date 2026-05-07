# Hierarchical Federated Reinforcement Learning for Privacy Preserving V2X Networks

## Overview
This project presents a scalable Hierarchical Federated Reinforcement Learning framework for intelligent vehicular communication systems.

The proposed architecture combines:
- Proximal Policy Optimization (PPO)
- Federated Learning (FL)
- Hierarchical RSU based aggregation
- Differential Privacy
- Vehicle to Vehicle (V2V) coordination
- Vehicle to Infrastructure (V2X) communication

The framework improves:
- Scalability
- Communication efficiency
- Convergence speed
- Privacy preservation
- Driving decision accuracy

---

## System Architecture

The architecture consists of three layers:

1. Vehicle Layer
   - Local PPO training
   - State observation
   - Action selection

2. RSU Layer
   - Regional aggregation
   - Communication reduction
   - Edge coordination

3. Cloud Layer
   - Global aggregation
   - Global model distribution
   - Federated synchronization

---

## Implemented Algorithms

- PPO
- FedAvg
- FedProx
- FedMedian
- Hierarchical Federated Reinforcement Learning (HFRL)

---

## Experimental Scenarios

| Scenario | Vehicles | RSUs | Road Length |
|----------|----------|------|-------------|
| Small | 200 | 20 | 10 km |
| Medium | 500 | 50 | 30 km |
| Large | 1000 | 100 | 50 km |
| Dense Urban | 2000 | 200 | 80 km |

---

## Performance Results

| Method | Reward | Collision Rate | Convergence |
|--------|--------|----------------|-------------|
| FedAvg | 73.4 ± 1.8 | 8.1% | 17 |
| FedProx | 76.2 ± 1.4 | 6.4% | 15 |
| FedMedian | 77.1 ± 1.2 | 5.9% | 14 |
| HFRL | 85.6 ± 0.9 | 3.0% | 9 |

---

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- Federated Learning
- Reinforcement Learning

---

## Repository Structure

```bash
envs/
federated/
results/
graphs/
train_rl.py
train_fedavg.py
train_fedprox.py
train_hierarchical_fedrl.py
```

---

## Research Contributions

- Hierarchical federated aggregation
- Privacy preserving vehicular learning
- Communication complexity reduction
- Improved convergence stability
- Large scale vehicular scalability

---

## Author

Hamza Mohammed Dayekh

Lebanese American University

---

## Supervisor

Dr. Mohamed Watfa
