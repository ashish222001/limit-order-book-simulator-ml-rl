# Limit Order Book Simulator with ML Prediction & RL Execution Optimization

## Overview
This project implements an exchange-style Limit Order Book (LOB) simulator integrated with machine learning and reinforcement learning components. The system models market microstructure and enables experimentation with predictive modeling and execution strategies.

## Features
- Price–time priority matching engine
- Market and limit order handling
- FIFO execution
- Partial fills and cancellations
- Bid–ask spread and depth tracking
- Snapshot-based feature extraction

## Machine Learning
- Logistic Regression baseline
- LSTM-based price direction prediction
- Feature engineering from LOB snapshots

## Reinforcement Learning
- Custom trading environment
- Deep Q-Network (DQN)
- Experience replay buffer
- Target network
- ε-greedy exploration
- PnL-based reward function

## Tech Stack
Python, NumPy, Pandas, Scikit-learn, TensorFlow

## Purpose
Research-oriented prototype for quantitative trading simulation and execution optimization.
