# master-ia-tfm


# Equity Price Prediction using Artificial Intelligence and Algorithmic Trading

This repository contains the source code developed for the **Final Master’s Thesis (TFM)** titled:

> **Mejora de la predicción de cotizaciones de activos de renta variable mediante inteligencia artificial**

> or **"Enhancing Equity Price Prediction through Artificial Intelligence and Algorithmic Trading"**

by **Marc Delos**,  
Master’s Degree in Applied Artificial Intelligence,  
**Universidad Europea de Madrid (2025)**.

---

## 🎯 Project Overview

The project explores how **deep learning models** can improve the **forecasting of equity price movements**, aiming to support **speculative trading strategies**.

A **synthetic ETF** was constructed from five major technology stocks — *AAPL, MSFT, NVDA, IBM, and GOOG* — using hourly data between **2021 and 2024**.  
The study compares the performance of three AI architectures:

- **LSTM (Long Short-Term Memory)** — for capturing temporal dependencies  
- **N-BEATS** — for residual and trend decomposition  
- **TSFEDL Hybrid Model** — combining CNN and RNN components  

The models are evaluated using classical error metrics (*MAE, RMSE, MSE, R²*) and financial ones (*directional accuracy, simulated Sharpe ratio*).

---

# Installation 

```bash
# Create environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Allows jupyter to recognize my environment
pip install ipykernel

# Add the environment as a Jupyter Kernel
python -m ipykernel install --user --name=venv --display-name "Python (master)"

# Optional: install jupyter in the env
pip install notebook jupyter

# Install required modules
pip install -r requirements.txt
```
