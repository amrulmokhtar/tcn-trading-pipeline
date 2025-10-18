import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load hybrid trades file
trades = pd.read_csv('results/phase6_hybrid_trades.csv')

# Basic equity curve
trades['cum_r'] = trades['R_net'].cumsum()
plt.figure(figsize=(10,6))
plt.plot(trades['timestamp_utc'], trades['cum_r'], label='Equity Curve (R)')
plt.xlabel('Time')
plt.ylabel('Cumulative R')
plt.title('Hybrid Backtest Equity Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Risk metrics
returns = trades['R_net']
mean_r = returns.mean()
std_r = returns.std()
sharpe = np.sqrt(252) * mean_r / std_r if std_r != 0 else np.nan

# Max drawdown
cum_r = trades['cum_r']
roll_max = cum_r.cummax()
drawdown = roll_max - cum_r
max_dd = drawdown.max()
mar = cum_r.iloc[-1] / max_dd if max_dd != 0 else np.nan

# Sortino ratio
neg_std = returns[returns < 0].std()
sortino = np.sqrt(252) * mean_r / neg_std if neg_std != 0 else np.nan

print(f"Final Cumulative R: {cum_r.iloc[-1]:.2f}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Max Drawdown (R): {max_dd:.2f}")
print(f"MAR Ratio: {mar:.2f}")
