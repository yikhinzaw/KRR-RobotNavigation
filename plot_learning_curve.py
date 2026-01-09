import pickle
import numpy as np
import matplotlib.pyplot as plt

def moving_avg(x, w):
    x = np.asarray(x)
    if len(x) < 2:
        return x
    w = min(w, len(x))
    return np.convolve(x, np.ones(w)/w, mode="valid")

with open("q_table.pkl", "rb") as f:
    data = pickle.load(f)
q_rewards = np.array(data["q_rewards"])
rand_rewards = np.array(data["rand_rewards"])

window = 100
q_smooth = moving_avg(q_rewards, window)
rand_smooth = moving_avg(rand_rewards, window)

# use same x-length as smoothed arrays
episodes_smooth = np.arange(len(q_smooth)) + 1

print("q_rewards shape:", q_rewards.shape)
print("rand_rewards shape:", rand_rewards.shape)

# Expect shapes like (N,) for both
if q_rewards.ndim != 1 or rand_rewards.ndim != 1:
    raise ValueError("q_rewards and rand_rewards must be 1D arrays of per-episode returns.")

episodes = np.arange(1, len(q_rewards) + 1)

# Simple case: plot raw curves first to verify
if len(q_rewards) != len(rand_rewards):
    print("Warning: lengths differ: Q =", len(q_rewards), "Random =", len(rand_rewards))
    # Align to min length
    n = min(len(q_rewards), len(rand_rewards))
    q_rewards = q_rewards[:n]
    rand_rewards = rand_rewards[:n]
    episodes = episodes[:n]

plt.figure(figsize=(10,5))
#plt.plot(episodes_smooth, q_smooth, label="Q-learning (mean)", color="tab:blue")
plt.plot(episodes_smooth, rand_smooth, label="Random (mean)", color="tab:orange")
plt.xlabel("Episode")
plt.ylabel("Mean total reward")
plt.title("Learning Curve (smoothed)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()