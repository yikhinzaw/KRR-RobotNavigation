import numpy as np
import pickle
import random
from collections import deque
#Constants
GRID_W, GRID_H = 6, 6
N_ACTIONS = 4
START = (0, 0)

ALPHA, GAMMA = 0.1, 0.99
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 5000
EPISODES = 40000
MAX_STEPS = 500

TARGET_COLLECT_RATE = 0.85
TARGET_SURVIVAL_RATE = 0.90
RECENT_EPS = 100
#Constants

ALPHA_START = 0.2  # Start higher to learn the new "expensive" costs quickly
ALPHA_END = 0.01   # End lower to stabilize the optimal path
GAMMA = 0.99

def get_alpha(episode):
    """Linearly decays alpha from ALPHA_START to ALPHA_END."""
    # Logic: reduce alpha based on progress through total episodes
    return ALPHA_START - (episode / EPISODES) * (ALPHA_START - ALPHA_END)
                                                 
def state_to_idx(s):
    return s[1] * GRID_W + s[0]

def generate_random_env_old():
    all_positions = [(x,y) for x in range(GRID_W) for y in range(GRID_H)]
    forbidden = {START}

    n_treasures = random.randint(1, GRID_W-1)
    treasures = set()
    while len(treasures) < n_treasures:
        pos = random.choice([p for p in all_positions if p not in forbidden])
        treasures.add(pos)
        forbidden.add(pos)

    n_traps = random.randint(1, GRID_W-1)
    traps = set()
    while len(traps) < n_traps:
        pos = random.choice([p for p in all_positions if p not in forbidden])
        traps.add(pos)
        forbidden.add(pos)

    return list(treasures), list(traps)
def generate_random_env():
    all_positions = [(x, y) for x in range(GRID_W) for y in range(GRID_H)]
    forbidden = {START}

    # More treasures
    n_treasures = random.randint(6, 10)
    treasures = set()
    while len(treasures) < n_treasures:
        pos = random.choice([p for p in all_positions if p not in forbidden])
        treasures.add(pos)
        forbidden.add(pos)

    # More traps
    n_traps = random.randint(8, 12)
    traps = set()
    while len(traps) < n_traps:
        pos = random.choice([p for p in all_positions if p not in forbidden])
        traps.add(pos)
        forbidden.add(pos)

    return list(treasures), list(traps)

def epsilon_by_step(t):
    frac = min(1.0, t / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)

def step_env(state, action, treasures, traps):
    """Environment transition + raw reward. + Prioritize speed"""
    x, y = state
    if action == 0: y = max(0, y-1)# up
    elif action == 1: x = min(GRID_W-1, x+1)# right
    elif action == 2: y = min(GRID_H-1, y+1) # down
    elif action == 3: x = max(0, x-1) # left

    new_state = (x, y)
    # 1. Increased Step Cost: Higher penalty per move makes time "expensive"
    reward = -2       # step cost
    done = False
    # 2. Strict Boundary Penalty: Moving into a wall wastes time
    if new_state == state:
        # Tried to move into wall / out of bounds
        # Optional: extra penalty to discourage useless actions
        reward -= 10.0  # Increased penalty for useless actions
    # 3. High Treasure Reward: Ensures the agent doesn't become "suicidal" to avoid step costs
    if new_state in treasures:
        treasures.remove(new_state)
        reward += 20 # Increased to compensate for higher step costs

    # 4. Fatal Trap Penalty
    if new_state in traps:
        traps.remove(new_state)
        reward -= 20
        done = True
    # episode ends when all treasures collected
    if not treasures:
        done = True
 
    return new_state, reward, done, treasures[:], traps[:]

def q_update(q, si, a, r, s2i, done,alpha):
    """Q-learning style reward-backed update:a dynamic alpha.
        Q <- Q + α[r + γ max_a' Q(s',a') - Q]."""
    best_next = 0.0 if done else np.max(q[s2i])
    td_target = r + GAMMA * best_next
    td_error = td_target - q[si, a]
    q[si, a] += alpha * td_error

def train():
    q_rewards_history = []
    rand_rewards_history = []
    treasures, traps = generate_random_env()
    q = np.zeros((GRID_W * GRID_H, N_ACTIONS), dtype=np.float32)
    global_step = 0
    # ... [Deques for tracking stats] ...
    recent_rewards   = deque(maxlen=RECENT_EPS)
    recent_completes = deque(maxlen=RECENT_EPS)
    recent_survivals = deque(maxlen=RECENT_EPS)

    for ep in range(EPISODES):
        # Calculate current alpha for this episode
        current_alpha = get_alpha(ep)
        
        s = START
        curr_treasures = treasures[:]
        curr_traps = traps[:]
        # ... [Episode initialization] ...
        episode_reward = 0.0
        complete = True   # collected all treasures
        survival = True   # did not hit trap

        for t in range(MAX_STEPS):
            global_step += 1
            eps = epsilon_by_step(global_step)
            si = state_to_idx(s)

            # ε-greedy policy from Q-table
            if np.random.rand() < eps:
                a = np.random.randint(N_ACTIONS)
            else:
                a = np.argmax(q[si])

            # environment step produces reward
            s2, r, done, curr_treasures, curr_traps = step_env(
                s, a, curr_treasures, curr_traps
            )
            s2i = state_to_idx(s2)

            # Q-learning style update using this reward
            q_update(q, si, a, r, s2i, done,current_alpha)

            s = s2
            episode_reward += r

            if done:# or t == MAX_STEPS - 1:
                 # complete only if no treasures left
                #complete = (len(curr_treasures) == 0)
                 # survival concept becomes optional since traps are non-terminal
                #break
                # if any traps remain, we *might* have ended by trap or by collecting all;
                # decide completion/survival based on state:
                if s in traps:     # trap cell
                    survival = False
                    complete = False
                elif curr_treasures:  # ended early without collecting all
                    complete = False
                break
        
         # --- simulate random agent on the same treasures/traps layout ---
        rand_state = START
        rand_treasures = treasures[:]     # same layout, fresh lists
        rand_traps = traps[:]
        rand_total_reward = 0

        for t in range(MAX_STEPS):
             # random action
            a_rand = np.random.randint(N_ACTIONS)
            rand_state2, r_rand, rand_done, rand_treasures, rand_traps = step_env(
                rand_state, a_rand, rand_treasures, rand_traps
            )
            rand_total_reward += r_rand
            rand_state = rand_state2
            if rand_done:
                break

        rand_rewards_history.append(float(rand_total_reward))       
        q_rewards_history.append(float(episode_reward))
        recent_rewards.append(episode_reward)
        recent_completes.append(1 if complete else 0)
        recent_survivals.append(1 if survival else 0)

        if ep % 1000 == 0:
            collect_rate  = float(np.mean(recent_completes)) if recent_completes else 0.0
            survival_rate = float(np.mean(recent_survivals)) if recent_survivals else 0.0
            avg_reward    = float(np.mean(recent_rewards))   if recent_rewards   else 0.0
            print(f"Ep {ep}: complete={collect_rate:.2f}, "
                  f"survival={survival_rate:.2f}, rew={avg_reward:.1f}")

            if collect_rate >= TARGET_COLLECT_RATE and survival_rate >= TARGET_SURVIVAL_RATE:
                print("Targets achieved!")
                break

   
    # Final stats for logging/visualization
    collect_rate  = float(np.mean(recent_completes)) if recent_completes else 0.0
    survival_rate = float(np.mean(recent_survivals)) if recent_survivals else 0.0
    avg_reward    = float(np.mean(recent_rewards))   if recent_rewards   else 0.0

    env_data = {
        "q_table": q,
        "treasures": treasures,
        "traps": traps,
        "q_rewards": q_rewards_history,
        "rand_rewards": rand_rewards_history,
        "n_treasures": len(treasures),
        "n_traps": len(traps),
        "final_collect": collect_rate,
        "final_survival": survival_rate,
        "final_avg_reward": avg_reward,
    }
    with open("q_table.pkl", "wb") as f:
        pickle.dump(env_data, f)
    print("Saved q_table.pkl")

if __name__ == "__main__":
    train()
