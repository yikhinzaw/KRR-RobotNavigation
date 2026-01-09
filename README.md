 **q_train_random_multi.py.**  
 The main role of this file is to train the Q-learning agent (and optionally a random agent) on the multi‑treasure, multi‑trap gridworld and save the learned Q-table plus reward histories.  
 <br/>
   **step_env(state, action, treasures, traps) **: function updates robot position, checks treasures/traps, computes reward (+20 treasure, -20 trap, -10 no-move in your latest design), and returns next_state and done flag.  
   <br/>
   **q_update(q, si, a, r, s2i, done)** function  applies the standard Q-learning formula to update the Q-table for a given transition.  
   <br/>
   **train()** function that loops over episodes, uses ε‑greedy or greedy policy to select actions, calls step_env each step, updates Q via q_update, accumulates episode_reward, and logs per‑episode rewards into q_rewards_history (and rand_rewards_history for the random agent).  
  <br/>
   <br/>
 **robot_treasure_icons.py**  
This file’s main role is to load the trained Q-table, run the greedy (or near‑greedy) policy visually with Pygame, and show the robot, treasures, traps, and score on a grid.  
 <br/>
   **step(...)** function takes current state and an action. Computes new_state (handling walls so that actions that hit a border yield same_state).  
  Detects collected (if new_state in treasures), hit_trap (if new_state in traps), and same_state (no movement).  
   <br/>
  Returns new_state plus booleans so the main loop can update score with +20, -20, and -10 respectively.  
   <br/>
   **Main game loop**   
   <br/>
   Reads the Q-table, picks the best action for the current state.  
    <br/>
   Calls step(...) to move the robot and compute whether treasure/trap/no-move occurred.  
    <br/>
   Updates score: +20 for treasure, -20 for trap, -10 for no-move, while keeping the robot moving even after stepping on traps (no terminal on trap).  
    <br/>
​   Draws the grid, icons, and HUD (score, collected count, status text) each frame.  
 <br/>
 **​Episode/reset logic:** Resets the robot and regenerates or repositions treasures when all treasures are collected, without stopping the visualization, so you can observe continuous behavior of the learned policy  
  <br/>
   <br/>
 **plot_learning_curve.py:**
 This generate Q-learning curve trending the random agent over time as the policy improves
