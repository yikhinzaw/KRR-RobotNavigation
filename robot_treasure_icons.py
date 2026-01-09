import pygame
import pickle
import numpy as np
import sys
import os
import q_train_random_multi
GRID_W, GRID_H = 6, 6
CELL_SIZE = 80
ICON_SIZE = 32
HUD_H=90
WIDTH, HEIGHT = GRID_W * CELL_SIZE, (GRID_H * CELL_SIZE)+HUD_H
N_ACTIONS = 4
HUD_COL = 0                 # x index
HUD_ROW = GRID_H - 1        # y index (bottom row)

BLACK, WHITE, GRAY = (0,0,0), (255,255,255), (80,80,80)
BLUE, GOLD, RED = (50,100,255), (255,215,0), (255,50,50)

def create_icon(surface, color, pattern="solid"):
    """Fallback: generate simple patterned icon"""
    surface.fill((0,0,0))
    surface.set_colorkey((0,0,0))
    
    if pattern == "robot":
        # Simple robot body/head/eyes
        pygame.draw.rect(surface, color, (8,12,16,16))  # body
        pygame.draw.circle(surface, color, (16,8), 6)   # head
        pygame.draw.circle(surface, WHITE, (12,6), 2)   # eyes
        pygame.draw.circle(surface, WHITE, (20,6), 2)
    elif pattern == "treasure":
        # Chest shape
        pygame.draw.rect(surface, color, (6,16,20,12))  # chest
        pygame.draw.polygon(surface, (139,69,19), [(8,16),(16,8),(24,16)])  # top
        pygame.draw.line(surface, BLACK, (10,20), (22,20), 2)  # lid crack
    elif pattern == "trap":
        # Spikes
        points = [(16,0),(24,16),(16,32),(8,16)]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.circle(surface, BLACK, (16,16), 8)  # pit
    
    pygame.draw.rect(surface, WHITE, (0,0,ICON_SIZE,ICON_SIZE), 1)  # border

def load_or_create_icon(filename):
    """Load PNG or create fallback"""
    path = f"assets/icons/{filename}"
    if os.path.exists(path):
        icon = pygame.image.load(path).convert_alpha()
    else:
        print(f"Creating fallback {filename}")
        os.makedirs("assets/icons", exist_ok=True)
        icon = pygame.Surface((ICON_SIZE, ICON_SIZE), pygame.SRCALPHA)
        if filename == "robot.png":
            create_icon(icon, BLUE, "robot")
        elif filename == "treasure.png":
            create_icon(icon, GOLD, "treasure")
        elif filename == "trap.png":
            create_icon(icon, RED, "trap")
        pygame.image.save(icon, path)
    icon = pygame.transform.scale(icon, (ICON_SIZE, ICON_SIZE))
    return icon

def state_to_idx(s):
    return s[1] * GRID_W + s[0]

def greedy_action(q, state):
    return int(np.argmax(q[state_to_idx(state)]))
def greedy_eps_policy(q, state, epsilon_eval=0.05):
    si = state_to_idx(state)
    if np.random.rand() < epsilon_eval:
        return np.random.randint(N_ACTIONS)
    row = q[si]
    max_q = np.max(row)
    best_actions = np.flatnonzero(row == max_q)
    return int(np.random.choice(best_actions))

def step(state, action, treasures, traps):
    def apply_action(s, a):
        x, y = s
        if action == 0: 
            y = max(0, y-1)
        elif action == 1:
            x = min(GRID_W-1, x+1)
        elif action == 2: 
            y = min(GRID_H-1, y+1)
        elif action == 3:
            x = max(0, x-1)
        return (x, y)

    new_state = apply_action(state, action)

     # Detect no movement (e.g. bumping into wall / border)
      # If no movement, pick another action that changes state
    if new_state == state:
        alt_actions = [a for a in range(N_ACTIONS) if a != action]
        new_action = np.random.choice(alt_actions)
        candidate = apply_action(state, new_action)
        if candidate != state:
            action = new_action
            new_state = candidate
            print(new_action)
    collected = new_state in treasures
    hit_trap = new_state in traps
    same_state = (new_state == state)  
    if collected:
        treasures.remove(new_state)
    if hit_trap:
        traps.remove(new_state)
    
    # empty move = no treasure, no trap
    empty_move = (not collected) and (not hit_trap)

    return new_state, collected, hit_trap,same_state,empty_move

def main():
    # Load Q-table
    with open("q_table.pkl", "rb") as f:
        data = pickle.load(f)
    q_table = data['q_table']
    treasures = data['treasures'][:]
    traps = data['traps']
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robot Navigation System using Q-Learning Algorithm")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)
    
    # Load/create icons
    robot_icon = load_or_create_icon("robot.png")
    treasure_icon = load_or_create_icon("treasure.png")
    trap_icon = load_or_create_icon("trap.png")
    
    state = (0, 0)
    score = 0
    steps = 0
    curr_treasures = treasures[:]
    curr_traps = traps[:]
    running = True
    auto_run = True # keep this True unless ALL treasures are collected

    #hit_trap = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: auto_run = not auto_run
                if event.key == pygame.K_r:
                    #state, score, steps, curr_treasures, hit_trap = (0,0), 0, 0, treasures[:], False
                    # reset episode, BUT DO NOT TOUCH traps
                    state = (0, 0)
                    score = 0
                    steps = 0
                    curr_treasures = treasures[:]
                    curr_traps =traps[:]
                    auto_run = True
        
        if auto_run and curr_treasures :#and not hit_trap
            action = greedy_eps_policy(q_table, state)#greedy_eps_policy(q_table, state)#
            state, collected, trap_hit,same_state,empty_move = step(state, action, curr_treasures, traps)
            # If you want to force a "meaningful" move (treasure or trap),
            # retry other actions when move is empty:
            attempts = 0
            while empty_move and attempts < N_ACTIONS - 1:
                # choose another action different from current
                alt_actions = [a for a in range(N_ACTIONS) if a != action]
                action = np.random.choice(alt_actions)
                state, collected, hit_trap, same_state, empty_move = step(
                    state, action, curr_treasures, traps
                )
                attempts += 1
            score -= 1
            steps += 1
            if collected: 
                print("Trap at finding treasure", state, "Q:", q_table[state_to_idx(state)])
                score += 10
            if trap_hit:
                print("Trap at", state, "Q:", q_table[state_to_idx(state)])
                score -= 10
                #hit_trap = True
                #auto_run = False
        if not curr_treasures:
             auto_run = False  # finished successfully
       
        screen.fill(BLACK)
        # Grid
        for x in range(GRID_W):
            for y in range(GRID_H):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRAY, rect, 2)
        
        # Trap icons
        for tx, ty in traps:
            icon_rect = trap_icon.get_rect(center=(tx*CELL_SIZE + CELL_SIZE//2, ty*CELL_SIZE + CELL_SIZE//2))
            screen.blit(trap_icon, icon_rect)
        
        # Remaining treasure icons
        for tx, ty in curr_treasures:
            icon_rect = treasure_icon.get_rect(center=(tx*CELL_SIZE + CELL_SIZE//2, ty*CELL_SIZE + CELL_SIZE//2))
            screen.blit(treasure_icon, icon_rect)
        
        # Robot icon
        rx, ry = state
        robot_rect = robot_icon.get_rect(center=(rx*CELL_SIZE + CELL_SIZE//2, ry*CELL_SIZE + CELL_SIZE//2))
        screen.blit(robot_icon, robot_rect)
        
        # UI
        status = "ALL Treasures FOUND! 🎉" if not curr_treasures else "Searching..."
        if hit_trap: status = "TRAP! 💥"
        lines = [
            f"Score: {score}",
            f"Collected: {len(treasures)-len(curr_treasures)}/{len(treasures)}",
            status,
            "SPACE:pause | R:restart"
        ]
        for i, line in enumerate(lines):
            color = GOLD if not curr_treasures else WHITE
            text = font.render(line, True, color)
            text_rect = text.get_rect()
            text_rect.topleft = (HUD_COL * CELL_SIZE + HUD_H, HUD_ROW * CELL_SIZE + HUD_H + i*text_rect.height)
            screen.blit(text, text_rect)
    
        pygame.display.flip()
        clock.tick(3)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
