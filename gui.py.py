import pygame
import heapq
import time
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 50
GRID_WIDTH = 20
GRID_HEIGHT = 15

# Info panel width (on the right)
INFO_WIDTH = 400

# Window size: grid on left, info on right
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE + INFO_WIDTH
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)
PURPLE = (147, 112, 219)
PANEL_BG = (245, 245, 250)

# Directions
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Cat & Mouse Chase - A* vs BFS")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 34)

        # Game state
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.mouse_pos = (1, 1)
        self.cat_pos = (GRID_HEIGHT - 2, GRID_WIDTH - 2)
        self.cheese_pos = (GRID_HEIGHT // 2, GRID_WIDTH // 2)

        # Algorithm mode
        self.mode = 'astar'  # 'astar' or 'bfs'
        self.running = True
        self.paused = True
        self.game_over = False
        self.won = False

        # Stats
        self.moves = 0
        self.astar_wins = 0
        self.bfs_wins = 0
        self.astar_losses = 0
        self.bfs_losses = 0

        # Algorithm metrics
        self.astar_time = 0.0
        self.bfs_time = 0.0
        self.astar_steps = 0
        self.bfs_steps = 0

        # Animation
        self.path_cache = []
        self.speed = 5  # Moves per second
        self.last_move_time = 0

        # Maze
        self.generate_maze()

    def generate_maze(self):
        import random
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        # Add border walls
        for i in range(GRID_HEIGHT):
            self.grid[i][0] = 1
            self.grid[i][GRID_WIDTH - 1] = 1
        for j in range(GRID_WIDTH):
            self.grid[0][j] = 1
            self.grid[GRID_HEIGHT - 1][j] = 1

        # Add random internal walls
        for _ in range(50):
            x = random.randint(2, GRID_HEIGHT - 3)
            y = random.randint(2, GRID_WIDTH - 3)
            self.grid[x][y] = 1

        # Clear important cells
        self.mouse_pos = (1, 1)
        self.cat_pos = (GRID_HEIGHT - 2, GRID_WIDTH - 2)
        self.cheese_pos = (GRID_HEIGHT // 2, GRID_WIDTH // 2)
        for pos in [self.mouse_pos, self.cat_pos, self.cheese_pos]:
            self.grid[pos[0]][pos[1]] = 0

    def reset_game(self):
        self.generate_maze()
        self.moves = 0
        self.game_over = False
        self.won = False
        self.paused = True
        self.path_cache = []

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < GRID_HEIGHT and 0 <= y < GRID_WIDTH

    def passable(self, pos):
        x, y = pos
        return self.in_bounds(pos) and self.grid[x][y] != 1

    def neighbors(self, pos):
        for dx, dy in DIRS:
            np = (pos[0] + dx, pos[1] + dy)
            if self.passable(np):
                yield np

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self, start, goal):
        if start == goal:
            return [start], 0.0, 0
        import time
        start_time = time.perf_counter()
        open_heap = []
        heapq.heappush(open_heap, (self.manhattan(start, goal), 0, start, [start]))
        visited = {start: 0}
        steps = 0
        while open_heap:
            steps += 1
            f, g, current, path = heapq.heappop(open_heap)
            if current == goal:
                elapsed = (time.perf_counter() - start_time) * 1000
                return path, elapsed, steps
            for nb in self.neighbors(current):
                g2 = g + 1
                if nb not in visited or g2 < visited[nb]:
                    visited[nb] = g2
                    heapq.heappush(open_heap, (g2 + self.manhattan(nb, goal), g2, nb, path + [nb]))
        elapsed = (time.perf_counter() - start_time) * 1000
        return None, elapsed, steps

    def bfs(self, start, goal):
        if start == goal:
            return [start], 0.0, 0
        import time
        start_time = time.perf_counter()
        q = deque([(start, [start])])
        seen = {start}
        steps = 0
        while q:
            steps += 1
            cur, path = q.popleft()
            if cur == goal:
                elapsed = (time.perf_counter() - start_time) * 1000
                return path, elapsed, steps
            for nb in self.neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, path + [nb]))
        elapsed = (time.perf_counter() - start_time) * 1000
        return None, elapsed, steps

    def greedy_chase(self, chaser_pos, target_pos):
        best = None
        best_d = 10**9
        for nb in self.neighbors(chaser_pos):
            d = self.manhattan(nb, target_pos)
            if d < best_d:
                best_d = d
                best = nb
        return best if best is not None else chaser_pos

    def update_game(self):
        if self.paused or self.game_over:
            return
        current_time = pygame.time.get_ticks()
        if current_time - self.last_move_time < 1000 / self.speed:
            return
        self.last_move_time = current_time
        self.moves += 1

        # Mouse move
        if self.mode == 'astar':
            if not self.path_cache:
                path, time_taken, steps = self.a_star(self.mouse_pos, self.cheese_pos)
                self.astar_time = time_taken
                self.astar_steps = steps
                self.path_cache = path
                if not self.path_cache:
                    self.game_over = True
                    return
            if len(self.path_cache) >= 2:
                self.mouse_pos = self.path_cache[1]
                self.path_cache = self.path_cache[1:]
        else:
            if not self.path_cache:
                path, time_taken, steps = self.bfs(self.mouse_pos, self.cheese_pos)
                self.bfs_time = time_taken
                self.bfs_steps = steps
                self.path_cache = path
                if not self.path_cache:
                    self.game_over = True
                    return
            if len(self.path_cache) >= 2:
                self.mouse_pos = self.path_cache[1]
                self.path_cache = self.path_cache[1:]

        # Win check
        if self.mouse_pos == self.cheese_pos:
            self.game_over = True
            self.won = True
            if self.mode == 'astar':
                self.astar_wins += 1
            else:
                self.bfs_wins += 1
            return

        # Cat move
        self.cat_pos = self.greedy_chase(self.cat_pos, self.mouse_pos)

        # Lose check
        if self.cat_pos == self.mouse_pos:
            self.game_over = True
            self.won = False
            if self.mode == 'astar':
                self.astar_losses += 1
            else:
                self.bfs_losses += 1

    def draw(self):
        self.screen.fill(WHITE)

        # Grid offset (left area)
        grid_offset_x = 0

        # Draw grid (left)
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                rect = pygame.Rect(grid_offset_x + j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[i][j] == 1:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                else:
                    pygame.draw.rect(self.screen, LIGHT_BLUE, rect)
                pygame.draw.rect(self.screen, GRAY, rect, 1)

        # Draw path highlights (if any) on the grid (left)
        if self.path_cache:
            for pos in self.path_cache:
                rect = pygame.Rect(grid_offset_x + pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                surf.fill((200, 255, 200, 120))  # translucent
                self.screen.blit(surf, rect.topleft)

        # Draw cheese (on grid)
        cheese_rect = pygame.Rect(grid_offset_x + self.cheese_pos[1] * CELL_SIZE + 5,
                                  self.cheese_pos[0] * CELL_SIZE + 5,
                                  CELL_SIZE - 10, CELL_SIZE - 10)
        pygame.draw.rect(self.screen, YELLOW, cheese_rect)
        pygame.draw.rect(self.screen, ORANGE, cheese_rect, 3)

        # Draw mouse
        mouse_rect = pygame.Rect(grid_offset_x + self.mouse_pos[1] * CELL_SIZE + 5,
                                 self.mouse_pos[0] * CELL_SIZE + 5,
                                 CELL_SIZE - 10, CELL_SIZE - 10)
        pygame.draw.circle(self.screen, GRAY, mouse_rect.center, CELL_SIZE // 3)
        pygame.draw.circle(self.screen, GRAY, (mouse_rect.centerx - 8, mouse_rect.centery - 8), 5)
        pygame.draw.circle(self.screen, GRAY, (mouse_rect.centerx + 8, mouse_rect.centery - 8), 5)

        # Draw cat
        cat_rect = pygame.Rect(grid_offset_x + self.cat_pos[1] * CELL_SIZE + 5,
                               self.cat_pos[0] * CELL_SIZE + 5,
                               CELL_SIZE - 10, CELL_SIZE - 10)
        pygame.draw.circle(self.screen, ORANGE, cat_rect.center, CELL_SIZE // 3)

        # --- Info Panel (Right) ---
        panel_x = GRID_WIDTH * CELL_SIZE
        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, 0, INFO_WIDTH, WINDOW_HEIGHT))

        # Title and mode
        title = self.title_font.render("Cat & Mouse Chase", True, PURPLE)
        self.screen.blit(title, (panel_x + 16, 18))

        algo_text = "A* Algorithm" if self.mode == 'astar' else "BFS Algorithm"
        algo = self.font.render(f"Mode: {algo_text}", True, BLUE)
        self.screen.blit(algo, (panel_x + 16, 70))

        # Stats block
        stats_lines = [
            f"Moves: {self.moves}",
            f"Speed: {self.speed}x",
            "",
            f"A* W/L: {self.astar_wins}/{self.astar_losses}",
            f"BFS W/L: {self.bfs_wins}/{self.bfs_losses}",
            "",
            "Controls:",
            "Space - Play / Pause",
            "R - Reset",
            "M - Switch Mode",
            "+ / - - Speed"
        ]
        y = 110
        for line in stats_lines:
            surf = self.font.render(line, True, BLACK)
            self.screen.blit(surf, (panel_x + 16, y))
            y += 26

        # Algorithm details and metrics
        y += 10  # Small gap
        if self.mode == 'astar':
            algo_detail = [
                "A* Algorithm Details:",
                "A* finds the shortest path by combining",
                "actual cost (g) and estimated cost (h).",
                "Heuristic: Manhattan distance.",
                "Efficient for large grids, guarantees",
                "optimal path if heuristic is admissible.",
                "",
                f"Time: {self.astar_time:.2f}ms"
            ]
        else:
            algo_detail = [
                "BFS Algorithm Details:",
                "BFS explores all neighbors level by level.",
                "Guarantees shortest path in unweighted",
                "grids. No heuristics, explores broadly",
                "before deep. Slower on large grids.",
                "",
                f"Time: {self.bfs_time:.2f}ms"
            ]
        for line in algo_detail:
            surf = self.font.render(line, True, BLACK)
            self.screen.blit(surf, (panel_x + 16, y))
            y += 26

        # If game over -> overlay only on grid area
        if self.game_over:
            overlay = pygame.Surface((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
            overlay.set_alpha(160)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))

            if self.won:
                msg = "MOUSE WINS! Got the cheese!"
                color = GREEN
            else:
                msg = "CAT WINS! Caught the mouse!"
                color = RED

            text = self.title_font.render(msg, True, color)
            text_rect = text.get_rect(center=(GRID_WIDTH * CELL_SIZE // 2, GRID_HEIGHT * CELL_SIZE // 2))
            self.screen.blit(text, text_rect)

            restart = self.font.render("Press R to restart", True, WHITE)
            restart_rect = restart.get_rect(center=(GRID_WIDTH * CELL_SIZE // 2, GRID_HEIGHT * CELL_SIZE // 2 + 40))
            self.screen.blit(restart, restart_rect)

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_m:
                    self.mode = 'bfs' if self.mode == 'astar' else 'astar'
                    self.reset_game()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.speed = min(20, self.speed + 1)
                elif event.key == pygame.K_MINUS:
                    self.speed = max(1, self.speed - 1)

    def run(self):
        while self.running:
            self.handle_events()
            self.update_game()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    Game().run()