# maze_solver/

├── maze_solver/
│   ├── __init__.py
│   ├── maze.py
│   └── solver.py
├── tests/
│   ├── __init__.py
│   └── test_maze_solver.py
├── README.md
├── requirements.txt
└── main.py

# maze_solver/maze.py
import numpy as np
import random

class Maze:
    def __init__(self, width=20, height=20, complexity=0.75, density=0.75):
        """
        Generate a random maze using a cellular automata-like approach.
        
        Args:
            width (int): Width of the maze
            height (int): Height of the maze
            complexity (float): Complexity of the maze generation
            density (float): Density of walls in the maze
        """
        # Initialize the maze with walls
        self.width = width
        self.height = height
        self.maze = np.ones((height, width), dtype=np.uint8)
        
        # Make aisles using random walk
        z = np.zeros((height, width), dtype=np.uint8)
        z[0, :] = z[-1, :] = 1
        z[:, 0] = z[:, -1] = 1
        
        # Adjust complexity and density
        complexity = int(complexity * (5 * (height + width)))
        density = int(density * ((height // 2) * (width // 2)))
        
        # Random walk to create paths
        for _ in range(density):
            x, y = np.random.randint(0, height//2) * 2, np.random.randint(0, width//2) * 2
            self.maze[y, x] = 0
            for _ in range(complexity):
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx*2, y + dy*2
                    if 0 <= nx < width and 0 <= ny < height and self.maze[ny, nx] == 1:
                        self.maze[ny, nx] = 0
                        self.maze[y + dy, x + dx] = 0
                        x, y = nx, ny
                        break
        
        # Ensure start and end are open
        self.maze[1, 1] = 0
        self.maze[height-2, width-2] = 0
        
        self.start = (1, 1)
        self.end = (height-2, width-2)
    
    def is_walkable(self, point):
        """Check if a point is walkable."""
        y, x = point
        return (0 <= y < self.height and 
                0 <= x < self.width and 
                self.maze[y, x] == 0)
    
    def get_neighbors(self, point):
        """Get walkable neighboring points."""
        y, x = point
        neighbors = [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]
        return [p for p in neighbors if self.is_walkable(p)]

# maze_solver/solver.py
import heapq

class MazeSolver:
    @staticmethod
    def heuristic(a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    @staticmethod
    def solve_astar(maze):
        """
        Solve maze using A* algorithm.
        
        Args:
            maze (Maze): Maze object to solve
        
        Returns:
            list: Path from start to end, or None if no path exists
        """
        start, end = maze.start, maze.end
        
        # Priority queue to store frontier nodes
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            
            if current == end:
                break
            
            for next_node in maze.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if (next_node not in cost_so_far or 
                    new_cost < cost_so_far[next_node]):
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + MazeSolver.heuristic(end, next_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        # Reconstruct path
        if end not in came_from:
            return None
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path

# main.py
from maze_solver.maze import Maze
from maze_solver.solver import MazeSolver
import matplotlib.pyplot as plt

def visualize_maze_solution(maze, path):
    """Visualize the maze and solution path."""
    plt.figure(figsize=(10, 10))
    plt.imshow(maze.maze, cmap='binary', interpolation='nearest')
    
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], color='red', linewidth=3)
    
    plt.title('Maze Solution')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Create a random maze
    maze = Maze(width=30, height=30)
    
    # Solve the maze
    solution = MazeSolver.solve_astar(maze)
    
    if solution:
        print(f"Path found! Length: {len(solution)}")
        visualize_maze_solution(maze, solution)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()

# requirements.txt
numpy
matplotlib

# README.md
# A* Maze Solver

## Overview
This project implements an A* pathfinding algorithm to solve randomly generated mazes. It creates mazes using a cellular automata-like approach and finds the optimal path between start and end points.

## Features
- Random maze generation
- A* pathfinding algorithm
- Visualization of maze and solution path

## Installation
```bash
git clone https://github.com/yourusername/maze-solver.git
cd maze-solver
pip install -r requirements.txt
python main.py
```

## Dependencies
- NumPy
- Matplotlib

## License
MIT License
