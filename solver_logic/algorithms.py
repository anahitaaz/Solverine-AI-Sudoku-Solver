import numpy as np
import random
import time
from collections import defaultdict
from solver_logic.utilities import grid_to_string, get_grid_dimensions

# --- ALGORITHM METRIC HOLDERS ---
# Global or class-level counters are needed to track metrics across recursive calls.
# We will pass a metrics dictionary to each recursive solver.

# ====================================================================
# 1. BACKTRACKING (BT) - Deterministic Search
# ====================================================================

def is_safe(grid, r, c, num, n, bx, by):
    """Checks if placing 'num' at (r, c) is valid according to Sudoku rules."""
    
    # Check row and column
    if num in grid[r, :] or num in grid[:, c]:
        return False

    # Check box
    start_row, start_col = r - r % bx, c - c % by
    if num in grid[start_row:start_row + bx, start_col:start_col + by]:
        return False

    return True

def find_empty_cell(grid, n):
    """Finds the next empty cell (row, col)."""
    for r in range(n):
        for c in range(n):
            if grid[r, c] == 0:
                return r, c
    return -1, -1 # Indicates the grid is full (solved)

def solve_backtracking(grid, n, bx, by, metrics, use_mrv=False):
    """Recursive Backtracking Sudoku Solver with optional MRV heuristic."""
    
    metrics['nodes_expanded'] += 1
    
    # Use MRV heuristic for 10x10 or when explicitly requested
    if use_mrv or n == 10:
        r, c = find_mrv_cell(grid, n, bx, by)
        if r == -1:
            return True  # Solved
        # Get possible values for MRV cell
        possible_values = sorted(list(get_possible_values(grid, r, c, n, bx, by)))
        if not possible_values:
            return False  # Dead end
    else:
        r, c = find_empty_cell(grid, n)
        if r == -1:
            return True  # Solved
        possible_values = range(1, n + 1)

    # Iterate through possible values
    for num in possible_values:
        if is_safe(grid, r, c, num, n, bx, by):
            
            # Place the number
            grid[r, c] = num
            metrics['branches'] += 1 # A successful placement that leads to recursion is a branch

            # Recurse
            if solve_backtracking(grid, n, bx, by, metrics, use_mrv):
                return True

            # Backtrack: Reset the cell and continue the loop
            grid[r, c] = 0

    return False # No solution found from this path

def backtracking_solver(initial_grid, n):
    """Wrapper for the Backtracking Solver."""
    _, bx, by = get_grid_dimensions(f"{n}x{n}")
    
    # Create a deep copy to modify
    grid = initial_grid.copy()
    metrics = {'branches': 0, 'nodes_expanded': 0, 'iterations': 0}
    
    start_time = time.time()
    # Use MRV for 10x10 puzzles for better performance
    success = solve_backtracking(grid, n, bx, by, metrics, use_mrv=(n == 10))
    end_time = time.time()
    
    # Execution time is handled by the solve_and_benchmark wrapper
    
    solved_str = grid_to_string(grid) if success else grid_to_string(initial_grid)
    return solved_str, metrics, success

# ====================================================================
# 2. CONSTRAINT SATISFACTION (CSP) - BT with MRV Heuristic
# ====================================================================

def get_possible_values(grid, r, c, n, bx, by):
    """Returns a set of possible values for cell (r, c)."""
    possible = set(range(1, n + 1))
    
    # Remove values in row and column
    possible -= set(grid[r, :])
    possible -= set(grid[:, c])
    
    # Remove values in box
    start_row, start_col = r - r % bx, c - c % by
    box_values = grid[start_row:start_row + bx, start_col:start_col + by].flatten()
    possible -= set(box_values)
    
    return possible

def find_mrv_cell(grid, n, bx, by):
    """Finds the cell with the Minimum Remaining Values (MRV) heuristic."""
    min_r, min_c, min_count = -1, -1, n + 1
    
    for r in range(n):
        for c in range(n):
            if grid[r, c] == 0:
                possible = get_possible_values(grid, r, c, n, bx, by)
                count = len(possible)
                
                if count == 0:
                    return r, c # Return this cell immediately if it has zero options (dead end)
                
                if count < min_count:
                    min_count = count
                    min_r, min_c = r, c
                    
    return min_r, min_c

def forward_check(grid, r, c, num, n, bx, by):
    """Forward checking: returns False if assigning num to (r,c) leaves any cell with no options."""
    # Check all cells in the same row, column, and box
    affected_cells = set()
    
    # Same row
    for col in range(n):
        if col != c and grid[r, col] == 0:
            affected_cells.add((r, col))
    
    # Same column
    for row in range(n):
        if row != r and grid[row, c] == 0:
            affected_cells.add((row, c))
    
    # Same box
    start_row, start_col = r - r % bx, c - c % by
    for br in range(start_row, start_row + bx):
        for bc in range(start_col, start_col + by):
            if (br != r or bc != c) and grid[br, bc] == 0:
                affected_cells.add((br, bc))
    
    # Temporarily assign the value
    grid[r, c] = num
    
    # Check if any affected cell has no possible values
    for ar, ac in affected_cells:
        possible = get_possible_values(grid, ar, ac, n, bx, by)
        if len(possible) == 0:
            grid[r, c] = 0  # Restore before returning
            return False
    
    grid[r, c] = 0  # Restore
    return True

def solve_csp(grid, n, bx, by, metrics, use_forward_checking=False):
    """CSP Sudoku Solver using Backtracking, MRV heuristic, and optional forward checking."""
    
    metrics['nodes_expanded'] += 1
    
    r, c = find_mrv_cell(grid, n, bx, by)

    if r == -1:
        return True # Solved

    possible_values = sorted(list(get_possible_values(grid, r, c, n, bx, by)))

    if not possible_values:
        return False # Dead end (MRV found a cell with 0 options)

    # Iterate through possible values (MRV-based order)
    for num in possible_values:
        # Since we use get_possible_values, this step implicitly checks is_safe
        
        # Forward checking: only use for 10x10 puzzles (adds overhead for smaller puzzles)
        if use_forward_checking and not forward_check(grid, r, c, num, n, bx, by):
            continue
        
        grid[r, c] = num
        metrics['branches'] += 1
        
        # Recurse
        if solve_csp(grid, n, bx, by, metrics, use_forward_checking):
            return True

        # Backtrack
        grid[r, c] = 0

    return False

def csp_solver(initial_grid, n):
    """Wrapper for the CSP Solver."""
    _, bx, by = get_grid_dimensions(f"{n}x{n}")
    
    grid = initial_grid.copy()
    metrics = {'branches': 0, 'nodes_expanded': 0, 'iterations': 0}
    
    # Only use forward checking for 10x10 puzzles (it adds overhead for smaller puzzles)
    use_forward_checking = (n == 10)
    success = solve_csp(grid, n, bx, by, metrics, use_forward_checking)
    
    solved_str = grid_to_string(grid) if success else grid_to_string(initial_grid)
    return solved_str, metrics, success

# ====================================================================
# 3. DANCING LINKS / EXACT COVER (DLX)
# (Simplified self-contained implementation based on Algorithm X)
# ====================================================================

class DLXNode:
    """A node in the toroidal doubly-linked list structure."""
    def __init__(self, col=None):
        self.L = self.R = self.U = self.D = self
        self.C = col
        self.row_id = -1 # Used to link back to the Sudoku solution

class Column(DLXNode):
    """A column header node, tracking size (number of 1s in the column)."""
    def __init__(self, name):
        super().__init__(self)
        self.name = name
        self.size = 0

class DLX:
    """Dancing Links (DLX) implementation for Exact Cover."""
    
    def __init__(self, n):
        self.n = n
        self.root = Column("root")
        self.solution = []
        self.metrics = {'branches': 0, 'nodes_expanded': 0, 'iterations': 0}
        self.solved_grid = None
        self.success = False

    def build_constraints(self, grid):
        """Builds the 4 types of constraints (4 * N^2 total columns) and the matrix rows."""
        _, self.bx, self.by = get_grid_dimensions(f"{self.n}x{self.n}")
        N = self.n
        
        # 1. Initialize Column Headers
        col_names = []
        # Constraint 1: Cell (r, c) must contain a value
        for r in range(N):
            for c in range(N):
                col_names.append(f"P({r},{c})")
        # Constraint 2: Row (r) must contain value (v) once
        for r in range(N):
            for v in range(1, N + 1):
                col_names.append(f"R({r},{v})")
        # Constraint 3: Col (c) must contain value (v) once
        for c in range(N):
            for v in range(1, N + 1):
                col_names.append(f"C({c},{v})")
        # Constraint 4: Block (b) must contain value (v) once
        for b in range(N):
            for v in range(1, N + 1):
                col_names.append(f"B({b},{v})")

        header_nodes = [Column(name) for name in col_names]
        
        # Link Column Headers to the root
        current = self.root
        for col_header in header_nodes:
            col_header.L = current
            col_header.R = current.R
            current.R.L = col_header
            current.R = col_header
            current = col_header
            
        col_map = {name: header for name, header in zip(col_names, header_nodes)}
        
        # 2. Build Rows based on Sudoku possibilities
        for r in range(N):
            for c in range(N):
                if grid[r, c] == 0: # If cell is empty, create N rows of possibilities
                    for v in range(1, N + 1):
                        self._add_row(r, c, v, col_map)
                else: # If cell is pre-filled, only create one row for that fixed value
                    v = grid[r, c]
                    self._add_row(r, c, v, col_map)

    def _add_row(self, r, c, v, col_map):
        """Helper to create a row of nodes based on a single placement (r, c, v)."""
        N = self.n
        blocks_per_row = N // self.by
        b = (r // self.bx) * blocks_per_row + (c // self.by)
        
        col_keys = [
            f"P({r},{c})", # Position Constraint
            f"R({r},{v})", # Row-Value Constraint
            f"C({c},{v})", # Column-Value Constraint
            f"B({b},{v})"  # Block-Value Constraint
        ]
        
        first_node = None
        prev_node = None
        
        for key in col_keys:
            col = col_map[key]
            node = DLXNode(col)
            node.row_id = (r * N * N) + (c * N) + (v - 1)
            
            # Vertical Linkage
            node.U = col.U
            node.D = col
            col.U.D = node
            col.U = node
            col.size += 1
            
            # Horizontal Linkage
            if first_node is None:
                first_node = node
            else:
                node.L = prev_node
                node.R = prev_node.R
                prev_node.R.L = node
                prev_node.R = node
            prev_node = node

        # Close the circular list for the row
        if first_node and prev_node:
            first_node.L = prev_node
            prev_node.R = first_node

    def cover(self, col):
        """Removes the column and all rows containing a 1 in that column."""
        col.R.L = col.L
        col.L.R = col.R
        
        i = col.D
        while i != col:
            j = i.R
            while j != i:
                j.D.U = j.U
                j.U.D = j.D
                j.C.size -= 1
                j = j.R
            i = i.D

    def uncover(self, col):
        """Restores the column and all rows removed by the cover operation."""
        i = col.U
        while i != col:
            j = i.L
            while j != i:
                j.C.size += 1
                j.D.U = j
                j.U.D = j
                j = j.L
            i = i.U
        col.R.L = col
        col.L.R = col

    def search(self, k=0):
        """The recursive search function (Algorithm X)."""
        self.metrics['nodes_expanded'] += 1
        
        if self.root.R == self.root:
            self.success = True
            return True # Solution found
        
        # Choose column c: Select the column with the minimum size (S-Heuristic)
        c = self.root.R
        min_size = c.size
        j = c.R
        while j != self.root:
            if j.size < min_size:
                min_size = j.size
                c = j
            j = j.R
            
        self.cover(c)

        r = c.D
        while r != c:
            self.metrics['branches'] += 1
            self.solution.append(r)
            
            j = r.R
            while j != r:
                self.cover(j.C)
                j = j.R

            if self.search(k + 1):
                return True
            
            # Backtrack
            r_node = self.solution.pop()
            c_node = r_node.C
            
            j = r_node.L
            while j != r_node:
                self.uncover(j.C)
                j = j.L
            
            self.uncover(c_node)
            r = r.D # Continue to the next row
            
        return False

    def get_solved_grid_string(self, initial_grid):
        """Converts the internal solution into the solved grid string."""
        N = self.n
        solved_grid = initial_grid.copy()

        for node in self.solution:
            rid = int(node.row_id)

            v = (rid % N) + 1
            temp = rid // N
            c = temp % N
            r = temp // N  # == rid // (N*N)

            # Guardrail: ensure indices are in range
            if not (0 <= r < N and 0 <= c < N):
                raise ValueError(f"Decoded cell out of bounds: r={r}, c={c}, N={N}, row_id={rid}")

            solved_grid[r, c] = int(v)

        return grid_to_string(solved_grid)

def dlx_solver(initial_grid, n):
    """Wrapper for the Dancing Links Solver."""
    dlx_instance = DLX(n)
    
    # 1. Build the matrix
    dlx_instance.build_constraints(initial_grid)
    
    # 2. Search
    start_time = time.time()
    success = dlx_instance.search()
    end_time = time.time()
    
    solved_str = dlx_instance.get_solved_grid_string(initial_grid) if success else grid_to_string(initial_grid)
    
    metrics = dlx_instance.metrics
    print(f"=> After solving, the DLX result is {solved_str}")
    return solved_str, metrics, success

# ====================================================================
# 4. SIMULATED ANNEALING (SA) - Stochastic Search
# ====================================================================

def calculate_fitness(grid, n, bx, by):
    """Fitness is the negative count of constraint violations (we minimize violations)."""
    violations = 0
    
    # 1. Row/Col Violations
    for i in range(n):
        # Row violations
        row_counts = np.bincount(grid[i, :])
        violations += sum(count - 1 for count in row_counts[1:] if count > 1)
        
        # Column violations
        col_counts = np.bincount(grid[:, i])
        violations += sum(count - 1 for count in col_counts[1:] if count > 1)

    # 2. Block Violations
    for r in range(0, n, bx):
        for c in range(0, n, by):
            block = grid[r:r + bx, c:c + by].flatten()
            block_counts = np.bincount(block)
            violations += sum(count - 1 for count in block_counts[1:] if count > 1)
            
    return -violations # Return as negative (SA tries to maximize fitness, target = 0)

def sa_solver(initial_grid, n, max_iterations=None, initial_temp=None, cooling_rate=None):
    # Adaptive parameters for larger grids (10x10)
    if max_iterations is None:
        max_iterations = 500000 if n == 10 else 200000
    if initial_temp is None:
        initial_temp = 2.0 if n == 10 else 1.5
    if cooling_rate is None:
        cooling_rate = 0.9999 if n == 10 else 0.9997
    
    bx, by = get_grid_dimensions(f"{n}x{n}")[1:]
    fixed = initial_grid != 0
    grid = initial_grid.copy()

    # --- Block-wise initialization: fill each block with its missing digits
    for r0 in range(0, n, bx):
        for c0 in range(0, n, by):
            block = grid[r0:r0+bx, c0:c0+by].copy()
            missing = [v for v in range(1, n+1) if v not in block]
            random.shuffle(missing)
            k = 0
            for r in range(r0, r0+bx):
                for c in range(c0, c0+by):
                    if not fixed[r, c]:
                        grid[r, c] = missing[k]
                        k += 1

    # Row/col conflict fitness (blocks assumed valid)
    def rc_conflicts(g):
        conflicts = 0
        for i in range(n):
            row = g[i, :]
            col = g[:, i]
            # duplicates count as count-1
            conflicts += sum(max(0, np.bincount(row, minlength=n+1)[v]-1) for v in range(1, n+1))
            conflicts += sum(max(0, np.bincount(col, minlength=n+1)[v]-1) for v in range(1, n+1))
        return -conflicts  # maximize

    current = rc_conflicts(grid)
    best_grid = grid.copy()
    best = current
    temp = initial_temp

    # Precompute mutable indices per block for fast neighbor generation
    mutable_in_block = {}
    for r0 in range(0, n, bx):
        for c0 in range(0, n, by):
            cells = [(r, c) for r in range(r0, r0+bx) for c in range(c0, c0+by) if not fixed[r, c]]
            if len(cells) >= 2:
                mutable_in_block[(r0, c0)] = cells

    for it in range(max_iterations):
        # pick a random block that has at least two mutable cells
        if not mutable_in_block:
            break
        (r0, c0), cells = random.choice(list(mutable_in_block.items()))
        (r1, c1), (r2, c2) = random.sample(cells, 2)

        candidate = grid.copy()
        candidate[r1, c1], candidate[r2, c2] = candidate[r2, c2], candidate[r1, c1]

        new = rc_conflicts(candidate)

        if new > current or (temp > 0 and random.random() < np.exp((new - current) / temp)):
            grid, current = candidate, new
            if new > best:
                best, best_grid = new, candidate.copy()

        temp *= cooling_rate
        if best == 0:
            return grid_to_string(best_grid), {'branches': 0, 'nodes_expanded': 0, 'iterations': it+1}, True

    success = (best == 0)
    return grid_to_string(best_grid if success else best_grid), {'branches': 0, 'nodes_expanded': 0, 'iterations': max_iterations}, success


# ====================================================================
# 5. GENETIC ALGORITHM (GA) - Stochastic Search
# ====================================================================

def ga_solver(initial_grid, n, population_size=None, max_generations=None, mutation_rate=None):
    """Genetic Algorithm Sudoku Solver."""
    # Adaptive parameters for larger grids (10x10)
    if population_size is None:
        population_size = 200 if n == 10 else 100
    if max_generations is None:
        max_generations = 5000 if n == 10 else 2000
    if mutation_rate is None:
        mutation_rate = 0.08 if n == 10 else 0.05
    
    bx, by = get_grid_dimensions(f"{n}x{n}")[1:]
    fixed_cells = initial_grid != 0
    
    metrics = {'branches': 0, 'nodes_expanded': 0, 'iterations': 0}

    # 1. Initialization
    def generate_individual():
        """Creates an individual (grid) that satisfies the box constraints."""
        individual = initial_grid.copy()
        for r_start in range(0, n, bx):
            for c_start in range(0, n, by):
                block = individual[r_start:r_start + bx, c_start:c_start + by].flatten()
                
                # Find missing numbers (1 to N)
                available = list(set(range(1, n + 1)) - set(block))
                random.shuffle(available)
                
                # Fill in the zeros/empty cells within the block
                k = 0
                for r in range(r_start, r_start + bx):
                    for c in range(c_start, c_start + by):
                        if individual[r, c] == 0:
                            individual[r, c] = available[k]
                            k += 1
        return individual

    population = [generate_individual() for _ in range(population_size)]

    # 2. Main Loop
    for gen in range(max_generations):
        metrics['iterations'] += 1
        
        fitness_scores = [calculate_fitness(ind, n, bx, by) for ind in population]
        best_fitness = max(fitness_scores)
        best_index = np.argmax(fitness_scores)
        best_individual = population[best_index]

        if best_fitness == 0:
            return grid_to_string(best_individual), metrics, True # Solved!

        # Selection (Tournament Selection)
        def select_parent(pop, scores):
            """Selects the best of two random individuals."""
            idx1, idx2 = random.sample(range(len(pop)), 2)
            if scores[idx1] > scores[idx2]:
                return pop[idx1]
            return pop[idx2]

        new_population = [best_individual.copy()] # Elitism: Keep the best one
        
        while len(new_population) < population_size:
            parent1 = select_parent(population, fitness_scores)
            parent2 = select_parent(population, fitness_scores)
            
            # Crossover (Block-based Crossover)
            child = parent1.copy()
            # Swap a random block from parent2 into child
            r_start = random.choice(range(0, n, bx))
            c_start = random.choice(range(0, n, by))
            
            block_slice = (slice(r_start, r_start + bx), slice(c_start, c_start + by))
            child[block_slice] = parent2[block_slice]
            
            # Mutation (Swap two random non-fixed cells within a block)
            if random.random() < mutation_rate:
                block_r = random.choice(range(0, n, bx))
                block_c = random.choice(range(0, n, by))
                
                mutable_indices = []
                for r in range(block_r, block_r + bx):
                    for c in range(block_c, block_c + by):
                        if not fixed_cells[r, c]:
                            mutable_indices.append((r, c))
                
                if len(mutable_indices) >= 2:
                    (r1, c1), (r2, c2) = random.sample(mutable_indices, 2)
                    child[r1, c1], child[r2, c2] = child[r2, c2], child[r1, c1]

            new_population.append(child)
            
        population = new_population

    # If loop finishes, return the best individual found
    final_fitness = calculate_fitness(best_individual, n, bx, by)
    success = final_fitness == 0
    final_grid = best_individual if success else initial_grid
    
    return grid_to_string(final_grid), metrics, success


# Mapping of algorithm names to their functions
ALGORITHMS = {
    "Backtracking": backtracking_solver,
    "CSP": csp_solver,
    "DLX": dlx_solver,
    "SA": sa_solver,
    "GA": ga_solver,
}
