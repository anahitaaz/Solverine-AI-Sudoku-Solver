import time
import os
import psutil
import json
import numpy as np
import random
from datetime import datetime

# --- GRID UTILITIES ---

# Only support 4x4, 5x5, and 9x9 grids
GRID_DIMS = {
    "4x4": (2, 2), # 2x2 blocks (total 16 cells)
    "5x5": (1, 5), # 1x5 block of 5x5 cells (total 25 cells)
    "9x9": (3, 3), # 3x3 blocks (total 81 cells)
}

def get_grid_dimensions(grid_size_str):
    """Returns the dimensions (N) and block structure (bx, by). Only supports 4x4, 5x5, 9x9."""
    if grid_size_str not in GRID_DIMS:
        raise ValueError(f"Unsupported grid size: {grid_size_str}. Only 4x4, 5x5, and 9x9 are supported.")
    
    bx, by = GRID_DIMS[grid_size_str]
    n = bx * by
    return n, bx, by

def string_to_grid(puzzle_str, n):
    """Converts a flat string into an N x N numpy array."""
    puzzle_str = puzzle_str.replace('.', '0')
    if len(puzzle_str) != n * n:
        raise ValueError(f"Input string length ({len(puzzle_str)}) does not match expected size ({n*n}) for {n}x{n} grid.")
    
    # Convert to integers
    try:
        data = []
        for c in puzzle_str:
            if c.isdigit():
                data.append(int(c))
            else:
                data.append(0)  # Unknown character becomes 0
        return np.array(data, dtype=int).reshape(n, n)
    except ValueError as e:
        raise ValueError(f"Invalid character in puzzle string: {e}")

def grid_to_string(grid):
    """Converts an N x N numpy array back to a flat string."""
    return "".join(map(str, grid.flatten()))

def generate_mock_solved_grid(n):
    """Generates a simple mock solved grid for demonstration."""
    # This is a placeholder. In a real system, you'd use a puzzle generator.
    grid = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            grid[i, j] = (i * n + j) % n + 1
    return grid

def _to_plain(obj):
    """Recursively convert numpy scalars/arrays and datetimes to JSON-friendly Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _to_plain(x) for x in obj ]
    # NumPy scalars
    if isinstance(obj, np.generic):
        return obj.item()
    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def calculate_approx_memory_kb(data_object):
    """Approximates the memory usage of the result object in KB."""
    safe = _to_plain(data_object)
    return len(json.dumps(safe)) / 1024.0

# --- CORE BENCHMARK WRAPPER ---

def solve_and_benchmark(solver_func, initial_grid_str, grid_size_str, algorithm_name, user_id):
    """
    Executes the given solver function and collects all performance metrics.
    Only supports 4x4, 5x5, and 9x9 grids.
    
    Args:
        solver_func: The function implementing the solving algorithm.
        initial_grid_str (str): The unsolved Sudoku puzzle string.
        grid_size_str (str): The selected grid size (must be "4x4", "5x5", or "9x9").
        algorithm_name (str): Name of the algorithm (e.g., "DLX").
        user_id (str): The ID of the current user.

    Returns:
        dict: A standardized result dictionary including metrics.
    """
    # Validate grid size first
    if grid_size_str not in ['4x4', '5x5', '9x9']:
        return {
            "error": f"Grid size {grid_size_str} not supported. Only 4x4, 5x5, and 9x9 are allowed.",
            "grid_size": grid_size_str,
            "algorithm": algorithm_name,
            "input": initial_grid_str,
            "success": False,
            "solved": "N/A",
            "execution_time_ms": 0.0,
            "memory_kb": 0.0,
            "branches": 0,
            "nodes_expanded": 0,
            "iterations": 0,
            "timestamp": datetime.now().isoformat(),
            "userId": user_id
        }
    
    print(f"The fn value is {get_grid_dimensions(grid_size_str)}")
    [n, bx, by] = get_grid_dimensions(grid_size_str)
    print(f"=> Working on {grid_size_str} and {algorithm_name}")
    
    try:
        initial_grid = string_to_grid(initial_grid_str, n)
        print(f"Got the initial grid as {initial_grid}")
    except ValueError as e:
        print("Encountered an error, moving in the exception block!")
        # Return an error result if input is invalid
        return {
            "error": str(e),
            "grid_size": grid_size_str,
            "algorithm": algorithm_name,
            "input": initial_grid_str,
            "success": False,
            "solved": "N/A",
            "execution_time_ms": 0.0,
            "memory_kb": 0.0,
            "branches": 0,
            "nodes_expanded": 0,
            "iterations": 0,
            "timestamp": datetime.now().isoformat(),
            "userId": user_id
        }

    # Instrumentation Setup
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss 

    # --- EXECUTE SOLVER ---
    try:
        # Solvers must return (solved_grid_string, metrics_dict, is_successful)
        solved_grid_str, metrics, success = solver_func(initial_grid, n)
    except Exception as e:
        print(f"Algorithm error: {e}")
        return {
            "error": f"Solver failed: {e}",
            "grid_size": grid_size_str,
            "algorithm": algorithm_name,
            "input": initial_grid_str,
            "success": False,
            "solved": "N/A",
            "execution_time_ms": (time.time() - start_time) * 1000,
            "memory_kb": (process.memory_info().rss - start_memory) / 1024,
            "branches": metrics.get("branches", 0) if isinstance(metrics, dict) else 0,
            "nodes_expanded": metrics.get("nodes_expanded", 0) if isinstance(metrics, dict) else 0,
            "iterations": metrics.get("iterations", 0) if isinstance(metrics, dict) else 0,
            "timestamp": datetime.now().isoformat(),
            "userId": user_id
        }
    # --- END EXECUTE SOLVER ---

    # Instrumentation Finalization
    end_time = time.time()
    end_memory = process.memory_info().rss
    
    execution_time_ms = (end_time - start_time) * 1000
    memory_kb = (end_memory - start_memory) / 1024 # Convert bytes difference to KB

    solved_matrix = string_to_grid(solved_grid_str, n).tolist()

    # Create the standardized result record
    result = {
        "grid_size": grid_size_str,
        "algorithm": algorithm_name,
        "input": initial_grid_str,
        "solved": solved_grid_str,
        "success": success,
        "execution_time_ms": round(execution_time_ms, 3),
        "memory_kb": round(memory_kb, 3),
        # Pull metrics from the algorithm's return value
        "branches": metrics.get("branches", 0),
        "nodes_expanded": metrics.get("nodes_expanded", 0),
        "iterations": metrics.get("iterations", 0),
        "timestamp": datetime.now().isoformat(),
        "userId": user_id,
        "solved_matrix": solved_matrix
    }
    
    # Final sanity check for memory (ensure it's not negative due to OS effects)
    if result["memory_kb"] < 0:
        result["memory_kb"] = calculate_approx_memory_kb(result)

    # Make sure the whole payload is JSON-friendly (and consistent for downstream)
    return _to_plain(result)