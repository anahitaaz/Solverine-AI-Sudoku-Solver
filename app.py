import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import numpy as np
import random
from datetime import datetime

# Import the core logic modules
from solver_logic.utilities import solve_and_benchmark, get_grid_dimensions, string_to_grid, grid_to_string
from solver_logic.algorithms import ALGORITHMS

# Import enhanced analytics
from enhanced_analytics import render_enhanced_analytics_dashboard

# --- CONFIGURATION ---
DATA_FILE_PATH = "data/results.json"
PUZZLE_INPUT_KEY = "sudoku_input_str"

# --- DATA HANDLING ---

def load_results():
    """Loads all benchmarking results from the JSON file."""
    if not os.path.exists(DATA_FILE_PATH):
        os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)
        with open(DATA_FILE_PATH, 'w') as f:
            json.dump([], f)
        return pd.DataFrame()
    
    with open(DATA_FILE_PATH, 'r') as f:
        try:
            data = json.load(f)
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            numeric_cols = ['execution_time_ms', 'memory_kb', 'branches', 'nodes_expanded', 'iterations']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            return df
        except json.JSONDecodeError:
            st.error(f"Error reading JSON from {DATA_FILE_PATH}. File might be corrupted.")
            return pd.DataFrame()

def save_result(result_record):
    """Appends a new result to the JSON file."""
    df = load_results()
    new_row_df = pd.DataFrame([result_record])
    df_updated = pd.concat([df, new_row_df], ignore_index=True)
    df_updated.to_json(DATA_FILE_PATH, orient='records', indent=2)


# --- SOLVER INTERFACE FUNCTIONS ---
def draw_sudoku_grid(grid_data, n, container, grid_size_str=None):
    """Draws the Sudoku grid from either a flat string or a 2D matrix."""
    if not grid_data:
        container.empty()
        container.markdown("<p style='text-align: center; color: gray;'>No puzzle to display.</p>", unsafe_allow_html=True)
        return

    try:
        if isinstance(grid_data, str):
            grid = string_to_grid(grid_data, n)
        else:
            grid = np.array(grid_data, dtype=int)
            if grid.shape != (n, n):
                raise ValueError("Grid matrix has wrong shape")
    except Exception:
        container.empty()
        container.markdown("<p style='text-align: center; color: gray;'>Invalid puzzle data.</p>", unsafe_allow_html=True)
        return

    try:
        if grid_size_str:
            _, bx, by = get_grid_dimensions(grid_size_str)
        else:
            bx = int(np.sqrt(n)) if np.sqrt(n).is_integer() else 3
            by = bx
    except:
        bx = int(np.sqrt(n)) if np.sqrt(n).is_integer() else 3
        by = bx

    cell_size = min(480 / n, 50)
    max_width = n * cell_size

    html_grid = f"""
    <div style="display: grid; 
                grid-template-columns: repeat({n}, 1fr); 
                width: {max_width}px;
                max-width: 100%;
                margin: 0 auto;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border: 3px solid #374151;
                border-radius: 4px;">
    """

    for r in range(n):
        for c in range(n):
            val = str(grid[r, c])
            cell_content = val if val != '0' else '&nbsp;'

            if c == n - 1:
                border_right = 'none'
            elif (c + 1) % by == 0:
                border_right = '3px solid #374151'
            else:
                border_right = '1px solid #d1d5db'
            
            if r == n - 1:
                border_bottom = 'none'
            elif (r + 1) % bx == 0:
                border_bottom = '3px solid #374151'
            else:
                border_bottom = '1px solid #d1d5db'
            
            if c == 0:
                border_left = 'none'
            elif c % by == 0:
                border_left = '3px solid #374151'
            else:
                border_left = '1px solid #d1d5db'
            
            if r == 0:
                border_top = 'none'
            elif r % bx == 0:
                border_top = '3px solid #374151'
            else:
                border_top = '1px solid #d1d5db'

            cell_style = f"""
                display: flex;
                align-items: center;
                justify-content: center;
                height: {cell_size}px;
                width: {cell_size}px;
                font-size: {max(cell_size * 0.35, 12)}px;
                font-weight: 600;
                color: #1f2937;
                background-color: white;
                border-right: {border_right};
                border-bottom: {border_bottom};
                border-left: {border_left};
                border-top: {border_top};
            """

            html_grid += f"""<div style="{cell_style}">{cell_content}</div>"""

    html_grid += "</div>"
    container.markdown(html_grid, unsafe_allow_html=True)

def solve_all_algorithms(puzzle_str, grid_size_str, user_id):
    """Solves the puzzle with all algorithms and returns results."""
    all_results = []
    algorithm_names = list(ALGORITHMS.keys())
    
    for algorithm_name in algorithm_names:
        solver_func = ALGORITHMS.get(algorithm_name)
        if not solver_func:
            continue
        
        try:
            result = solve_and_benchmark(solver_func, puzzle_str, grid_size_str, algorithm_name, user_id)
            
            if result.get("error"):
                result["success"] = False
                result["solved"] = puzzle_str
            
            all_results.append(result)
            
            if result.get("success"):
                save_result(result)
        except Exception as e:
            error_result = {
                "algorithm": algorithm_name,
                "success": False,
                "error": str(e),
                "execution_time_ms": 0,
                "memory_kb": 0,
                "branches": 0,
                "nodes_expanded": 0,
                "iterations": 0,
                "solved": puzzle_str
            }
            all_results.append(error_result)
    
    return all_results

# --- COMPARISON VISUALIZATION FUNCTIONS ---

def create_comparison_bar_chart(results_data, metric_key, title, y_label, log_scale=False):
    """Creates a bar chart comparing algorithms for a specific metric."""
    if not results_data:
        return None

    rows = []
    for r in results_data:
        alg = r.get('algorithm', 'Unknown')
        raw_val = r.get(metric_key, 0)
        success = bool(r.get('success', False))

        try:
            if metric_key in ('execution_time_ms', 'memory_kb'):
                val_num = float(raw_val)
            else:
                val_num = int(raw_val)
        except Exception:
            val_num = 0

        rows.append({'Algorithm': alg, 'Value': val_num, 'Success': success})

    df = pd.DataFrame(rows)

    try:
        from solver_logic.algorithms import ALGORITHMS as _ALGS
        alg_order = list(_ALGS.keys())
    except Exception:
        alg_order = sorted(df['Algorithm'].unique())

    fig = px.bar(
        df,
        x='Algorithm',
        y='Value',
        color='Success',
        category_orders={'Algorithm': alg_order},
        title=title,
        labels={'Value': y_label, 'Algorithm': 'Algorithm'},
        color_discrete_map={True: '#10b981', False: '#ef4444'},
    )

    if metric_key == 'execution_time_ms':
        text_tmpl = '%{y:.2f} ms'
    elif metric_key == 'memory_kb':
        text_tmpl = '%{y:.2f} KB'
    else:
        text_tmpl = '%{y:,}'

    fig.update_traces(
        texttemplate=text_tmpl,
        textposition='outside',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.9,
        cliponaxis=False,
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_type='log' if log_scale else 'linear'
    )

    if not log_scale:
        fig.update_yaxes(tickformat=',')

    return fig

def create_comparison_table(results_data):
    """Creates a comprehensive comparison table of all metrics."""
    if not results_data:
        return pd.DataFrame()
    
    table_data = []
    for result in results_data:
        table_data.append({
            'Algorithm': result.get('algorithm', 'Unknown'),
            'Success': 'Yes' if result.get('success', False) else 'No',
            'Time (ms)': f"{result.get('execution_time_ms', 0):.3f}",
            'Memory (KB)': f"{result.get('memory_kb', 0):.2f}",
            'Branches': f"{result.get('branches', 0):,}",
            'Nodes Expanded': f"{result.get('nodes_expanded', 0):,}",
            'Iterations': f"{result.get('iterations', 0):,}"
        })
    
    return pd.DataFrame(table_data)

def render_comparison_results(results_data):
    """Renders comparison visualizations for all algorithms."""
    if not results_data:
        st.info("No results to display. Please solve a puzzle first.")
        return
    
    st.markdown("### 📊 Algorithm Performance Comparison")
    
    comparison_df = create_comparison_table(results_data)
    if not comparison_df.empty:
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        fig_time = create_comparison_bar_chart(
            results_data,
            'execution_time_ms',
            'Execution Time Comparison',
            'Time (milliseconds)',
            log_scale=False
        )
        if fig_time:
            st.plotly_chart(fig_time, use_container_width=True)
        
        fig_memory = create_comparison_bar_chart(
            results_data,
            'memory_kb',
            'Memory Usage Comparison',
            'Memory (KB)',
            log_scale=True
        )
        if fig_memory:
            st.plotly_chart(fig_memory, use_container_width=True)
    
    with col2:
        fig_nodes = create_comparison_bar_chart(
            results_data,
            'nodes_expanded',
            'Nodes Expanded Comparison',
            'Nodes Expanded',
            log_scale=True
        )
        if fig_nodes:
            st.plotly_chart(fig_nodes, use_container_width=True)
        
        fig_branches = create_comparison_bar_chart(
            results_data,
            'branches',
            'Branching Factor Comparison',
            'Branches',
            log_scale=True
        )
        if fig_branches:
            st.plotly_chart(fig_branches, use_container_width=True)
    
    st.markdown("---")
    fig_iterations = create_comparison_bar_chart(
        results_data,
        'iterations',
        'Iterations Comparison (Stochastic Algorithms)',
        'Iterations',
        log_scale=False
    )
    if fig_iterations:
        st.plotly_chart(fig_iterations, use_container_width=True)

# --- MAIN STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(
        page_title="Solverine: AI Sudoku Benchmarking", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    if 'solved_grid_str' not in st.session_state:
        st.session_state.solved_grid_str = ""
    if 'all_results' not in st.session_state:
        st.session_state.all_results = []
    if 'solved_grid_matrix' not in st.session_state:
        st.session_state.solved_grid_matrix = None

    st.title("Solverine: AI Sudoku Solver & Benchmarking Platform")

    # --- Tab Navigation ---
    tab1, tab2 = st.tabs(["🎮 User Solver Interface", "📊 Enhanced Analytics Dashboard"])

    with tab1:
        st.markdown("## Sudoku Solver - Multi-Algorithm Benchmarking")
        st.markdown("Enter your Sudoku puzzle below and we'll solve it using all available algorithms!")
        
        col_input, col_preview = st.columns([2, 1])
        
        with col_input:
            st.markdown("### Puzzle Input")
            grid_size = st.selectbox(
                "Select Grid Size",
                options=["4x4", "5x5", "9x9", "10x10"],
                index=2,
                key='grid_size',
                help="Choose the size of your Sudoku grid"
            )
            
            puzzle_input = st.text_area(
                "Unsolved Sudoku String",
                key=PUZZLE_INPUT_KEY,
                placeholder="Example 9x9: 530070000600195000098000060800060003400803001700020006060000280000419005000080079",
                height=120,
                help="Enter the puzzle as a string. Use 0 or . for empty cells."
            )
            
            solve_button = st.button(
                "🚀 Solve with All Algorithms", 
                type="primary", 
                use_container_width=True,
                help=f"Will solve using all {len(ALGORITHMS)} algorithms"
            )
            
            if solve_button:
                puzzle_str = st.session_state.get(PUZZLE_INPUT_KEY, '')
                grid_size_str = st.session_state.get('grid_size', '9x9')
                
                user_id = st.session_state.get('user_id', f"anonymous_{random.randint(1000, 9999)}")
                st.session_state['user_id'] = user_id
                
                if not puzzle_str:
                    st.error("Please enter a Sudoku string.")
                else:
                    try:
                        n, _, _ = get_grid_dimensions(grid_size_str)
                        if len(puzzle_str.replace('.', '0')) != n * n:
                            st.error(f"Input string length ({len(puzzle_str)}) must match expected size ({n*n}) for {grid_size_str}.")
                        else:
                            with st.spinner(f"🔄 Solving puzzle with {len(ALGORITHMS)} algorithms..."):
                                all_results = solve_all_algorithms(puzzle_str, grid_size_str, user_id)
                            
                            st.session_state.all_results = all_results
                            st.session_state.grid_size_str = grid_size_str
                            
                            successful_results = [r for r in all_results if r.get("success")]
                            if successful_results:
                                st.session_state.solved_grid_str = successful_results[0].get("solved", "")
                                st.session_state.solved_grid_matrix = successful_results[0].get("solved_matrix")
                            elif all_results:
                                st.session_state.solved_grid_str = all_results[0].get("solved", puzzle_str)
                                st.session_state.solved_grid_matrix = all_results[0].get("solved_matrix")
                            
                            successful_count = len(successful_results)
                            if successful_count > 0:
                                st.success(f"✅ Completed! {successful_count}/{len(ALGORITHMS)} algorithms solved successfully.")
                            else:
                                st.warning(f"⚠️ Completed! None of the algorithms found a solution.")
                    except ValueError as e:
                        st.error(str(e))
        
        with col_preview:
            st.markdown("### Puzzle Preview")
            n = 9
            grid_size_str = st.session_state.get('grid_size', '9x9')
            try:
                n, _, _ = get_grid_dimensions(grid_size_str)
            except:
                pass
            
            preview_container = st.container()
            puzzle_str = st.session_state.get(PUZZLE_INPUT_KEY, '')
            if puzzle_str:
                draw_sudoku_grid(puzzle_str, n, preview_container, grid_size_str)
            else:
                preview_container.info("Enter a puzzle to see preview")
        
        st.markdown("---")
        
        if st.session_state.get('all_results'):
            st.markdown("## Results & Comparison")
            
            if st.session_state.solved_grid_str:
                st.markdown("### Solved Puzzle")
                solved_container = st.container()
                grid_size_str = st.session_state.get('grid_size_str', st.session_state.get('grid_size', '9x9'))
                draw_sudoku_grid(st.session_state.solved_grid_str, n, solved_container, grid_size_str)
                st.markdown("---")
            
            render_comparison_results(st.session_state.all_results)
        else:
            st.info("Enter a Sudoku puzzle and click 'Solve with All Algorithms' to see results!")


    with tab2:
        all_results_df = load_results()
        render_enhanced_analytics_dashboard(all_results_df)

if __name__ == "__main__":
    main()