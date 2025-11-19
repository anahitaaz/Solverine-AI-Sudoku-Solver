import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- ENHANCED ANALYTICS FUNCTIONS ---

def calculate_difficulty_level(grid_size, branches, nodes_expanded):
    """Classify puzzle difficulty based on computational metrics. Only for 4x4, 5x5, and 9x9."""
    if grid_size == "4x4":
        if branches < 10 and nodes_expanded < 50:
            return "Easy"
        elif branches < 50 and nodes_expanded < 200:
            return "Medium"
        else:
            return "Hard"
    elif grid_size == "5x5":
        if branches < 50 and nodes_expanded < 300:
            return "Easy"
        elif branches < 300 and nodes_expanded < 1500:
            return "Medium"
        else:
            return "Hard"
    elif grid_size == "9x9":
        if branches < 100 and nodes_expanded < 500:
            return "Easy"
        elif branches < 500 and nodes_expanded < 2000:
            return "Medium"
        else:
            return "Hard"
    else:
        return "Unknown"

def create_performance_summary_table(df, grid_size=None):
    """Creates Algorithm Performance Summary Table per grid size."""
    if df.empty:
        return pd.DataFrame()
    
    if grid_size and grid_size != 'All':
        df_filtered = df[df['grid_size'] == grid_size]
    else:
        df_filtered = df
    
    summary = df_filtered.groupby('algorithm').agg({
        'execution_time_ms': ['mean', 'std', 'min', 'max'],
        'memory_kb': ['mean', 'std', 'min', 'max'],
        'branches': ['mean', 'std'],
        'nodes_expanded': ['mean', 'std'],
        'iterations': ['mean', 'std'],
        'success': 'sum'
    }).round(2)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary['total_runs'] = df_filtered.groupby('algorithm').size()
    summary['success_rate_%'] = (summary['success_sum'] / summary['total_runs'] * 100).round(1)
    
    summary = summary.reset_index()
    return summary

def create_combined_statistics_table(df):
    """Creates Combined Parameter Statistics Table across all algorithms."""
    if df.empty:
        return pd.DataFrame()
    
    stats = df.groupby(['grid_size', 'algorithm']).agg({
        'execution_time_ms': 'mean',
        'memory_kb': 'mean',
        'branches': 'mean',
        'nodes_expanded': 'mean',
        'iterations': 'mean'
    }).round(2)
    
    return stats.reset_index()

def create_difficulty_performance_table(df):
    """Creates Difficulty-Level Performance Table."""
    if df.empty:
        return pd.DataFrame()
    
    # Add difficulty level to dataframe
    df_copy = df.copy()
    df_copy['difficulty'] = df_copy.apply(
        lambda row: calculate_difficulty_level(
            row['grid_size'], 
            row['branches'], 
            row['nodes_expanded']
        ), axis=1
    )
    
    difficulty_stats = df_copy.groupby(['difficulty', 'algorithm']).agg({
        'execution_time_ms': 'mean',
        'memory_kb': 'mean',
        'branches': 'mean',
        'nodes_expanded': 'mean',
        'success': 'sum'
    }).round(2)
    
    difficulty_stats['total_puzzles'] = df_copy.groupby(['difficulty', 'algorithm']).size()
    difficulty_stats['success_rate_%'] = (
        difficulty_stats['success'] / difficulty_stats['total_puzzles'] * 100
    ).round(1)
    
    return difficulty_stats.reset_index()

def create_algorithm_ranking_table(df):
    """Creates Algorithm Ranking Table based on multiple criteria."""
    if df.empty:
        return pd.DataFrame()
    
    # Calculate rankings for each metric (lower is better for time/memory/branches/nodes)
    rankings = df.groupby('algorithm').agg({
        'execution_time_ms': 'mean',
        'memory_kb': 'mean',
        'branches': 'mean',
        'nodes_expanded': 'mean',
        'success': 'sum'
    }).round(2)
    
    rankings['total_runs'] = df.groupby('algorithm').size()
    rankings['success_rate_%'] = (rankings['success'] / rankings['total_runs'] * 100).round(1)
    
    # Rank each metric (1 = best)
    rankings['time_rank'] = rankings['execution_time_ms'].rank()
    rankings['memory_rank'] = rankings['memory_kb'].rank()
    rankings['branches_rank'] = rankings['branches'].rank()
    rankings['nodes_rank'] = rankings['nodes_expanded'].rank()
    rankings['success_rank'] = rankings['success_rate_%'].rank(ascending=False)
    
    # Calculate composite score (lower is better)
    rankings['composite_score'] = (
        rankings['time_rank'] + 
        rankings['memory_rank'] + 
        rankings['branches_rank'] + 
        rankings['nodes_rank'] + 
        rankings['success_rank']
    ) / 5
    
    rankings['overall_rank'] = rankings['composite_score'].rank()
    
    return rankings.reset_index().sort_values('overall_rank')

# --- ENHANCED VISUALIZATION FUNCTIONS ---

def create_execution_time_comparison(df, grid_size=None):
    """Execution Time Bar Charts with error bars."""
    if df.empty:
        return None
    
    df_filtered = df if grid_size == 'All' else df[df['grid_size'] == grid_size]
    
    fig = px.bar(
        df_filtered,
        x='algorithm',
        y='execution_time_ms',
        color='algorithm',
        title=f'Execution Time Comparison - {grid_size if grid_size else "All Sizes"}',
        labels={'execution_time_ms': 'Time (ms)', 'algorithm': 'Algorithm'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_memory_usage_comparison(df, grid_size=None):
    """Memory Usage Bar Charts."""
    if df.empty:
        return None
    
    df_filtered = df if grid_size == 'All' else df[df['grid_size'] == grid_size]
    
    fig = px.bar(
        df_filtered,
        x='algorithm',
        y='memory_kb',
        color='algorithm',
        title=f'Memory Usage Comparison - {grid_size if grid_size else "All Sizes"}',
        labels={'memory_kb': 'Memory (KB)', 'algorithm': 'Algorithm'},
        color_discrete_sequence=px.colors.qualitative.Pastel,
        log_y=True
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_nodes_expanded_line_chart(df):
    """Nodes Expanded Line Chart across grid sizes."""
    if df.empty:
        return None
    
    df_grouped = df.groupby(['grid_size', 'algorithm'])['nodes_expanded'].mean().reset_index()
    
    fig = px.line(
        df_grouped,
        x='grid_size',
        y='nodes_expanded',
        color='algorithm',
        markers=True,
        title='Nodes Expanded vs Grid Size',
        labels={'nodes_expanded': 'Avg Nodes Expanded', 'grid_size': 'Grid Size'},
        log_y=True
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        height=450
    )
    
    return fig

def create_branches_line_chart(df):
    """Branches/Recursion Depth Line Chart."""
    if df.empty:
        return None
    
    df_grouped = df.groupby(['grid_size', 'algorithm'])['branches'].mean().reset_index()
    
    fig = px.line(
        df_grouped,
        x='grid_size',
        y='branches',
        color='algorithm',
        markers=True,
        title='Branching Factor vs Grid Size',
        labels={'branches': 'Avg Branches', 'grid_size': 'Grid Size'},
        log_y=True
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        height=450
    )
    
    return fig

def create_iterations_chart(df):
    """Iterations chart for stochastic algorithms (SA, GA)."""
    if df.empty:
        return None
    
    # Filter for algorithms that use iterations
    df_stochastic = df[df['iterations'] > 0]
    
    if df_stochastic.empty:
        return None
    
    df_grouped = df_stochastic.groupby(['grid_size', 'algorithm'])['iterations'].mean().reset_index()
    
    fig = px.bar(
        df_grouped,
        x='grid_size',
        y='iterations',
        color='algorithm',
        barmode='group',
        title='Average Iterations (Stochastic Algorithms)',
        labels={'iterations': 'Avg Iterations', 'grid_size': 'Grid Size'}
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        height=400
    )
    
    return fig

def create_time_heatmap(df):
    """Time Heatmap (Algorithm × Grid Size)."""
    if df.empty:
        return None
    
    # Pivot table for heatmap
    pivot_data = df.pivot_table(
        values='execution_time_ms',
        index='algorithm',
        columns='grid_size',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='YlOrRd',
        text=np.round(pivot_data.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Time (ms)")
    ))
    
    fig.update_layout(
        title='Execution Time Heatmap (Algorithm × Grid Size)',
        xaxis_title='Grid Size',
        yaxis_title='Algorithm',
        height=500
    )
    
    return fig

def create_radar_chart(df):
    """Radar Chart for overall algorithm comparison."""
    if df.empty:
        return None
    
    # Normalize metrics to 0-100 scale for fair comparison
    metrics = df.groupby('algorithm').agg({
        'execution_time_ms': 'mean',
        'memory_kb': 'mean',
        'branches': 'mean',
        'nodes_expanded': 'mean',
        'success': 'mean'
    })
    
    # Invert time, memory, branches, nodes (lower is better)
    # Scale success to 0-100
    normalized = pd.DataFrame()
    normalized['Speed'] = 100 - (metrics['execution_time_ms'] / metrics['execution_time_ms'].max() * 100)
    normalized['Memory Efficiency'] = 100 - (metrics['memory_kb'] / metrics['memory_kb'].max() * 100)
    normalized['Low Branching'] = 100 - (metrics['branches'] / metrics['branches'].max() * 100)
    normalized['Node Efficiency'] = 100 - (metrics['nodes_expanded'] / metrics['nodes_expanded'].max() * 100)
    normalized['Success Rate'] = metrics['success'] * 100
    
    fig = go.Figure()
    
    categories = list(normalized.columns)
    
    for algorithm in normalized.index:
        values = normalized.loc[algorithm].tolist()
        values.append(values[0])  # Close the radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=algorithm
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title='Algorithm Performance Radar Chart',
        height=600
    )
    
    return fig

def create_success_rate_chart(df):
    """Success rate comparison across algorithms and grid sizes."""
    if df.empty:
        return None
    
    success_data = df.groupby(['algorithm', 'grid_size']).agg({
        'success': ['sum', 'count']
    }).reset_index()
    
    success_data.columns = ['algorithm', 'grid_size', 'successes', 'total']
    success_data['success_rate'] = (success_data['successes'] / success_data['total'] * 100).round(1)
    
    fig = px.bar(
        success_data,
        x='algorithm',
        y='success_rate',
        color='grid_size',
        barmode='group',
        title='Success Rate by Algorithm and Grid Size',
        labels={'success_rate': 'Success Rate (%)', 'algorithm': 'Algorithm'},
        text='success_rate'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        plot_bgcolor='white',
        height=450,
        xaxis_tickangle=-45
    )
    
    return fig

# --- ENHANCED ANALYTICS DASHBOARD ---

def render_enhanced_analytics_dashboard(df):
    """Renders comprehensive analytics dashboard with all tables and graphs."""
    st.header("🎯 Enhanced Analytics Dashboard")
    
    if df.empty:
        st.info("No benchmarking results found yet. Solve puzzles to populate the analytics!")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Benchmarks", len(df))
    with col2:
        st.metric("Algorithms Tested", df['algorithm'].nunique())
    with col3:
        st.metric("Grid Sizes", df['grid_size'].nunique())
    with col4:
        success_rate = (df['success'].sum() / len(df) * 100)
        st.metric("Overall Success Rate", f"{success_rate:.1f}%")
    
    st.markdown("---")
    
    # Filter controls
    st.subheader("🔍 Filters")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        grid_sizes = sorted(df['grid_size'].unique(), key=lambda x: int(x.split('x')[0]))
        selected_size = st.selectbox("Grid Size", ['All'] + grid_sizes, index=0)
    with col_f2:
        algorithms = sorted(df['algorithm'].unique())
        selected_algorithms = st.multiselect("Algorithms", algorithms, default=algorithms)
    
    # Apply filters
    df_filtered = df.copy()
    if selected_size != 'All':
        df_filtered = df_filtered[df_filtered['grid_size'] == selected_size]
    if selected_algorithms:
        df_filtered = df_filtered[df_filtered['algorithm'].isin(selected_algorithms)]
    
    if df_filtered.empty:
        st.warning("No data matches the selected filters.")
        return
    
    st.markdown("---")
    
    # TABLES SECTION
    st.header("📊 Performance Tables")
    
    tab_t1, tab_t2, tab_t3, tab_t4 = st.tabs([
        "Algorithm Summary", 
        "Combined Statistics", 
        "Difficulty Analysis", 
        "Algorithm Rankings"
    ])
    
    with tab_t1:
        st.subheader("Algorithm Performance Summary")
        summary_table = create_performance_summary_table(df_filtered, selected_size)
        if not summary_table.empty:
            st.dataframe(summary_table, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for this view.")
    
    with tab_t2:
        st.subheader("Combined Parameter Statistics")
        combined_table = create_combined_statistics_table(df_filtered)
        if not combined_table.empty:
            st.dataframe(combined_table, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for this view.")
    
    with tab_t3:
        st.subheader("Difficulty-Level Performance")
        difficulty_table = create_difficulty_performance_table(df_filtered)
        if not difficulty_table.empty:
            st.dataframe(difficulty_table, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for this view.")
    
    with tab_t4:
        st.subheader("Algorithm Rankings (Composite Score)")
        ranking_table = create_algorithm_ranking_table(df_filtered)
        if not ranking_table.empty:
            st.dataframe(
                ranking_table[['algorithm', 'execution_time_ms', 'memory_kb', 
                              'success_rate_%', 'composite_score', 'overall_rank']], 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info("No data available for this view.")
    
    st.markdown("---")
    
    # GRAPHS SECTION
    st.header("📈 Performance Visualizations")
    
    # Row 1: Time and Memory
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_time = create_execution_time_comparison(df_filtered, selected_size)
        if fig_time:
            st.plotly_chart(fig_time, use_container_width=True)
    
    with col_g2:
        fig_memory = create_memory_usage_comparison(df_filtered, selected_size)
        if fig_memory:
            st.plotly_chart(fig_memory, use_container_width=True)
    
    # Row 2: Nodes and Branches
    col_g3, col_g4 = st.columns(2)
    with col_g3:
        fig_nodes = create_nodes_expanded_line_chart(df_filtered)
        if fig_nodes:
            st.plotly_chart(fig_nodes, use_container_width=True)
    
    with col_g4:
        fig_branches = create_branches_line_chart(df_filtered)
        if fig_branches:
            st.plotly_chart(fig_branches, use_container_width=True)
    
    # Row 3: Iterations and Success Rate
    col_g5, col_g6 = st.columns(2)
    with col_g5:
        fig_iterations = create_iterations_chart(df_filtered)
        if fig_iterations:
            st.plotly_chart(fig_iterations, use_container_width=True)
        else:
            st.info("No iteration data available (stochastic algorithms only)")
    
    with col_g6:
        fig_success = create_success_rate_chart(df_filtered)
        if fig_success:
            st.plotly_chart(fig_success, use_container_width=True)
    
    # Row 4: Heatmap
    st.subheader("🔥 Execution Time Heatmap")
    fig_heatmap = create_time_heatmap(df_filtered)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Row 5: Radar Chart
    st.subheader("🎯 Overall Performance Radar")
    fig_radar = create_radar_chart(df_filtered)
    if fig_radar:
        st.plotly_chart(fig_radar, use_container_width=True)