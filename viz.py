import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple
from scipy import stats
import numpy as np

def plot_network_metrics(df: pd.DataFrame, tvl_goals: List[Tuple[float, float]]) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "TVL Over Time (csUSDL)", "Active vs Inactive Nodes",
            "Treasury Balance", "Haircut Evolution (%)",
            "Daily Rewards", "Growth Rate"
        ]
    )
    
    # TVL plot
    tvl_trace = go.Scatter(
        x=df["Time"],
        y=df["TVL"] / 1e6,  # Convert to millions
        name="TVL (M)",
        mode='lines+markers'
    )
    
    fig.add_trace(tvl_trace, row=1, col=1)
    
    # Update y-axis to ensure it starts from 0 or lower
    min_tvl = (df["TVL"] / 1e6).min()  # Convert to millions
    max_tvl = (df["TVL"] / 1e6).max()  # Convert to millions
    y_range = [
        min_tvl * 0.9 if min_tvl > 0 else 0,
        max_tvl * 1.2  # Increase upper bound by 20% for better visibility
    ]
    
    fig.update_yaxes(
        range=y_range,
        title_text="TVL (M)",
        row=1, col=1,
        tickformat=".1f"
    )
    
    # Add TVL goals
    for goal, haircut_start in tvl_goals:
        goal_in_m = goal / 1e6  # Convert goal to millions
        goal_reached_time = next(
            (t for t, v in zip(df["Time"], df["TVL"]) if v >= goal), 
            None
        )
        
        fig.add_hline(
            y=goal_in_m,  # Use goal in millions
            line=dict(
                color="green" if goal_reached_time else "red",
                dash="dash"
            ),
            row=1, col=1
        )
        
        if goal_reached_time:
            fig.add_annotation(
                x=goal_reached_time,
                y=goal_in_m,  # Use goal in millions
                text=f"Goal {goal_in_m:.1f}M reached",
                showarrow=True,
                arrowhead=1,
                row=1, col=1
            )
            fig.add_vline(
                x=goal_reached_time,
                line=dict(color="green", dash="dash"),
                row=1, col=1
            )
        else:
            fig.add_annotation(
                x=df["Time"].iloc[-1],
                y=goal_in_m,  # Use goal in millions
                text=f"Goal: {goal_in_m:.1f}M",
                showarrow=False,
                xanchor="left",
                row=1, col=1
            )
    
    # Node counts
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Active Nodes"], name="Active Nodes"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Inactive Nodes"], name="Inactive Nodes"),
        row=1, col=2
    )
    
    # Treasury
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Treasury"], name="Treasury"),
        row=2, col=1
    )
    
    # Haircut
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Haircut"].mul(100), name="Haircut %"),
        row=2, col=2
    )
    
    # Rewards
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Rewards"], name="Daily Rewards"),
        row=3, col=1
    )
    
    # Growth rate - Calculate daily growth in nodes
    if len(df) > 1:
        df["Node Growth"] = df["Active Nodes"].diff().fillna(0)
        # Smooth the growth rate with a rolling window
        df["Smoothed Growth"] = df["Node Growth"].rolling(window=7, min_periods=1).mean()
        
        fig.add_trace(
            go.Scatter(x=df["Time"], y=df["Node Growth"], name="Daily New Nodes", 
                      line=dict(color='lightblue'), opacity=0.4),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=df["Time"], y=df["Smoothed Growth"], name="7-Day Average Growth",
                      line=dict(color='blue')),
            row=3, col=2
        )
    
    fig.update_layout(
        height=900,
        showlegend=True,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    # Update axes titles
    fig.update_xaxes(title_text="Days", row=1, col=1)
    fig.update_xaxes(title_text="Days", row=1, col=2)
    fig.update_xaxes(title_text="Days", row=2, col=1)
    fig.update_xaxes(title_text="Days", row=2, col=2)
    fig.update_xaxes(title_text="Days", row=3, col=1)
    fig.update_xaxes(title_text="Days", row=3, col=2)
    
    fig.update_yaxes(title_text="TVL (M)", row=1, col=1)
    fig.update_yaxes(title_text="Node Count", row=1, col=2)
    fig.update_yaxes(title_text="Treasury", row=2, col=1)
    fig.update_yaxes(title_text="Haircut %", row=2, col=2)
    fig.update_yaxes(title_text="Daily Rewards", row=3, col=1)
    fig.update_yaxes(title_text="Nodes/Day", row=3, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        margin=dict(t=30, l=50, r=50, b=50)
    )

    fig.update_xaxes(title_text="Days", row=1, col=1)
    fig.update_xaxes(title_text="Treasury Balance", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    return fig

def plot_milestone_metrics(result, milestone_data: dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Active Nodes Distribution",
            "Network Growth Rate",
            "Node Activity",
            "Treasury Distribution"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
    )

    nodes = milestone_data['nodes_at_milestone'][result.target]

    if len(nodes) > 1:
        nodes_kde = stats.gaussian_kde(nodes)
        nodes_points = np.linspace(min(nodes), max(nodes), 100)
        nodes_density = nodes_kde(nodes_points)

        fig.add_trace(
            go.Scatter(
                x=nodes_points,
                y=nodes_density,
                mode='lines',
                fill='tozeroy',
                showlegend=False,
                line=dict(color='blue')
            ),
            row=1, col=1
        )

    times = milestone_data['time_at_milestone'][result.target]
    nodes = milestone_data['nodes_at_milestone'][result.target]
    if times and nodes:
        df = pd.DataFrame({'time': times, 'nodes': nodes})
        df['growth_rate'] = df['nodes'] / df['time']
        df['time_rounded'] = df['time'].round()
        
        grouped = df.groupby('time_rounded').agg({
            'growth_rate': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['time', 'mean', 'std']
        
        sorted_idx = grouped['time'].argsort()
        grouped = grouped.iloc[sorted_idx]

        fig.add_trace(
            go.Scatter(
                x=grouped['time'],
                y=grouped['mean'],
                mode='lines',
                name='Growth Rate',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=1, col=2
        )

        # Plot standard deviation as error bars
        fig.add_trace(
            go.Scatter(
                x=grouped['time'],
                y=grouped['mean'],
                error_y=dict(
                    type='data',
                    array=grouped['std'],
                    color='blue',
                    thickness=1,
                    width=0
                ),
                mode='lines',
                line=dict(color='blue'),
                name='Growth Rate ± σ',
                showlegend=False
            ),
            row=1, col=2
        )

    if times and nodes:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=nodes,
                mode='markers',
                marker=dict(color='orange', size=6),
                name='Node Activity',
                showlegend=False
            ),
            row=2, col=1
        )

    treasuries = milestone_data['treasury_at_milestone'][result.target]
    if treasuries and len(treasuries) > 1:
        try:
            treasuries_kde = stats.gaussian_kde(treasuries)
            treasuries_points = np.linspace(min(treasuries), max(treasuries), 100)
            treasuries_density = treasuries_kde(treasuries_points)

            fig.add_trace(
                go.Scatter(
                    x=treasuries_points,
                    y=treasuries_density,
                    mode='lines',
                    fill='tozeroy',
                    showlegend=False,
                    line=dict(color='green')
                ),
                row=2, col=2
            )
        except:
            # Fallback if KDE fails
            fig.add_trace(
                go.Histogram(
                    x=treasuries,
                    histnorm='probability density',
                    marker_color='green',
                    showlegend=False
                ),
                row=2, col=2
            )

    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    fig.update_xaxes(title_text="Number of Nodes", row=1, col=1)
    fig.update_xaxes(title_text="Time (days)", row=1, col=2)
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text="Treasury Amount", row=2, col=2)
    
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Growth Rate (nodes/day)", row=1, col=2)
    fig.update_yaxes(title_text="Active Nodes", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=2)

    return fig

def plot_optimization_results(result, milestone_data: dict) -> go.Figure:
    # Create figure with 4 subplots (2 rows, 2 columns)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Time to Target Distribution",
            "Node Growth Trajectories",
            "Haircut Collection Distribution",
            "Growth Phase Impact"
        ]
    )

    # 1. Time to Target Distribution
    times = milestone_data['time_at_milestone'][result.target_tvl]
    if times and len(times) > 1:
        try:
            kde = stats.gaussian_kde(times)
            x = np.linspace(min(times), max(times), 100)
            y = kde(x)
            fig.add_trace(
                go.Scatter(x=x, y=y, fill='tozeroy', name="Time Distribution"),
                row=1, col=1
            )
        except np.linalg.LinAlgError:
            # Fallback to histogram if KDE fails
            fig.add_trace(
                go.Histogram(x=times, histnorm='probability density', name="Time Distribution"),
                row=1, col=1
            )
        
        fig.add_vline(
            x=np.median(times),
            line=dict(color="red", dash="dash"),
            annotation=dict(text=f"Median: {np.median(times):.1f} days"),
            row=1, col=1
        )

    # 2. Add node growth trajectories plot (top right)
    if 'node_trajectories' in milestone_data and milestone_data['node_trajectories']:
        for ii, trajectory in enumerate(milestone_data['node_trajectories']):
            if ii < 100:  # Limit to 100 trajectories for performance
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(trajectory))),
                        y=trajectory,
                        mode='lines',
                        line=dict(color='blue', width=1),
                        opacity=0.1,
                        showlegend=False,
                        hovertemplate="Day: %{x}<br>Nodes: %{y}<extra></extra>"
                    ),
                    row=1, col=2
                )
        
        # Add median trajectory
        trajectories_array = np.array([
            t[:min(len(traj) for traj in milestone_data['node_trajectories'])] 
            for t in milestone_data['node_trajectories']
        ])
        median_trajectory = np.median(trajectories_array, axis=0)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(median_trajectory))),
                y=median_trajectory,
                mode='lines',
                line=dict(color='red', width=2),
                name='Median Growth',
                hovertemplate="Day: %{x}<br>Nodes: %{y:.0f}<extra></extra>"
            ),
            row=1, col=2
        )

    # 3. Haircut Distribution
    haircuts = milestone_data['haircuts_collected_at_milestone'][result.target_tvl]
    if haircuts and len(haircuts) > 1:
        try:
            kde = stats.gaussian_kde(haircuts)
            x = np.linspace(min(haircuts), max(haircuts), 100)
            y = kde(x)
            fig.add_trace(
                go.Scatter(x=x, y=y, fill='tozeroy', name="Haircut Distribution"),
                row=2, col=1
            )
        except np.linalg.LinAlgError:
            fig.add_trace(
                go.Histogram(x=haircuts, histnorm='probability density', name="Haircut Distribution"),
                row=2, col=1
            )
        
        median_haircut = np.median(haircuts)
        fig.add_vline(
            x=median_haircut,
            line=dict(color="red", dash="dash"),
            annotation=dict(text=f"Median: {median_haircut:,.0f}"),
            row=2, col=1
        )

    # 4. Growth Phase Impact Chart
    # Create a visual representation of the growth curve and current position
    if hasattr(result, 'growth_phase'):
        phases = ["Early Growth", "Accelerating Growth", "Peak Growth", "Decelerating Growth", "Maturity"]
        phase_positions = [0, 25, 50, 75, 100]
        phase_colors = ['lightblue', 'blue', 'green', 'orange', 'red']
        
        # Map the growth phase to a position
        current_phase_idx = phases.index(result.growth_phase)
        current_position = phase_positions[current_phase_idx]
        
        # Create growth phase chart
        fig.add_trace(
            go.Scatter(
                x=phase_positions,
                y=[0, 25, 100, 25, 10],  # Simulate a growth curve shape
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add points for each phase
        for i, (phase, pos, color) in enumerate(zip(phases, phase_positions, phase_colors)):
            fig.add_trace(
                go.Scatter(
                    x=[pos],
                    y=[0 if i==0 else 25 if i==1 else 100 if i==2 else 25 if i==3 else 10],
                    mode='markers',
                    marker=dict(
                        color=color, 
                        size=12 if phase == result.growth_phase else 8,
                        line=dict(
                            color='black', 
                            width=2 if phase == result.growth_phase else 0
                        )
                    ),
                    name=phase,
                    showlegend=True
                ),
                row=2, col=2
            )
        
        # Add TVL strategy recommendation based on phase
        tvl_increase = f"{(result.target_tvl/(result.target_tvl/1.5) - 1)*100:.1f}%"
        fig.add_annotation(
            x=50, y=110,
            text=f"Current Phase: {result.growth_phase}",
            showarrow=False,
            font=dict(size=14, color="black"),
            row=2, col=2
        )
        fig.add_annotation(
            x=50, y=50,
            text=f"Optimal TVL Target: +{tvl_increase}",
            showarrow=False,
            font=dict(size=12),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Optimization Results (Target TVL: ${float(result.target_tvl/1e6):.1f}M, Haircut: {result.haircut*100:.1f}%)"
    )

    # Update axes titles
    fig.update_xaxes(title_text="Days to Target", row=1, col=1)
    fig.update_xaxes(title_text="Days", row=1, col=2)
    fig.update_xaxes(title_text="Total Haircut Collected", row=2, col=1)
    fig.update_xaxes(title_text="Network Growth Phase", range=[-5, 105], row=2, col=2)

    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Number of Nodes", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_yaxes(title_text="Growth Rate", range=[-5, 120], row=2, col=2)

    # Add grid lines for node trajectories plot
    fig.update_xaxes(
        gridcolor='lightgrey',
        showgrid=True,
        row=1, col=2
    )
    fig.update_yaxes(
        gridcolor='lightgrey',
        showgrid=True,
        row=1, col=2
    )
    
    # Hide ticks and gridlines in the growth phase visualization
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        row=2, col=2
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        row=2, col=2
    )

    return fig

def display_monte_carlo_metrics(result, n_simulations: int) -> str:
    if result.times_reached == 0:
        return f"Success Rate: {(result.times_reached/n_simulations)*100:.1f}%"
    
    metrics = [
        f"Success Rate: {(result.times_reached/n_simulations)*100:.1f}%",
        f"Active Nodes: {result.mean_active_nodes:,.0f}",
        f"Avg Lock Period: {result.mean_lock_period:.1f}d" if result.mean_lock_period else ""
    ]
    
    return " | ".join(filter(None, metrics))

def plot_withdrawal_distributions(result, n_simulations: int) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Total Haircut Distribution",
            "Total Withdrawals Distribution"
        ]
    )

    if result.times_reached > 0 and result.mean_treasury is not None:
        haircut_points = np.linspace(0, result.mean_treasury * 2, 100)
        haircut_kde = stats.gaussian_kde([0, result.mean_treasury, result.mean_treasury * 2])
        haircut_density = haircut_kde(haircut_points)

        fig.add_trace(
            go.Scatter(
                x=haircut_points,
                y=haircut_density,
                mode='lines',
                fill='tozeroy',
                name="Haircut Distribution",
                showlegend=False
            ),
            row=1, col=1
        )

        fig.add_vline(
            x=result.median_treasury,
            line_dash="dash",
            line_color="green",
            annotation=dict(
                text=f"Median: ${result.median_treasury:,.0f}",
                y=0.8,
                yanchor='bottom'
            ),
            row=1, col=1
        )

        # Calculate total withdrawals with fallback for missing haircut_rate
        haircut_rate = getattr(result, 'haircut_rate', 0.5)  # Default to 0.5 if not available
        if haircut_rate is not None and haircut_rate != 1:  # Avoid division by zero
            total_withdrawals = result.mean_treasury / (1 - haircut_rate)
        else:
            total_withdrawals = result.mean_treasury * 2  # Fallback calculation

        withdrawal_points = np.linspace(0, total_withdrawals * 2, 100)
        withdrawal_kde = stats.gaussian_kde([0, total_withdrawals, total_withdrawals * 2])
        withdrawal_density = withdrawal_kde(withdrawal_points)

        fig.add_trace(
            go.Scatter(
                x=withdrawal_points,
                y=withdrawal_density,
                mode='lines',
                fill='tozeroy',
                name="Withdrawals Distribution",
                showlegend=False
            ),
            row=1, col=2
        )

        fig.add_vline(
            x=total_withdrawals,
            line_dash="dash",
            line_color="green",
            annotation=dict(
                text=f"Median: ${total_withdrawals:,.0f}",
                y=0.8,
                yanchor='bottom'
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(t=30, l=50, r=50, b=50)
    )

    fig.update_xaxes(title_text="Total Haircut", row=1, col=1)
    fig.update_xaxes(title_text="Total Withdrawals", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    return fig

def plot_withdrawal_metrics(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Daily Reward Withdrawals",
            "Cumulative Reward Withdrawals",
            "Average Lock Period",
            "Principal vs Reward Withdrawals"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "domain"}]  # "domain" type for pie chart
        ]
    )
    
    # Daily reward withdrawals
    fig.add_trace(
        go.Bar(
            x=df["Time"],
            y=df["Net Claimed Rewards"],
            name="Net Claimed",
            marker_color='green'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=df["Time"],
            y=df["Haircut Collected"],
            name="Haircut",
            marker_color='red'
        ),
        row=1, col=1
    )
    
    # Cumulative withdrawals
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Net Claimed Rewards"].cumsum(),
            name="Net Claimed (Cum.)",
            line=dict(color='green')
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Haircut Collected"].cumsum(),
            name="Haircut (Cum.)",
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    # Average Lock Period over time
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Average Lock Period"],
            name="Avg Lock Period",
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # Principal vs Reward withdrawals pie chart
    total_rewards = df["Net Claimed Rewards"].sum()
    total_haircut = df["Haircut Collected"].sum()
    total_principal = df["Principal Withdrawn"].sum()
    
    fig.add_trace(
        go.Pie(
            values=[total_rewards, total_haircut, total_principal],
            labels=["Net Rewards", "Haircut", "Principal"],
            marker=dict(colors=['green', 'red', 'blue'])
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800, 
        showlegend=True, 
        barmode='stack',
        margin=dict(t=50, l=50, r=50, b=50)
    )

    # Update axes titles
    fig.update_xaxes(title_text="Days", row=1, col=1)
    fig.update_xaxes(title_text="Days", row=1, col=2)
    fig.update_xaxes(title_text="Days", row=2, col=1)
    
    fig.update_yaxes(title_text="Amount", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Amount", row=1, col=2)
    fig.update_yaxes(title_text="Lock Period (days)", row=2, col=1)
    
    return fig

def plot_monte_carlo_results(result, milestone_data: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Time to Reach ${result.target:,.0f}",
            f"Treasury at ${result.target:,.0f}"
        ]
    )

    if result.times_reached > 0:
        times = milestone_data['time_at_milestone'][result.target]
        time_points = np.linspace(min(times), max(times), 100)
        time_kde = stats.gaussian_kde(times)
        time_density = time_kde(time_points)

        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=time_density,
                mode='lines',
                fill='tozeroy',
                name="Time Distribution",
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_vline(
            x=result.median_time,
            line_dash="dash",
            line_color="green", 
            annotation=dict(
                text=f"Median: {result.median_time:.1f}d",
                y=0.8,
                yanchor='bottom'
            ),
            row=1, col=1
        )

        treasuries = milestone_data['treasury_at_milestone'][result.target]
        treasury_points = np.linspace(min(treasuries), max(treasuries), 100)
        treasury_kde = stats.gaussian_kde(treasuries)
        treasury_density = treasury_kde(treasury_points)

        fig.add_trace(
            go.Scatter(
                x=treasury_points,
                y=treasury_density,
                mode='lines',
                fill='tozeroy',
                name="Treasury Distribution",
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_vline(
            x=result.median_treasury,
            line_dash="dash",
            line_color="green", 
            annotation=dict(
                text=f"Median: ${result.median_treasury:,.0f}",
                y=0.8,
                yanchor='bottom'
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=400,
        showlegend=True,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    return fig