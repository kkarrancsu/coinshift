import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple
from scipy import stats
import numpy as np

def plot_network_metrics(df: pd.DataFrame, tvl_goals: List[Tuple[float, float]]) -> go.Figure:
    # Add more detailed debug prints
    print("\nDetailed TVL data analysis:")
    print("DataFrame shape:", df.shape)
    print("TVL column info:", df["TVL"].describe())
    print("Time values:", df["Time"].values[:5])
    print("TVL values:", df["TVL"].values[:5])
    
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "TVL Over Time (csUSDL)", "Active vs Inactive Nodes",
            "Treasury Balance (SHIFT)", "Haircut Evolution (%)",
            "Daily Rewards (SHIFT)", "Referral Pool & Rewards (SHIFT)",
            "SHIFT Token Distribution", "Locked vs Circulating Ratio (%)"
        ]
    )
    
    # TVL plot with goals - add more explicit trace creation
    tvl_trace = go.Scatter(
        x=df["Time"],
        y=df["TVL"] / 1e6,  # Convert to millions
        name="TVL (M csUSDL)",
        mode='lines+markers'  # Add markers to see individual points more clearly
    )
    print("\nTVL trace data (in millions):")
    print("x values:", tvl_trace.x[:5])
    print("y values:", tvl_trace.y[:5])
    
    fig.add_trace(tvl_trace, row=1, col=1)
    
    # Update y-axis to ensure it starts from 0 or lower and has a reasonable upper bound
    min_tvl = (df["TVL"] / 1e6).min()  # Convert to millions
    max_tvl = (df["TVL"] / 1e6).max()  # Convert to millions
    y_range = [
        min_tvl * 0.9 if min_tvl > 0 else 0,
        max_tvl * 1.2  # Increase upper bound by 20% for better visibility
    ]
    
    print(f"\nY-axis range: {y_range}")  # Debug print
    
    fig.update_yaxes(
        range=y_range,
        title_text="TVL (M csUSDL)",
        row=1, col=1,
        tickformat=".0f"  # Format ticks without decimal places
    )
    
    # Update TVL goals visualization to use millions
    for goal, haircut_start in tvl_goals:
        goal_in_m = goal / 1e6  # Convert goal to millions
        print(f"Adding goal line at {goal_in_m}M")  # Debug print
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
                text=f"Goal {goal_in_m:.1f}M csUSDL reached",
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
                text=f"Goal: {goal_in_m:.1f}M csUSDL",
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
        go.Scatter(x=df["Time"], y=df["Treasury"], name="Treasury (SHIFT)"),
        row=2, col=1
    )
    
    # Haircut
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Haircut"].mul(100), name="Haircut %"),
        row=2, col=2
    )
    
    # Rewards
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Rewards"], name="Daily Rewards (SHIFT)"),
        row=3, col=1
    )
    
    # Referral Pool and Rewards
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Remaining Referral Pool"], name="Remaining Pool"),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Referral Rewards"], name="Referral Rewards"),
        row=3, col=2
    )
    
    # Calculate and add SHIFT token distribution
    df['Total_SHIFT_Minted'] = df['Rewards'].cumsum()
    df['Circulating_SHIFT'] = df['Total_SHIFT_Minted'] - df['Treasury']
    
    # Token distribution stacked area
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Treasury"],
            name="Locked in Treasury",
            fill='tozeroy',
            mode='lines',
            line=dict(width=0.5),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df['Total_SHIFT_Minted'],
            name="Total Minted",
            fill='tonexty',
            mode='lines',
            line=dict(width=0.5),
            fillcolor='rgba(0, 255, 0, 0.3)'
        ),
        row=4, col=1
    )
    
    # Locked ratio
    df['Locked_Ratio'] = (df['Treasury'] / df['Total_SHIFT_Minted'] * 100).fillna(0)
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df['Locked_Ratio'],
            name="Locked Ratio",
            mode='lines',
            line=dict(color='red')
        ),
        row=4, col=2
    )
    
    # Add a horizontal line at 100% for ratio reference
    fig.add_hline(
        y=100, 
        line_dash="dash", 
        line_color="gray",
        row=4, col=2
    )
    
    fig.update_layout(
        height=1200,  # Increased height to accommodate new row
        showlegend=True,
        yaxis7_title="SHIFT Tokens",  # Token distribution y-axis
        yaxis8_title="Locked Ratio (%)"  # Ratio y-axis
    )
    
    return fig

def plot_withdrawal_metrics(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Daily Reward Withdrawals (SHIFT)",
            "Cumulative Reward Withdrawals (SHIFT)",
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
        barmode='stack'
    )

    # Update axes titles
    fig.update_xaxes(title_text="Time (days)", row=1, col=1)
    fig.update_xaxes(title_text="Time (days)", row=1, col=2)
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    
    fig.update_yaxes(title_text="Amount (SHIFT)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Amount (SHIFT)", row=1, col=2)
    fig.update_yaxes(title_text="Lock Period (days)", row=2, col=1)
    
    return fig

def plot_milestone_metrics(result, milestone_data: dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Lock Period Distribution",
            "Rewards & Haircuts",
            "Principal Retention",
            "Node Activity"
        ]
    )

    # Lock period distribution
    if result.target in milestone_data['active_lock_periods_at_milestone']:
        lock_periods = milestone_data['active_lock_periods_at_milestone'][result.target]
        if lock_periods:
            fig.add_trace(
                go.Violin(
                    y=np.concatenate(lock_periods),
                    name="Lock Periods",
                    box_visible=True,
                    line_color='blue'
                ),
                row=1, col=1
            )

    # Rewards and haircuts
    if result.mean_rewards_haircut is not None:
        total_rewards = result.mean_treasury + result.mean_rewards_haircut
        fig.add_trace(
            go.Pie(
                values=[result.mean_treasury, result.mean_rewards_haircut],
                labels=["Net Rewards", "Haircut"],
                marker=dict(colors=['green', 'red'])
            ),
            row=1, col=2
        )

    # Principal retention
    times = milestone_data['time_at_milestone'][result.target]
    principals = milestone_data['locked_principal_at_milestone'][result.target]
    if times and principals:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=principals,
                mode='markers',
                marker=dict(color='blue'),
                name='Locked Principal'
            ),
            row=2, col=1
        )

    # Node activity
    nodes = milestone_data['nodes_at_milestone'][result.target]
    if times and nodes:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=nodes,
                mode='markers',
                marker=dict(color='orange'),
                name='Active Nodes'
            ),
            row=2, col=2
        )

    fig.update_layout(height=800, showlegend=False)
    return fig

def plot_monte_carlo_results(result, milestone_data: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Time to Reach ${result.target:,.0f}",
            "Portfolio Composition"
        ]
    )

    if result.times_reached > 0:
        # Time distribution
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
                name="Time Distribution"
            ),
            row=1, col=1
        )
        
        fig.add_vline(
            x=result.median_time,
            line_dash="dash",
            line_color="green",
            annotation=dict(
                text=f"Median: {result.median_time:.1f}d",
                y=0.8
            ),
            row=1, col=1
        )

        # Portfolio composition
        if result.locked_principal is not None:
            fig.add_trace(
                go.Pie(
                    values=[
                        result.locked_principal,
                        result.mean_treasury,
                        result.mean_rewards_haircut or 0
                    ],
                    labels=["Locked Principal", "Treasury", "Haircuts"],
                    marker=dict(colors=['blue', 'green', 'red'])
                ),
                row=1, col=2
            )

    fig.update_layout(height=400, showlegend=False)
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
        height=300,
        showlegend=False,
        margin=dict(t=30, l=50, r=50, b=50)
    )

    fig.update_xaxes(title_text="Days", row=1, col=1)
    fig.update_xaxes(title_text="Treasury Balance", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

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

    fig.update_xaxes(title_text="Total Haircut (csUSDL)", row=1, col=1)
    fig.update_xaxes(title_text="Total Withdrawals (csUSDL)", row=1, col=2)
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
            "Withdrawal Rate"
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
    if times and treasuries:
        df = pd.DataFrame({'time': times, 'treasury': treasuries})
        df['withdrawal_rate'] = df['treasury'] / df['time']
        df['time_rounded'] = df['time'].round()
        
        grouped = df.groupby('time_rounded').agg({
            'withdrawal_rate': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['time', 'mean', 'std']
        
        sorted_idx = grouped['time'].argsort()
        grouped = grouped.iloc[sorted_idx]

        fig.add_trace(
            go.Scatter(
                x=grouped['time'],
                y=grouped['mean'],
                mode='lines',
                name='Withdrawal Rate',
                line=dict(color='red'),
                showlegend=False
            ),
            row=2, col=2
        )

        # Plot standard deviation as error bars
        fig.add_trace(
            go.Scatter(
                x=grouped['time'],
                y=grouped['mean'],
                error_y=dict(
                    type='data',
                    array=grouped['std'],
                    color='red',
                    thickness=1,
                    width=0
                ),
                mode='lines',
                line=dict(color='red'),
                name='Withdrawal Rate ± σ',
                showlegend=False
            ),
            row=2, col=2
        )

    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(t=30, l=50, r=50, b=50)
    )

    fig.update_xaxes(title_text="Number of Nodes", row=1, col=1)
    fig.update_xaxes(title_text="Time (days)", row=1, col=2)
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Growth Rate (nodes/day)", row=1, col=2)
    fig.update_yaxes(title_text="Active Nodes", row=2, col=1)
    fig.update_yaxes(title_text="Withdrawal Rate (csUSDL/day)", row=2, col=2)

    return fig

def plot_optimization_results(result, milestone_data: dict) -> go.Figure:
    # Create figure with 5 subplots (2 rows, 3 columns)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Time to Target Distribution",
            "ROI Distribution",
            "Haircut Collection Distribution",
            "Network Growth Rate",
            "Node Growth Trajectories",  # Moved to bottom row
            "Node Count Distribution at Target"
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

    # 2. ROI Distribution
    treasuries = milestone_data['treasury_at_milestone'][result.target_tvl]
    principals = milestone_data['locked_principal_at_milestone'][result.target_tvl]
    if treasuries and principals:
        rois = np.array([t/p * 100 for t, p in zip(treasuries, principals) if p > 0])
        if len(rois) > 1:
            try:
                kde = stats.gaussian_kde(rois)
                x = np.linspace(min(rois), max(rois), 100)
                y = kde(x)
                fig.add_trace(
                    go.Scatter(x=x, y=y, fill='tozeroy', name="ROI Distribution"),
                    row=1, col=2
                )
            except np.linalg.LinAlgError:
                fig.add_trace(
                    go.Histogram(x=rois, histnorm='probability density', name="ROI Distribution"),
                    row=1, col=2
                )
            
            fig.add_vline(
                x=np.median(rois),
                line=dict(color="red", dash="dash"),
                annotation=dict(text=f"Median: {np.median(rois):.1f}%"),
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
                row=1, col=3
            )
        except np.linalg.LinAlgError:
            fig.add_trace(
                go.Histogram(x=haircuts, histnorm='probability density', name="Haircut Distribution"),
                row=1, col=3
            )
        
        median_haircut = np.median(haircuts)
        fig.add_vline(
            x=median_haircut,
            line=dict(color="red", dash="dash"),
            annotation=dict(text=f"Median: {median_haircut:,.0f}"),
            row=1, col=3
        )

    # 4. Network Growth - Updated implementation
    times = milestone_data['time_at_milestone'][result.target_tvl]
    nodes = milestone_data['nodes_at_milestone'][result.target_tvl]
    if times and nodes:
        df = pd.DataFrame({'time': times, 'nodes': nodes})
        
        # Sort by time and create evenly spaced bins
        df = df.sort_values('time')
        n_bins = min(20, len(df)//5)
        
        # Create time bins using linear spacing instead of quantiles
        df['time_bin'] = pd.cut(
            df['time'], 
            bins=n_bins,
            labels=False
        )
        
        # Calculate statistics for each bin
        bin_stats = df.groupby('time_bin').agg({
            'time': 'mean',
            'nodes': ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]
        }).reset_index()
        
        bin_stats.columns = ['bin', 'time', 'nodes_median', 'nodes_25', 'nodes_75']
        
        # Plot median line
        fig.add_trace(
            go.Scatter(
                x=bin_stats['time'],
                y=bin_stats['nodes_median'],
                mode='lines',
                name="Median Nodes",
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=list(bin_stats['time']) + list(bin_stats['time'])[::-1],
                y=list(bin_stats['nodes_75']) + list(bin_stats['nodes_25'])[::-1],
                fill='toself',
                fillcolor='rgba(0,0,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='25-75th Percentile'
            ),
            row=2, col=1
        )
        
        # Add scatter plot of actual points with low opacity
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['nodes'],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=3,
                    opacity=0.1
                ),
                name='Individual Simulations'
            ),
            row=2, col=1
        )

    # Add node growth trajectories plot
    if 'node_trajectories' in milestone_data:
        print(f"Found {len(milestone_data['node_trajectories'])} trajectories")
        
        for ii, trajectory in enumerate(milestone_data['node_trajectories']):
            if ii == 0:
                print(trajectory)
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
                row=2, col=2  # New position
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
            row=2, col=2  # New position
        )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Optimization Results (Target TVL: ${float(result.target_tvl/1e6):.1f}M, Haircut: {result.haircut*100:.1f}%)"
    )

    # Update axes titles
    fig.update_xaxes(title_text="Days to Target", row=1, col=1)
    fig.update_xaxes(title_text="Return on Investment (%)", row=1, col=2)
    fig.update_xaxes(title_text="Total Haircut Collected", row=1, col=3)
    fig.update_xaxes(title_text="Days", row=2, col=1)
    fig.update_xaxes(title_text="Days", row=2, col=2)  # For node trajectories
    fig.update_xaxes(title_text="Number of Nodes", row=2, col=3)

    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=3)
    fig.update_yaxes(title_text="Growth Rate", row=2, col=1)
    fig.update_yaxes(title_text="Number of Nodes", row=2, col=2)  # For node trajectories
    fig.update_yaxes(title_text="Density", row=2, col=3)

    # Add grid lines for node trajectories plot
    fig.update_xaxes(
        gridcolor='lightgrey',
        showgrid=True,
        row=2, col=2
    )
    fig.update_yaxes(
        gridcolor='lightgrey',
        showgrid=True,
        row=2, col=2
    )

    return fig