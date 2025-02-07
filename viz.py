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
            "Treasury Balance (csUSDL)", "Haircut Evolution (%)",
            "Daily Rewards (SHIFT)", "Referral Pool & Rewards (SHIFT)"
        ]
    )
    
    # TVL plot with goals
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["TVL"], name="TVL (csUSDL)"),
        row=1, col=1
    )
    
    # Add TVL goals visualization
    for goal, haircut_start in tvl_goals:
        goal_reached_time = next(
            (t for t, v in zip(df["Time"], df["TVL"]) if v >= goal), 
            None
        )
        
        fig.add_hline(
            y=goal,
            line=dict(
                color="green" if goal_reached_time else "red",
                dash="dash"
            ),
            row=1, col=1
        )
        
        if goal_reached_time:
            fig.add_annotation(
                x=goal_reached_time,
                y=goal,
                text=f"Goal {goal:,.0f} csUSDL reached",
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
                y=goal,
                text=f"Goal: {goal:,.0f} csUSDL",
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
        go.Scatter(x=df["Time"], y=df["Treasury"], name="Treasury (csUSDL)"),
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
        go.Scatter(
            x=df["Time"], 
            y=df["Remaining Referral Pool"], 
            name="Remaining Pool (SHIFT)"
        ),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=df["Time"], y=df["Referral Rewards"], 
                name="Referral Rewards (SHIFT)"),
        row=3, col=2
    )
    
    # Update axis labels
    fig.update_yaxes(title_text="csUSDL", row=1, col=1)
    fig.update_yaxes(title_text="Number of Nodes", row=1, col=2)
    fig.update_yaxes(title_text="csUSDL", row=2, col=1)
    fig.update_yaxes(title_text="Percentage", row=2, col=2)
    fig.update_yaxes(title_text="SHIFT", row=3, col=1)
    fig.update_yaxes(title_text="SHIFT", row=3, col=2)
    
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Time (days)", row=i, col=j)
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def plot_withdrawal_metrics(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Daily Withdrawal Breakdown",
            "Daily Withdrawal Count",
            "Cumulative Withdrawals",
            "Withdrawal Distribution"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "domain"}]
        ]
    )
    
    # Daily withdrawal breakdown
    fig.add_trace(
        go.Bar(
            x=df["Time"],
            y=df["Net Withdrawn"],
            name="Net Withdrawn",
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
    
    # Daily count
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Withdrawal Count"],
            name="Number of Withdrawals"
        ),
        row=1, col=2
    )
    
    # Cumulative
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Net Withdrawn"].cumsum(),
            name="Net Withdrawn",
            line=dict(color='green')
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Haircut Collected"].cumsum(),
            name="Haircut",
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    # Distribution pie
    total_net = df["Net Withdrawn"].sum()
    total_haircut = df["Haircut Collected"].sum()
    
    fig.add_trace(
        go.Pie(
            values=[total_net, total_haircut],
            labels=["Net Withdrawn", "Haircut"],
            marker=dict(colors=['green', 'red'])
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        barmode='stack'
    )
    
    return fig

def plot_monte_carlo_results(result, n_simulations: int) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Time to Reach ${result.target:,.0f}",
            f"Treasury at ${result.target:,.0f}"
        ]
    )

    if result.times_reached > 0:
        # Generate KDE for time distribution
        time_points = np.linspace(result.min_time, result.max_time, 100)
        time_kde = stats.gaussian_kde([result.min_time, result.mean_time, result.max_time])
        time_density = time_kde(time_points)

        # Time distribution
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
        
        # Add mean/median lines
        fig.add_vline(
            x=result.mean_time,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {result.mean_time:.1f}d",
            row=1, col=1
        )
        fig.add_vline(
            x=result.median_time,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: {result.median_time:.1f}d",
            row=1, col=1
        )

        # Generate KDE for treasury distribution
        treasury_points = np.linspace(0, result.mean_treasury * 2, 100)
        treasury_kde = stats.gaussian_kde([0, result.mean_treasury, result.mean_treasury * 2])
        treasury_density = treasury_kde(treasury_points)

        # Treasury distribution
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
            x=result.mean_treasury,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${result.mean_treasury:,.0f}",
            row=1, col=2
        )

    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(t=30, l=50, r=50, b=50)
    )

    fig.update_xaxes(title_text="Days", row=1, col=1)
    fig.update_xaxes(title_text="Treasury Balance", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=2)

    return fig

def display_monte_carlo_metrics(result, n_simulations: int) -> dict:
    if result.times_reached == 0:
        return {
            "Success Rate": f"{(result.times_reached/n_simulations)*100:.1f}%"
        }
    
    return {
        "Success Rate": f"{(result.times_reached/n_simulations)*100:.1f}%",
        "Active Nodes": f"{result.mean_active_nodes:,.0f}"
    }