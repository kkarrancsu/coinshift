import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from coinshift import CoinshiftNetwork
import pandas as pd

st.set_page_config(page_title="Coinshift Network Simulation", layout="wide")
st.title("Coinshift Network Simulation")

with st.sidebar:
    st.header("Simulation Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        n_steps = st.number_input(
            "Simulation steps", 
            1, 10000, 365,
            help="Number of days to simulate. Each step represents one day."
        )
        referral_rate = st.number_input(
            "Referral rate", 
            0.0, 1.0, 0.05, 0.01,
            help="Average number of new users each existing user brings per day. A rate of 0.05 means each user brings 1 new user every 20 days on average."
        )
        spontaneous_rate = st.number_input(
            "Spontaneous rate", 
            0.0, 10.0, 2.0, 0.1,
            help="Number of new users that join organically per day without referrals."
        )
        reward_decay_rate = st.number_input(
            "Reward decay rate", 
            0.0, 1.0, 0.99, 0.01,
            help="Daily decay factor for rewards. A rate of 0.99 means rewards decrease by 1% each day."
        )
        
    with col2:
        initial_nodes = st.number_input(
            "Initial nodes", 
            1, 100, 10,
            help="Number of users at the start of the simulation."
        )
        total_population = st.number_input(
            "Total population", 
            initial_nodes, 10000, 1000,
            help="Maximum number of users that can join the network."
        )
        referral_bonus = st.number_input(
            "Referral bonus %", 
            0.0, 100.0, 5.0, 0.1,
            help="Percentage of referred user's deposit paid as bonus to referrer."
        ) / 100
        withdrawal_prob = st.number_input(
            "Withdrawal probability", 
            0.0, 1.0, 0.1, 0.01,
            help="Chance of a user making a withdrawal on any given day."
        )

    st.header("Deposit Range")
    min_deposit = st.number_input(
        "Min deposit", 
        0.0, 1e6, 1000.0, 100.0,
        help="Minimum amount in csUSDL that a user can deposit."
    )
    max_deposit = st.number_input(
        "Max deposit", 
        min_deposit, 1e8, 10_000.0, 1000.0,
        help="Maximum amount in csUSDL that a user can deposit."
    )

    st.header("Referral Pool")
    referral_pool = st.number_input(
        "Total referral bonus pool (SHIFT)", 
        0.0, 1e8, 1_000_000.0, 100_000.0,
        help="Total amount of SHIFT tokens allocated for referral bonuses. Once depleted, no more referral bonuses are paid."
    )

    st.header("TVL Goals")
    num_goals = st.number_input(
        "Number of TVL goals", 
        1, 5, 2, 1,
        help="Number of Total Value Locked (TVL) milestones to track."
    )
    
    tvl_goals = []
    for i in range(num_goals):
        col1, col2 = st.columns(2)
        with col1:
            tvl = st.number_input(
                f"TVL Goal {i+1} (csUSDL)", 
                min_value=0.0,
                max_value=1e9,
                value=1_000_000.0 * (i + 1),
                step=100_000.0,
                key=f"tvl_{i}",
                help=f"Target {i+1} for Total Value Locked in csUSDL. Haircut decreases as TVL approaches this goal."
            )
        with col2:
            haircut = st.number_input(
                f"Initial Haircut {i+1} (%)", 
                min_value=0.0,
                max_value=100.0,
                value=80.0 - i * 20.0,
                step=5.0,
                key=f"haircut_{i}",
                help=f"Starting withdrawal penalty percentage for TVL Goal {i+1}. Decreases linearly as TVL approaches goal."
            )
        tvl_goals.append((tvl, haircut / 100.0))
    
    tvl_goals.sort()  # Sort by TVL target

    st.info("""
        📌 Quick Tips:
        - Higher referral and spontaneous rates lead to faster network growth
        - Lower withdrawal probability and higher lock periods help maintain TVL
        - Withdrawal haircuts decrease as TVL approaches goals
        - Ensure TVL goals are realistic given the population and deposit ranges
    """)

if st.sidebar.button("Run Simulation"):
    sim = CoinshiftNetwork(
        referral_rate=referral_rate,
        spontaneous_rate=spontaneous_rate,
        initial_nodes=initial_nodes,
        total_population=total_population,
        tvl_goals=tvl_goals,
        min_deposit=min_deposit,
        max_deposit=max_deposit,
        referral_bonus_pct=referral_bonus,
        reward_decay_rate=reward_decay_rate,
        withdrawal_prob=withdrawal_prob,
        total_referral_bonus_pool=referral_pool
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    data = {
        "Time": [], 
        "TVL": [], 
        "Active Nodes": [], 
        "Inactive Nodes": [],
        "Treasury": [], 
        "Haircut": [], 
        "Rewards": [], 
        "Referral Rewards": [],
        "Daily Withdrawals": [], 
        "Withdrawal Count": [], 
        "Haircut Collected": [],
        "Net Withdrawn": [],
        "Remaining Referral Pool": []
    }
    
    for step in range(n_steps):
        sim.step()
        
        data["Time"].append(sim.current_time)
        data["TVL"].append(sim.tvl_history[-1])
        active = sim.active_nodes_history[-1]
        data["Active Nodes"].append(active)
        total_nodes = sim.next_id
        data["Inactive Nodes"].append(total_nodes - active)
        data["Treasury"].append(sim.treasury_history[-1])
        data["Haircut"].append(sim.haircut_history[-1])
        data["Rewards"].append(sim.rewards_history[-1])
        data["Referral Rewards"].append(sim.referral_rewards_history[-1])
        data["Daily Withdrawals"].append(sim.withdrawal_amount_history[-1])
        data["Withdrawal Count"].append(sim.withdrawal_count_history[-1])
        data["Haircut Collected"].append(sim.total_haircut_collected_history[-1])
        data["Net Withdrawn"].append(sim.net_withdrawn_history[-1])
        data["Remaining Referral Pool"].append(sim.referral_bonus_history[-1])

        progress = (step + 1) / n_steps
        progress_bar.progress(progress)
        status_text.text(f"Step {step + 1}/{n_steps}")
    
    df = pd.DataFrame(data)
    
    tabs = st.tabs(["Network Metrics", "Withdrawal Metrics", "Node Analysis", "Raw Data"])
    
    with tabs[0]:
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
            # Find when goal was reached
            goal_reached_time = next(
                (t for t, v in zip(df["Time"], df["TVL"]) if v >= goal), 
                None
            )
            
            # Add horizontal line for the goal
            fig.add_hline(
                y=goal,
                line=dict(
                    color="green" if goal_reached_time else "red",
                    dash="dash"
                ),
                row=1, col=1
            )
            
            # Add annotation for the goal
            if goal_reached_time:
                fig.add_annotation(
                    x=goal_reached_time,
                    y=goal,
                    text=f"Goal {goal:,.0f} csUSDL reached",
                    showarrow=True,
                    arrowhead=1,
                    row=1, col=1
                )
                # Add vertical line at goal reached time
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
        
        # Node count plot
        fig.add_trace(
            go.Scatter(x=df["Time"], y=df["Active Nodes"], name="Active Nodes"),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df["Time"], y=df["Inactive Nodes"], name="Inactive Nodes"),
            row=1, col=2
        )
        
        # Treasury plot
        fig.add_trace(
            go.Scatter(x=df["Time"], y=df["Treasury"], name="Treasury (csUSDL)"),
            row=2, col=1
        )
        
        # Haircut plot
        fig.add_trace(
            go.Scatter(x=df["Time"], y=df["Haircut"].mul(100), name="Haircut %"),
            row=2, col=2
        )
        
        # Rewards plots
        fig.add_trace(
            go.Scatter(x=df["Time"], y=df["Rewards"], name="Daily Rewards (SHIFT)"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df["Time"], 
                y=df["Remaining Referral Pool"], 
                name="Remaining Pool (SHIFT)",
                line=dict(color='blue')
            ),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=df["Time"], y=df["Referral Rewards"], 
                      name="Referral Rewards (SHIFT)"),
            row=3, col=2
        )
        # Add a horizontal line for initial pool size
        fig.add_hline(
            y=referral_pool,
            line=dict(color="gray", dash="dash"),
            row=3, col=2
        )
        
        fig.add_annotation(
            x=0,
            y=referral_pool,
            text=f"Initial Pool: {referral_pool:,.0f} SHIFT",
            showarrow=False,
            xanchor="left",
            row=3, col=2
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="csUSDL", row=1, col=1)
        fig.update_yaxes(title_text="Number of Nodes", row=1, col=2)
        fig.update_yaxes(title_text="csUSDL", row=2, col=1)
        fig.update_yaxes(title_text="Percentage", row=2, col=2)
        fig.update_yaxes(title_text="SHIFT", row=3, col=1)
        fig.update_yaxes(title_text="SHIFT", row=3, col=2)
        
        # Update x-axis labels
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
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[1]:
        withdrawal_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Daily Withdrawal Breakdown (csUSDL)",
                "Daily Withdrawal Count",
                "Cumulative Withdrawals (csUSDL)",
                "Withdrawal Distribution"
            ],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "domain"}]  # "domain" type for pie chart
            ]
        )
        
        # Daily withdrawal breakdown - stacked bar
        withdrawal_fig.add_trace(
            go.Bar(
                x=df["Time"],
                y=df["Net Withdrawn"],
                name="Net Withdrawn (csUSDL)",
                marker_color='green'
            ),
            row=1, col=1
        )
        withdrawal_fig.add_trace(
            go.Bar(
                x=df["Time"],
                y=df["Haircut Collected"],
                name="Haircut (csUSDL)",
                marker_color='red'
            ),
            row=1, col=1
        )
        
        # Update barmode for stacking
        withdrawal_fig.update_layout(barmode='stack')
        
        # Daily withdrawal count
        withdrawal_fig.add_trace(
            go.Scatter(
                x=df["Time"],
                y=df["Withdrawal Count"],
                name="Number of Withdrawals"
            ),
            row=1, col=2
        )
        
        # Cumulative withdrawals
        withdrawal_fig.add_trace(
            go.Scatter(
                x=df["Time"],
                y=df["Net Withdrawn"].cumsum(),
                name="Net Withdrawn (csUSDL)",
                line=dict(color='green')
            ),
            row=2, col=1
        )
        withdrawal_fig.add_trace(
            go.Scatter(
                x=df["Time"],
                y=df["Haircut Collected"].cumsum(),
                name="Haircut (csUSDL)",
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Withdrawal distribution pie chart
        total_withdrawn = df["Daily Withdrawals"].sum()
        total_haircut = df["Haircut Collected"].sum()
        total_net = df["Net Withdrawn"].sum()
        
        withdrawal_fig.add_trace(
            go.Pie(
                values=[total_net, total_haircut],
                labels=["Net Withdrawn", "Haircut"],
                marker=dict(colors=['green', 'red'])
            ),
            row=2, col=2
        )
        
        withdrawal_fig.update_layout(
            height=800, 
            showlegend=True,
            # Adjust pie position and size
            margin=dict(t=50, l=50, r=50, b=50)
        )
        st.plotly_chart(withdrawal_fig, use_container_width=True)

    with tabs[2]:
        node_metrics = sim.get_node_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Nodes", len(node_metrics))
            st.metric("Active Nodes", node_metrics['is_active'].sum())
            
        with col2:
            st.metric("Total Deposits", f"${node_metrics['deposit'].sum():,.2f}")
            st.metric("Total Rewards", f"${node_metrics['total_rewards'].sum():,.2f}")
        
        st.dataframe(node_metrics)
    
    with tabs[3]:
        st.dataframe(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "simulation_data.csv",
            "text/csv"
        )