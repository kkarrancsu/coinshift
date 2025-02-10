import streamlit as st
from mc import MonteCarloSimulator
from coinshift import CoinshiftNetwork
import pandas as pd
from viz import (
    plot_network_metrics, 
    plot_withdrawal_metrics,
    plot_monte_carlo_results,
    plot_withdrawal_distributions,
    plot_milestone_metrics,
    display_monte_carlo_metrics
)

import streamlit as st
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class SimulationParams:
    n_steps: int
    referral_rate: float
    spontaneous_rate: float
    hyperbolic_scale: float
    initial_nodes: int
    total_population: int
    referral_bonus: float
    withdrawal_prob: float
    min_deposit: float
    max_deposit: float
    referral_pool: float
    tvl_goals: List[Tuple[float, float]]

def get_simulation_params() -> SimulationParams:
    with st.sidebar:
        st.header("Simulation Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            n_steps = st.number_input(
                "Simulation steps", 
                1, 10000, 365,
                help="Number of days to simulate"
            )
            referral_rate = st.number_input(
                "Referral rate", 
                0.0, 1.0, 0.05, 0.01,
                help="New users per existing user per day"
            )
            spontaneous_rate = st.number_input(
                "Spontaneous rate", 
                0.0, 10.0, 2.0, 0.1,
                help="Organic new users per day"
            )
            hyperbolic_scale = st.number_input(
                "Hyperbolic scale",
                1.0, 1000.0, 100.0, 10.0,
                help="Controls rewards decay speed"
            )
            
        with col2:
            initial_nodes = st.number_input(
                "Initial nodes", 
                1, 100, 10,
                help="Starting number of users"
            )
            total_population = st.number_input(
                "Total population", 
                initial_nodes, 10000, 1000,
                help="Maximum possible users"
            )
            referral_bonus = st.number_input(
                "Referral bonus %", 
                0.0, 100.0, 0.0, 0.1,
                help="Bonus % of referred user's deposit"
            ) / 100
            withdrawal_prob = st.number_input(
                "Withdrawal probability", 
                0.0, 1.0, 0.1, 0.01,
                help="Daily withdrawal chance"
            )

        st.header("Deposit Range")
        min_deposit = st.number_input(
            "Min deposit", 
            0.0, 1e6, 1000.0, 100.0,
            help="Minimum csUSDL deposit"
        )
        max_deposit = st.number_input(
            "Max deposit", 
            min_deposit, 1e8, 10_000.0, 1000.0,
            help="Maximum csUSDL deposit"
        )

        st.header("Referral Pool")
        referral_pool = st.number_input(
            "Total referral bonus pool (SHIFT)", 
            0.0, 1e8, 1_000_000.0, 100_000.0,
            help="Total SHIFT for referral bonuses"
        )

        st.header("TVL Goals")
        num_goals = st.number_input(
            "Number of TVL goals", 
            1, 5, 2, 1,
            help="Number of TVL milestones"
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
                    key=f"tvl_{i}"
                )
            with col2:
                haircut = st.number_input(
                    f"Initial Haircut {i+1} (%)", 
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0 - i * 20.0,
                    step=5.0,
                    key=f"haircut_{i}"
                )
            tvl_goals.append((tvl, haircut / 100.0))
        
        tvl_goals.sort()

    return SimulationParams(
        n_steps=n_steps,
        referral_rate=referral_rate,
        spontaneous_rate=spontaneous_rate,
        hyperbolic_scale=hyperbolic_scale,
        initial_nodes=initial_nodes,
        total_population=total_population,
        referral_bonus=referral_bonus,
        withdrawal_prob=withdrawal_prob,
        min_deposit=min_deposit,
        max_deposit=max_deposit,
        referral_pool=referral_pool,
        tvl_goals=tvl_goals
    )

def get_network_params(params: SimulationParams) -> Dict[str, Any]:
    return {
        "referral_rate": params.referral_rate,
        "spontaneous_rate": params.spontaneous_rate,
        "initial_nodes": params.initial_nodes,
        "total_population": params.total_population,
        "tvl_goals": params.tvl_goals,
        "min_deposit": params.min_deposit,
        "max_deposit": params.max_deposit,
        "referral_bonus_pct": params.referral_bonus,
        "hyperbolic_scale": params.hyperbolic_scale,
        "withdrawal_prob": params.withdrawal_prob,
        "total_referral_bonus_pool": params.referral_pool
    }

def get_monte_carlo_params(base_params: SimulationParams) -> Dict[str, Any]:
    col1, col2 = st.columns(2)
    with col1:
        n_simulations = st.number_input(
            "Number of Simulations",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of Monte Carlo simulations"
        )
    with col2:
        max_steps = st.number_input(
            "Maximum Steps",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Maximum simulation days"
        )
    
    network_params = get_network_params(base_params)
    
    return {
        "n_simulations": n_simulations,
        "max_steps": max_steps,
        "network_params": network_params
    }

def run_single_simulation(network_params: dict, n_steps: int):
    sim = CoinshiftNetwork(**network_params)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    data = {
        "Time": [], "TVL": [], "Active Nodes": [], "Inactive Nodes": [],
        "Treasury": [], "Haircut": [], "Rewards": [], "Referral Rewards": [],
        "Daily Withdrawals": [], "Withdrawal Count": [], 
        "Haircut Collected": [], "Net Withdrawn": [],
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
        fig = plot_network_metrics(df, network_params["tvl_goals"])
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        fig = plot_withdrawal_metrics(df)
        st.plotly_chart(fig, use_container_width=True)
    
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

def run_monte_carlo_simulation(mc_params: dict):
    simulator = MonteCarloSimulator(**mc_params)
    
    with st.spinner("Running Monte Carlo simulations..."):
        results = simulator.run_simulation()
        milestone_data = {
            'nodes_at_milestone': simulator.nodes_at_milestone,
            'referral_rewards_at_milestone': simulator.referral_rewards_at_milestone,
            'treasury_at_milestone': simulator.treasury_at_milestone,
            'time_at_milestone': simulator.milestone_data
        }
        
    for result in results:
        st.subheader(f"TVL Goal: ${result.target:,.0f}")
        st.text(display_monte_carlo_metrics(result, mc_params["n_simulations"]))
        
        if result.times_reached > 0:
            tabs = st.tabs(["Time & Treasury", "Withdrawals", "Network Metrics"])
            
            with tabs[0]:
                fig = plot_monte_carlo_results(result, milestone_data)
                st.plotly_chart(fig, use_container_width=True)
                
            with tabs[1]:
                fig = plot_withdrawal_distributions(result, mc_params["n_simulations"])
                st.plotly_chart(fig, use_container_width=True)
                
            with tabs[2]:
                fig = plot_milestone_metrics(result, milestone_data)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(page_title="Coinshift Network Simulation", layout="wide")
    
    tab1, tab2 = st.tabs(["Single Simulation", "Monte Carlo Analysis"])
    
    params = get_simulation_params()
    
    with tab1:
        if st.sidebar.button("Run Simulation"):
            network_params = get_network_params(params)
            run_single_simulation(network_params, params.n_steps)
            
    with tab2:
        mc_params = get_monte_carlo_params(params)
        if st.sidebar.button("Run Monte Carlo"):
            run_monte_carlo_simulation(mc_params)