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
    display_monte_carlo_metrics,
    plot_optimization_results
)

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from tvl_optimizer import TVLOptimizer, ConstantExchangeRate, GrowthPhaseEstimator

@dataclass
class SimulationParams:
    n_steps: int
    initial_nodes: int
    total_population: int
    # Simplified growth model parameters
    initial_growth_rate: float
    peak_growth_rate: float
    time_to_peak: int
    growth_curve_steepness: float
    # Rewards and Withdrawal parameters
    hyperbolic_scale: float
    reward_withdrawal_prob: float
    principal_withdrawal_prob: float
    min_deposit: float
    max_deposit: float
    tvl_goals: List[Tuple[float, float]]
    starting_tvl: float

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
            initial_nodes = st.number_input(
                "Initial nodes", 
                10, 1000, 100,
                help="Starting number of users"
            )
            
        with col2:
            starting_tvl = st.number_input(
                "Starting TVL (M)",
                0.0, 1000.0, 10.0,
                help="Initial TVL value in millions"
            ) * 1e6
            total_population = st.number_input(
                "Total population", 
                initial_nodes, 50000, 10000,
                help="Maximum possible users"
            )

        st.header("Growth Model Parameters")
        st.markdown("Configure the network growth S-curve")
        
        col1, col2 = st.columns(2)
        with col1:
            initial_growth_rate = st.number_input(
                "Initial growth rate",
                0.1, 100.0, 2.0, 0.1,
                help="New users per day at the start"
            )
            time_to_peak = st.number_input(
                "Time to peak growth (days)",
                1, 1000, 180,
                help="When growth rate reaches halfway to peak"
            )
        with col2:
            peak_growth_rate = st.number_input(
                "Peak growth rate",
                initial_growth_rate, 1000.0, 20.0, 1.0,
                help="Maximum new users per day"
            )
            growth_curve_steepness = st.number_input(
                "Growth curve steepness",
                0.001, 1.0, 0.05, 0.01,
                help="How quickly growth transitions from initial to peak rate"
            )

        st.header("Other Parameters")
        col1, col2 = st.columns(2)
        with col1:
            hyperbolic_scale = st.number_input(
                "Hyperbolic scale",
                1.0, 1000.0, 100.0, 10.0,
                help="Controls rewards decay speed"
            )
            reward_withdrawal_prob = st.number_input(
                "Reward withdrawal prob", 
                0.0, 1.0, 0.1, 0.01,
                help="Daily probability of withdrawing rewards"
            )
        with col2:
            principal_withdrawal_prob = st.number_input(
                "Principal withdrawal prob", 
                0.0, 1.0, 0.01, 0.01,
                help="Daily probability of withdrawing principal after lock expiry"
            )

        st.header("Deposit Range")
        min_deposit = st.number_input(
            "Min deposit", 
            0.0, 1e6, 100.0, 100.0,
            help="Minimum deposit amount"
        )
        max_deposit = st.number_input(
            "Max deposit", 
            min_deposit, 1e8, 1000.0, 1000.0,
            help="Maximum deposit amount"
        )

        # TVL Goals section
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
                increase_pct = (i + 1) * 50 
                goal_tvl = starting_tvl * (1 + increase_pct/100)
                goal_tvl_m = goal_tvl / 1e6
                
                tvl = st.number_input(
                    f"TVL Goal {i+1} (M) (+{increase_pct}%)", 
                    min_value=0.0,
                    max_value=1000.0,
                    value=goal_tvl_m,
                    step=0.1,
                    key=f"tvl_{i}",
                    help=f"TVL goal in millions (auto-set to {increase_pct}% above starting TVL)"
                ) * 1e6
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
        initial_nodes=initial_nodes,
        total_population=total_population,
        initial_growth_rate=initial_growth_rate,
        peak_growth_rate=peak_growth_rate,
        time_to_peak=time_to_peak,
        growth_curve_steepness=growth_curve_steepness,
        hyperbolic_scale=hyperbolic_scale,
        reward_withdrawal_prob=reward_withdrawal_prob,
        principal_withdrawal_prob=principal_withdrawal_prob,
        min_deposit=min_deposit,
        max_deposit=max_deposit,
        tvl_goals=tvl_goals,
        starting_tvl=starting_tvl
    )

def get_network_params(params: SimulationParams) -> Dict[str, Any]:
    return {
        "initial_growth_rate": params.initial_growth_rate,
        "peak_growth_rate": params.peak_growth_rate,
        "time_to_peak": params.time_to_peak,
        "growth_curve_steepness": params.growth_curve_steepness,
        "initial_nodes": params.initial_nodes,
        "total_population": params.total_population,
        "tvl_goals": params.tvl_goals,
        "min_deposit": params.min_deposit,
        "max_deposit": params.max_deposit,
        "hyperbolic_scale": params.hyperbolic_scale,
        "reward_withdrawal_prob": params.reward_withdrawal_prob,
        "principal_withdrawal_prob": params.principal_withdrawal_prob,
        "starting_tvl": params.starting_tvl
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
        "Time": [sim.current_time],
        "TVL": [sim.tvl_history[0]],
        "Active Nodes": [sim.active_nodes_history[0]],
        "Inactive Nodes": [sim.next_id - sim.active_nodes_history[0]],
        "Treasury": [sim.treasury_history[0]],
        "Haircut": [sim.haircut_history[0]],
        "Rewards": [sim.rewards_history[0]],
        "Withdrawal Count": [sim.withdrawal_count_history[0]],
        "Haircut Collected": [sim.total_haircut_collected_history[0]],
        "Net Claimed Rewards": [sim.net_claimed_rewards_history[0]],
        "Principal Withdrawn": [0],
        "Average Lock Period": [
            sum(sim.nodes[n].lock_period for n in sim.get_active_nodes()) / len(sim.get_active_nodes())
            if sim.get_active_nodes() else 0
        ]
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
        data["Withdrawal Count"].append(sim.withdrawal_count_history[-1])
        data["Haircut Collected"].append(sim.total_haircut_collected_history[-1])
        data["Net Claimed Rewards"].append(sim.net_claimed_rewards_history[-1])
        
        active_nodes = sim.get_active_nodes()
        if active_nodes:
            avg_lock = sum(sim.nodes[n].lock_period for n in active_nodes) / len(active_nodes)
        else:
            avg_lock = 0
        data["Average Lock Period"].append(avg_lock)
        
        principal_withdrawn = sum(
            1 for n in sim.nodes.values() 
            if n.principal_withdrawn and 
            n.active_until == sim.current_time
        )
        data["Principal Withdrawn"].append(principal_withdrawn)

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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", len(node_metrics))
            st.metric("Active Nodes", node_metrics['is_active'].sum())
        with col2:
            st.metric("Total Deposits", f"${node_metrics['deposit'].sum():,.2f}")
            st.metric("Total Rewards", f"${node_metrics['total_rewards'].sum():,.2f}")
        with col3:
            st.metric("Locked Nodes", 
                     len(node_metrics[~node_metrics['principal_withdrawn']]))
            st.metric("Average Lock Period", 
                     f"{df['Average Lock Period'].mean():.1f} days")
        
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
            'treasury_at_milestone': simulator.treasury_at_milestone,
            'time_at_milestone': simulator.milestone_data,
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

def get_optimization_params(params: SimulationParams) -> Dict[str, Any]:
    st.header("Optimization Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        target_days = st.number_input(
            "Target Days",
            min_value=30,
            max_value=365,
            value=90,
            help="Target number of days to reach TVL goal"
        )
        min_success_rate = st.number_input(
            "Minimum Success Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum required success rate for optimization"
        )
    
    with col2:
        shift_rate = st.number_input(
            "Token Exchange Rate",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            help="Exchange rate"
        )
        
        # Add current day parameter
        current_day = st.number_input(
            "Current Day",
            min_value=0,
            max_value=params.n_steps,
            value=0,
            help="Current day in the simulation (affects growth phase)"
        )
    
    # Add growth phase override
    st.subheader("Growth Phase")
    use_auto_detection = st.checkbox(
        "Auto-detect growth phase", 
        value=True,
        help="Automatically determine growth phase based on current day"
    )
    
    growth_phase_override = None
    if not use_auto_detection:
        growth_phase_override = st.selectbox(
            "Override Growth Phase",
            options=[
                "Early Growth", 
                "Accelerating Growth", 
                "Peak Growth", 
                "Decelerating Growth", 
                "Maturity"
            ],
            help="Manually select the network growth phase"
        )
        
        # Add visual indication of selected phase
        st.write("Selected growth phase parameters:")
        
        phase_info = {
            "Early Growth": {"tvl_range": "10-50%", "haircut_range": "70-90%", "aggressiveness": "Conservative"},
            "Accelerating Growth": {"tvl_range": "30-110%", "haircut_range": "60-80%", "aggressiveness": "Moderate"},
            "Peak Growth": {"tvl_range": "50-170%", "haircut_range": "50-70%", "aggressiveness": "Aggressive"},
            "Decelerating Growth": {"tvl_range": "30-100%", "haircut_range": "60-80%", "aggressiveness": "Moderate"},
            "Maturity": {"tvl_range": "10-50%", "haircut_range": "70-90%", "aggressiveness": "Conservative"}
        }
        
        selected_info = phase_info[growth_phase_override]
        cols = st.columns(3)
        cols[0].metric("Target TVL Increase", selected_info["tvl_range"])
        cols[1].metric("Haircut Range", selected_info["haircut_range"])
        cols[2].metric("Strategy", selected_info["aggressiveness"])
        
    # Display estimated growth phase if using auto-detection
    if use_auto_detection and current_day > 0:
        network_params = get_network_params(params)
        estimator = GrowthPhaseEstimator(network_params, current_day)
        detected_phase, growth_rate = estimator.estimate_growth_phase()
        
        st.info(f"Detected growth phase: **{detected_phase}** (Est. growth rate: {growth_rate:.2f} nodes/day)")
        
    return {
        "target_days": target_days,
        "min_success_rate": min_success_rate,
        "exchange_rate_provider": ConstantExchangeRate(shift_rate),
        "current_time": current_day,
        "growth_phase_override": growth_phase_override
    }

def create_optimization_page(network_params: Dict[str, Any], params: SimulationParams):
    opt_params = get_optimization_params(params)
    
    current_tvl = network_params.get("starting_tvl", 0)
    if current_tvl == 0:
        st.error("Please set a starting TVL value")
        return
        
    if st.button("Run TVL Optimization"):
        try:
            optimizer = TVLOptimizer(
                network_params=network_params,
                **opt_params
            )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current: int, total: int, message: str):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(message)
            
            with st.spinner("Running optimization..."):
                result = optimizer.optimize(current_tvl, progress_callback)
                
                fig = plot_optimization_results(result, optimizer.milestone_data)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Optimization Results Summary"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Growth Phase", result.growth_phase)
                        st.metric("Target TVL", f"${result.target_tvl/1e6:.2f}M")
                        st.metric("Success Rate", f"{result.success_rate*100:.1f}%")
                    with col2:
                        st.metric("Haircut", f"{result.haircut*100:.1f}%")
                        st.metric("Expected Days", f"{result.expected_days:.1f}")
                        st.metric("Confidence Interval", f"{result.confidence_interval[0]:.1f}-{result.confidence_interval[1]:.1f} days")
                    with col3:
                        st.metric("TVL Increase", f"{(result.target_tvl/current_tvl - 1)*100:.1f}%")
                        st.metric("Penalty Ratio", f"{result.shift_penalty_ratio*100:.1f}%")
                        st.metric("Est. Growth Rate", f"{result.estimated_growth_rate:.2f} nodes/day")
                
                with st.expander("Raw Optimization Results"):
                    st.json({
                        "target_tvl": result.target_tvl,
                        "haircut": result.haircut,
                        "expected_days": result.expected_days,
                        "success_rate": result.success_rate,
                        "shift_penalty_ratio": result.shift_penalty_ratio,
                        "avg_daily_rewards": result.avg_daily_rewards,
                        "growth_phase": result.growth_phase,
                        "estimated_growth_rate": result.estimated_growth_rate
                    })
            
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            raise e
        
if __name__ == "__main__":
    st.set_page_config(page_title="Network Simulation", layout="wide")
    
    tab1, tab2, tab3 = st.tabs([
        "Single Simulation", 
        "Monte Carlo Analysis",
        "TVL Optimization"
    ])
    
    params = get_simulation_params()
    
    with tab1:
        if st.sidebar.button("Run Simulation"):
            network_params = get_network_params(params)
            run_single_simulation(network_params, params.n_steps)
            
    with tab2:
        mc_params = get_monte_carlo_params(params)
        if st.sidebar.button("Run Monte Carlo"):
            run_monte_carlo_simulation(mc_params)
            
    with tab3:
        network_params = get_network_params(params)
        create_optimization_page(network_params, params)