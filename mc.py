from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from coinshift import CoinshiftNetwork

@dataclass
class TVLMilestoneStats:
    target: float
    times_reached: int
    final_tvl: Optional[float] = None
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    mean_time: Optional[float] = None
    median_time: Optional[float] = None
    p25_time: Optional[float] = None
    p75_time: Optional[float] = None
    mean_treasury: Optional[float] = None
    median_treasury: Optional[float] = None
    mean_active_nodes: Optional[float] = None
    mean_total_nodes: Optional[float] = None
    mean_lock_period: Optional[float] = None
    mean_rewards_haircut: Optional[float] = None
    locked_principal: Optional[float] = None
    haircut_rate: Optional[float] = None
    cumulative_rewards: Optional[float] = None
    total_haircut_collected: Optional[float] = None

class MonteCarloSimulator:
    def __init__(
        self,
        n_simulations: int,
        max_steps: int,
        network_params: dict
    ):
        self.n_simulations = n_simulations
        self.max_steps = max_steps
        self.network_params = network_params
        self.tvl_goals = network_params['tvl_goals']

        # Initialize all tracking dictionaries
        self.milestone_data: Dict[float, List[int]] = {}
        self.nodes_at_milestone: Dict[float, List[int]] = {}
        self.total_nodes_at_milestone: Dict[float, List[int]] = {}
        self.treasury_at_milestone: Dict[float, List[float]] = {}
        self.lock_periods_at_milestone: Dict[float, List[float]] = {}
        self.rewards_haircut_at_milestone: Dict[float, List[float]] = {}
        self.locked_principal_at_milestone: Dict[float, List[float]] = {}
        self.active_lock_periods_at_milestone: Dict[float, List[List[float]]] = {}
        self.haircut_rates_at_milestone: Dict[float, List[float]] = {}
        self.cumulative_rewards_at_milestone: Dict[float, List[float]] = {}
        self.final_tvl_at_milestone: Dict[float, List[float]] = {}
        
        for goal, _ in self.tvl_goals:
            self.milestone_data[goal] = []
            self.treasury_at_milestone[goal] = []
            self.nodes_at_milestone[goal] = []
            self.total_nodes_at_milestone[goal] = []
            self.lock_periods_at_milestone[goal] = []
            self.rewards_haircut_at_milestone[goal] = []
            self.locked_principal_at_milestone[goal] = []
            self.active_lock_periods_at_milestone[goal] = []
            self.haircut_rates_at_milestone[goal] = []
            self.cumulative_rewards_at_milestone[goal] = []
            self.final_tvl_at_milestone[goal] = []
        
        self.haircuts_collected_at_milestone = {goal: [] for goal, _ in self.tvl_goals}
        
        # Add new tracking for node trajectories
        self.node_trajectories = []  # Will store List[List[int]] - one trajectory per simulation
        
    def run_simulation(self) -> List[TVLMilestoneStats]:
        for sim in range(self.n_simulations):
            network = CoinshiftNetwork(**self.network_params)
            reached_goals = set()
            last_goal_time = 0
            cumulative_rewards = 0
            
            # Track node counts for this simulation
            node_counts = [len(network.get_active_nodes())]
            
            for step in range(self.max_steps):
                network.step()
                node_counts.append(len(network.get_active_nodes()))
                current_tvl = network.tvl_history[-1]
                
                # Track rewards minted since last goal
                cumulative_rewards += network.rewards_history[-1]
                
                for goal, _ in self.tvl_goals:
                    if goal not in reached_goals and current_tvl >= goal:
                        self.milestone_data[goal].append(step)
                        self.treasury_at_milestone[goal].append(network.treasury)
                        
                        active_nodes = network.get_active_nodes()
                        self.nodes_at_milestone[goal].append(len(active_nodes))
                        self.total_nodes_at_milestone[goal].append(network.next_id)
                        
                        current_haircut = network.calculate_haircut()
                        self.haircut_rates_at_milestone[goal].append(current_haircut)
                        self.cumulative_rewards_at_milestone[goal].append(cumulative_rewards)
                        self.haircuts_collected_at_milestone[goal].append(
                            sum(
                                haircut_amount 
                                for node in network.nodes.values()
                                for _, _, _, haircut_amount in node.reward_withdrawal_history
                            )
                        )
                        
                        active_lock_periods = []
                        total_locked_principal = 0
                        total_rewards_haircut = 0
                        
                        for node_id in active_nodes:
                            node = network.nodes[node_id]
                            active_lock_periods.append(node.lock_period)
                            total_locked_principal += node.deposit
                            
                            if node.reward_withdrawal_history:
                                total_rewards_haircut += sum(
                                    haircut_amount 
                                    for _, _, _, haircut_amount 
                                    in node.reward_withdrawal_history
                                )
                        
                        self.lock_periods_at_milestone[goal].append(
                            np.mean(active_lock_periods)
                        )
                        self.rewards_haircut_at_milestone[goal].append(
                            total_rewards_haircut
                        )
                        self.locked_principal_at_milestone[goal].append(
                            total_locked_principal
                        )
                        self.active_lock_periods_at_milestone[goal].append(
                            active_lock_periods
                        )
                        self.final_tvl_at_milestone[goal].append(
                            current_tvl
                        )
                        
                        # Reset cumulative rewards for next goal
                        cumulative_rewards = 0
                        last_goal_time = step
                        
                        reached_goals.add(goal)
                
                if len(reached_goals) == len(self.tvl_goals):
                    break
            
            # Store the trajectory for this simulation
            self.node_trajectories.append(node_counts)
        
        stats = []
        for goal, _ in self.tvl_goals:
            times = self.milestone_data[goal]
            
            if times:
                times_array = np.array(times)
                haircut_rates = np.array(self.haircut_rates_at_milestone[goal])
                
                stats.append(TVLMilestoneStats(
                    target=goal,
                    times_reached=len(times),
                    min_time=float(np.min(times_array)),
                    max_time=float(np.max(times_array)),
                    mean_time=float(np.mean(times_array)),
                    median_time=float(np.median(times_array)),
                    p25_time=float(np.percentile(times_array, 25)),
                    p75_time=float(np.percentile(times_array, 75)),
                    mean_treasury=float(np.mean(self.treasury_at_milestone[goal])),
                    median_treasury=float(np.median(self.treasury_at_milestone[goal])),
                    mean_active_nodes=float(np.mean(self.nodes_at_milestone[goal])),
                    mean_total_nodes=float(np.mean(self.total_nodes_at_milestone[goal])),
                    mean_lock_period=float(np.mean(self.lock_periods_at_milestone[goal])),
                    mean_rewards_haircut=float(np.mean(self.rewards_haircut_at_milestone[goal])),
                    locked_principal=float(np.mean(self.locked_principal_at_milestone[goal])),
                    haircut_rate=float(np.mean(haircut_rates)),
                    cumulative_rewards=float(np.mean(self.cumulative_rewards_at_milestone[goal])),
                    total_haircut_collected=float(np.mean(self.haircuts_collected_at_milestone[goal])),
                    final_tvl=float(np.mean(self.final_tvl_at_milestone[goal]))
                ))
            else:
                stats.append(TVLMilestoneStats(
                    target=goal,
                    times_reached=0
                ))
                    
        return stats