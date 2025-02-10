from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from coinshift import CoinshiftNetwork

@dataclass
class TVLMilestoneStats:
    target: float
    times_reached: int
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    mean_time: Optional[float] = None
    median_time: Optional[float] = None
    p25_time: Optional[float] = None
    p75_time: Optional[float] = None
    mean_treasury: Optional[float] = None
    median_treasury: Optional[float] = None
    mean_active_nodes: Optional[float] = None
    mean_referral_rewards: Optional[float] = None
    haircut_rate: Optional[float] = None

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
        self.results: List[Dict] = []
        self.nodes_at_milestone: Dict[float, List[int]] = {}
        self.referral_rewards_at_milestone: Dict[float, List[float]] = {}
        self.treasury_at_milestone: Dict[float, List[float]] = {}
        self.milestone_data: Dict[float, List[int]] = {}
        
    def run_simulation(self) -> List[TVLMilestoneStats]:
        self.milestone_data = {goal[0]: [] for goal in self.tvl_goals}
        self.treasury_at_milestone = {goal[0]: [] for goal in self.tvl_goals}
        self.nodes_at_milestone = {goal[0]: [] for goal in self.tvl_goals}
        self.referral_rewards_at_milestone = {goal[0]: [] for goal in self.tvl_goals}
        
        for sim in range(self.n_simulations):
            network = CoinshiftNetwork(**self.network_params)
            reached_goals = set()
            
            for step in range(self.max_steps):
                network.step()
                current_tvl = network.tvl_history[-1]
                
                for goal in self.tvl_goals:
                    target = goal[0]
                    if target not in reached_goals and current_tvl >= target:
                        self.milestone_data[target].append(step)
                        self.treasury_at_milestone[target].append(network.treasury)
                        self.nodes_at_milestone[target].append(
                            len(network.get_active_nodes())
                        )
                        self.referral_rewards_at_milestone[target].append(
                            sum(node.referral_rewards for node in network.nodes.values())
                        )
                        reached_goals.add(target)
                
                if len(reached_goals) == len(self.tvl_goals):
                    break
                        
        stats = []
        for goal in self.tvl_goals:
            target = goal[0]
            times = self.milestone_data[target]
            treasuries = self.treasury_at_milestone[target]
            nodes = self.nodes_at_milestone[target]
            rewards = self.referral_rewards_at_milestone[target]
            
            if times:
                times_array = np.array(times)
                haircut_rate = np.mean([goal[1] for goal in self.tvl_goals])
                stats.append(TVLMilestoneStats(
                    target=target,
                    times_reached=len(times),
                    min_time=float(np.min(times_array)),
                    max_time=float(np.max(times_array)),
                    mean_time=float(np.mean(times_array)),
                    median_time=float(np.median(times_array)),
                    p25_time=float(np.percentile(times_array, 25)),
                    p75_time=float(np.percentile(times_array, 75)),
                    mean_treasury=float(np.mean(treasuries)),
                    median_treasury=float(np.median(treasuries)),
                    mean_active_nodes=float(np.mean(nodes)),
                    mean_referral_rewards=float(np.mean(rewards)),
                    haircut_rate=haircut_rate
                ))
            else:
                stats.append(TVLMilestoneStats(
                    target=target,
                    times_reached=0
                ))
                    
        return stats