from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
import numpy as np
from datetime import datetime
import pandas as pd

@dataclass
class Node:
    id: int
    deposit: float
    join_time: int
    lock_period: int
    active_until: int
    rewards_generated: float = 0
    rewards_claimed: float = 0
    tvl_share: float = 0
    unclaimed_rewards: float = 0
    principal_withdrawn: bool = False
    reward_withdrawal_history: List[Tuple[int, float, float, float]] = field(default_factory=list)  # [(time, amount_withdrawn, haircut_rate, haircut_amount)]

        
class CoinshiftNetwork:
    def __init__(
        self,
        # Simplified growth model parameters
        initial_growth_rate: float,
        peak_growth_rate: float,
        time_to_peak: int,
        growth_curve_steepness: float,
        # Other parameters
        initial_nodes: int,
        total_population: int,
        tvl_goals: List[Tuple[float, float]],
        min_deposit: float = 1000,
        max_deposit: float = 10_000_000,
        min_lock: int = 180,
        max_lock: int = 365,
        daily_rewards: float = 100000,
        hyperbolic_scale: float = 100,
        reward_withdrawal_prob: float = 0.1,
        principal_withdrawal_prob: float = 0.01,
        starting_tvl: float = 0
    ):
        # Initialize simplified growth parameters
        self.initial_growth_rate = initial_growth_rate
        self.peak_growth_rate = peak_growth_rate
        self.time_to_peak = time_to_peak
        self.growth_curve_steepness = growth_curve_steepness
        
        # Initialize other parameters
        self.total_population = total_population
        self.min_deposit = min_deposit
        self.max_deposit = max_deposit
        self.min_lock = min_lock 
        self.max_lock = max_lock
        self.daily_rewards = daily_rewards          
        self.hyperbolic_scale = hyperbolic_scale
        self.reward_withdrawal_prob = reward_withdrawal_prob
        self.principal_withdrawal_prob = principal_withdrawal_prob
        self.starting_tvl = starting_tvl

        # Initialize empty history lists
        self.tvl_history = []
        self.rewards_history = []
        self.treasury_history = []
        self.active_nodes_history = []
        self.haircut_history = []
        self.total_haircut_collected_history = []
        self.withdrawal_count_history = []
        self.net_claimed_rewards_history = []

        self.nodes = {}
        self.next_id = 0
        self.treasury = 0
        
        self.tvl_goals = sorted(tvl_goals)
        self.current_goal_idx = 0

        # Initialize network with nodes and record initial state
        self._initialize(initial_nodes)

    def _generate_deposit(self) -> float:
        return np.random.uniform(self.min_deposit, self.max_deposit)

    def _generate_lock_period(self) -> int:
        return np.random.randint(self.min_lock, self.max_lock)

    def _add_node(self) -> Node:
        deposit = self._generate_deposit()
        lock_period = self._generate_lock_period()
        node = Node(
            id=self.next_id,
            deposit=deposit,
            join_time=self.current_time,
            lock_period=lock_period,
            active_until=self.current_time + lock_period,
        )
        
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node
    
    def get_node_metrics(self) -> pd.DataFrame:
        data = []
        for node in self.nodes.values():
            data.append({
                'id': node.id,
                'deposit': node.deposit,
                'rewards_generated': node.rewards_generated,
                'rewards_claimed': node.rewards_claimed,
                'unclaimed_rewards': node.unclaimed_rewards,
                'total_rewards': node.rewards_claimed,
                'is_active': node.join_time <= self.current_time and not node.principal_withdrawn,
                'lock_expired': self.current_time >= node.active_until,
                'principal_withdrawn': node.principal_withdrawn,
            })
        return pd.DataFrame(data)

    def _initialize(self, initial_nodes: int):
        self.current_time = 0
        
        if self.starting_tvl > 0:
            avg_deposit = self.starting_tvl / initial_nodes
            min_deposit = max(self.min_deposit, avg_deposit * 0.5)
            max_deposit = min(self.max_deposit, avg_deposit * 1.5)
            self.min_deposit = min_deposit
            self.max_deposit = max_deposit
            
        for _ in range(initial_nodes):
            self._add_node()

        if self.starting_tvl > 0:
            current_tvl = self.calculate_tvl()
            if current_tvl > 0:
                scale_factor = self.starting_tvl / current_tvl
                for node in self.nodes.values():
                    node.deposit *= scale_factor
                
        # Record initial state
        self.tvl_history.append(self.calculate_tvl())
        self.rewards_history.append(0)
        self.active_nodes_history.append(len(self.get_active_nodes()))
        self.treasury_history.append(self.treasury)
        self.haircut_history.append(self.calculate_haircut())
        self.total_haircut_collected_history.append(0)
        self.withdrawal_count_history.append(0)
        self.net_claimed_rewards_history.append(0)

    def get_active_nodes(self) -> Set[int]:
        return {
            id for id, node in self.nodes.items() 
            if node.join_time <= self.current_time and not node.principal_withdrawn
        }

    def calculate_tvl(self) -> float:
        return sum(
            node.deposit 
            for node in self.nodes.values() 
            if node.join_time <= self.current_time and not node.principal_withdrawn
        )
    
    def calculate_haircut(self) -> float:
        if self.current_goal_idx >= len(self.tvl_goals):
            return 0
            
        current_tvl = self.calculate_tvl()
        target_tvl, start_haircut = self.tvl_goals[self.current_goal_idx]
        
        if current_tvl >= target_tvl:
            self.current_goal_idx += 1
            return 0
            
        return start_haircut * (1 - current_tvl / target_tvl)
    
    def distribute_rewards(self):
        active_nodes = self.get_active_nodes()
        if not active_nodes:
            return 0
            
        total_tvl = self.calculate_tvl()
        if total_tvl == 0:
            return 0
            
        current_rewards = self.daily_rewards * (1 / np.cosh(self.current_time / self.hyperbolic_scale))
        
        for node_id in active_nodes:
            node = self.nodes[node_id]
            node.tvl_share = node.deposit / total_tvl
            reward = current_rewards * node.tvl_share
            
            # Add to unclaimed rewards - no haircut applied yet
            node.rewards_generated += reward
            node.unclaimed_rewards += reward
            
        return current_rewards

    def process_withdrawals(self):
        active_nodes = self.get_active_nodes()
        current_haircut = self.calculate_haircut()
        self.haircut_history.append(current_haircut)
        
        withdrawal_count = 0
        total_haircut_collected = 0
        total_claimed = 0
        
        for node_id in active_nodes:
            node = self.nodes[node_id]
            
            # Process reward withdrawals
            if np.random.random() < self.reward_withdrawal_prob and node.unclaimed_rewards > 0:
                haircut_amount = node.unclaimed_rewards * current_haircut
                claimed_amount = node.unclaimed_rewards - haircut_amount
                
                node.rewards_claimed += claimed_amount
                node.reward_withdrawal_history.append((
                    self.current_time,
                    node.unclaimed_rewards,
                    current_haircut,
                    haircut_amount
                ))
                
                node.unclaimed_rewards = 0
                self.treasury += haircut_amount
                total_haircut_collected += haircut_amount
                total_claimed += claimed_amount
                withdrawal_count += 1
            
            # Process principal withdrawals - only if lock period has expired
            if (not node.principal_withdrawn and 
                self.current_time >= node.active_until and 
                np.random.random() < self.principal_withdrawal_prob):
                node.principal_withdrawn = True
                # No haircut on principal
                withdrawal_count += 1

        self.withdrawal_count_history.append(withdrawal_count)
        self.total_haircut_collected_history.append(total_haircut_collected)
        self.net_claimed_rewards_history.append(total_claimed)

    def step(self):
        self.current_time += 1
        remaining = self.total_population - len(self.nodes)
        
        if remaining > 0:
            self._process_network_growth(remaining)

        self.tvl_history.append(self.calculate_tvl())
        rewards = self.distribute_rewards()
        self.rewards_history.append(rewards)
        
        self.active_nodes_history.append(len(self.get_active_nodes()))
        
        self.process_withdrawals()
        self.treasury_history.append(self.treasury)

    def calculate_growth_rate(self) -> float:
        """Calculate the growth rate using the simplified growth model"""
        
        # S-shaped logistic function for growth rate transition
        # This provides a smooth transition from initial to peak growth rate
        t = self.current_time
        k = self.growth_curve_steepness
        t0 = self.time_to_peak
        
        # Logistic function: initial_rate + (peak_rate - initial_rate) / (1 + exp(-k*(t-t0)))
        # When t = t0, the function value is exactly halfway between initial and peak
        logistic_factor = 1 / (1 + np.exp(-k * (t - t0)))
        growth_rate = self.initial_growth_rate + (self.peak_growth_rate - self.initial_growth_rate) * logistic_factor
        
        # Apply population constraint (slow down as we approach total population)
        current_population = len(self.nodes)
        population_factor = max(0, (self.total_population - current_population) / self.total_population)
        growth_rate *= population_factor
        
        # Add random noise (10% standard deviation)
        noise = np.random.normal(0, growth_rate * 0.1)
        return max(0, growth_rate + noise)

    def _process_network_growth(self, remaining: int):
        # Calculate growth rate using the simplified model
        growth_rate = self.calculate_growth_rate()
        
        # Determine number of new nodes (stochastic process using Poisson)
        new_nodes = np.random.poisson(growth_rate)
        new_nodes = min(new_nodes, remaining)
        
        # Add new nodes
        for _ in range(new_nodes):
            self._add_node()