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
    referrer_id: Optional[int] = None
    referred_ids: Set[int] = field(default_factory=set)
    referral_rewards: float = 0
    unclaimed_rewards: float = 0
    principal_withdrawn: bool = False
    reward_withdrawal_history: List[Tuple[int, float, float, float]] = field(default_factory=list)  # [(time, amount_withdrawn, haircut_rate, haircut_amount)]

        
class CoinshiftNetwork:
    def __init__(
        self,
        referral_rate: float,
        spontaneous_rate: float,
        referral_bonus_pct,
        # New growth params (used when referral_bonus_pct = 0)
        base_growth_rate: float,
        max_growth_rate: float,
        growth_inflection_point: int,
        growth_steepness: float,
        # End of new growth params
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
        total_referral_bonus_pool: float = 1_000_000,
        starting_tvl: float = 0
    ):
        # Initialize parameters
        self.referral_rate = referral_rate
        self.spontaneous_rate = spontaneous_rate
        self.total_population = total_population
        self.min_deposit = min_deposit
        self.max_deposit = max_deposit
        self.min_lock = min_lock 
        self.max_lock = max_lock
        self.daily_rewards = daily_rewards          
        self.hyperbolic_scale = hyperbolic_scale
        self.referral_bonus_pct = referral_bonus_pct
        self.reward_withdrawal_prob = reward_withdrawal_prob
        self.principal_withdrawal_prob = principal_withdrawal_prob
        self.starting_tvl = starting_tvl

        # New growth params
        self.base_growth_rate = base_growth_rate
        self.max_growth_rate = max_growth_rate
        self.growth_inflection_point = growth_inflection_point
        self.growth_steepness = growth_steepness

        self.remaining_referral_bonus = total_referral_bonus_pool
        
        # Initialize empty history lists
        self.tvl_history = []
        self.rewards_history = []
        self.referral_rewards_history = []
        self.treasury_history = []
        self.active_nodes_history = []
        self.referral_nodes_history = []
        self.haircut_history = []
        self.total_haircut_collected_history = []
        self.withdrawal_count_history = []
        self.net_claimed_rewards_history = []
        self.referral_bonus_history = []

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

    def _add_node(self, referrer_id: Optional[int] = None) -> Node:
        deposit = self._generate_deposit()
        lock_period = self._generate_lock_period()
        node = Node(
            id=self.next_id,
            deposit=deposit,
            join_time=self.current_time,
            lock_period=lock_period,
            active_until=self.current_time + lock_period,
            referrer_id=referrer_id
        )
        
        if referrer_id is not None:
            referrer = self.nodes[referrer_id]
            referrer.referred_ids.add(node.id)
            
            potential_bonus = deposit * self.referral_bonus_pct
            if self.remaining_referral_bonus >= potential_bonus:
                referrer.referral_rewards += potential_bonus
                self.remaining_referral_bonus -= potential_bonus
            
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
                'referral_rewards': node.referral_rewards,
                'total_rewards': node.rewards_claimed + node.referral_rewards,
                'is_active': node.join_time <= self.current_time and not node.principal_withdrawn,
                'lock_expired': self.current_time >= node.active_until,
                'principal_withdrawn': node.principal_withdrawn,
                'num_referrals': len(node.referred_ids),
                'successful_referrals': len([rid for rid in node.referred_ids 
                                        if self.nodes[rid].referrer_id == node.id])
            })
        return pd.DataFrame(data)

    def _initialize(self, initial_nodes: int):
        self.current_time = 0
        print(f"\nInitializing network:")
        print(f"Starting TVL target: {self.starting_tvl:,.2f}")
        print(f"Initial nodes: {initial_nodes}")
        
        if self.starting_tvl > 0:
            avg_deposit = self.starting_tvl / initial_nodes
            min_deposit = max(self.min_deposit, avg_deposit * 0.5)
            max_deposit = min(self.max_deposit, avg_deposit * 1.5)
            print(f"Calculated deposit range: {min_deposit:,.2f} - {max_deposit:,.2f}")
            self.min_deposit = min_deposit
            self.max_deposit = max_deposit
            
        for _ in range(initial_nodes):
            self._add_node()

        if self.starting_tvl > 0:
            current_tvl = self.calculate_tvl()
            print(f"Initial TVL before scaling: {current_tvl:,.2f}")
            if current_tvl > 0:
                scale_factor = self.starting_tvl / current_tvl
                print(f"Applying scale factor: {scale_factor:.2f}")
                for node in self.nodes.values():
                    node.deposit *= scale_factor
                
        final_tvl = self.calculate_tvl()
        print(f"Final initial TVL: {final_tvl:,.2f}")
        
        # Record initial state
        self.tvl_history.append(self.calculate_tvl())
        self.rewards_history.append(0)
        self.referral_rewards_history.append(0)
        self.active_nodes_history.append(len(self.get_active_nodes()))
        self.referral_nodes_history.append(
            sum(1 for n in self.nodes.values() if n.referrer_id is not None)
        )
        self.treasury_history.append(self.treasury)
        self.haircut_history.append(self.calculate_haircut())
        self.total_haircut_collected_history.append(0)
        self.withdrawal_count_history.append(0)
        self.net_claimed_rewards_history.append(0)
        self.referral_bonus_history.append(self.remaining_referral_bonus)

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

        self.referral_bonus_history.append(self.remaining_referral_bonus)
        
        total_referral_rewards = sum(node.referral_rewards for node in self.nodes.values())
        self.referral_rewards_history.append(total_referral_rewards)
        
        self.active_nodes_history.append(len(self.get_active_nodes()))
        self.referral_nodes_history.append(
            sum(1 for n in self.nodes.values() if n.referrer_id is not None)
        )
        
        self.process_withdrawals()
        self.treasury_history.append(self.treasury)

    def calculate_growth_rate(self) -> float:
        growth_factor = 1 / (1 + np.exp(-self.growth_steepness * 
                                      (self.current_time - self.growth_inflection_point)))
        rate = self.base_growth_rate + (self.max_growth_rate - self.base_growth_rate) * growth_factor
        
        # Add random noise
        noise = np.random.normal(0, rate * 0.1)  # 10% standard deviation
        return max(0, rate + noise)

    def _process_network_growth(self, remaining: int):
        pop_factor = remaining / self.total_population
        
        if self.referral_bonus_pct > 0:
            # Original growth model with referrals
            new_spontaneous = np.random.poisson(self.spontaneous_rate * pop_factor)
            new_spontaneous = min(new_spontaneous, remaining)
            
            for _ in range(new_spontaneous):
                self._add_node()
                remaining -= 1
                
            if remaining <= 0:
                return
                
            active_nodes = self.get_active_nodes()
            if active_nodes:
                referral_rate = self.referral_rate * (remaining / self.total_population) 
                new_referrals = np.random.poisson(referral_rate * len(active_nodes))
                new_referrals = min(new_referrals, remaining)
                
                for _ in range(new_referrals):
                    referrer = np.random.choice(list(active_nodes))
                    self._add_node(referrer)
        else:
            # New S-curve growth model
            growth_rate = self.calculate_growth_rate()
            new_nodes = np.random.poisson(growth_rate * pop_factor)
            new_nodes = min(new_nodes, remaining)
            
            for _ in range(new_nodes):
                self._add_node()  # No referrer needed since bonus is 0

