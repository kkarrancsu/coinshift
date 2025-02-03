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
    rewards_claimed: float = 0
    tvl_share: float = 0
    referrer_id: Optional[int] = None
    referred_ids: Set[int] = field(default_factory=set)
    withdrawn_amount: float = 0  # Total amount requested for withdrawal
    net_withdrawn: float = 0     # Amount actually received after haircut
    referral_rewards: float = 0
    withdrawal_history: List[Tuple[int, float, float]] = field(default_factory=list)  # [(time, amount, haircut)]
        
class CoinshiftNetwork:
    def __init__(
        self,
        referral_rate: float,
        spontaneous_rate: float,
        initial_nodes: int,
        total_population: int,
        tvl_goals: List[Tuple[float, float]],  # [(tvl_target, haircut_start)]
        min_deposit: float = 1000,
        max_deposit: float = 10_000_000,
        min_lock: int = 180,
        max_lock: int = 365,
        daily_rewards: float = 100000,
        reward_decay_rate: float = 0.95,
        withdrawal_prob: float = 0.1,
        max_withdrawal_pct: float = 0.5,
        referral_bonus_pct: float = 0.05,
        total_referral_bonus_pool: float = 1_000_000,  # Total SHIFT available for referrals
    ):
        self.referral_rate = referral_rate
        self.spontaneous_rate = spontaneous_rate
        self.total_population = total_population
        self.min_deposit = min_deposit
        self.max_deposit = max_deposit
        self.min_lock = min_lock 
        self.max_lock = max_lock
        self.daily_rewards = daily_rewards          
        self.reward_decay_rate = reward_decay_rate  
        self.referral_bonus_pct = referral_bonus_pct

        self.remaining_referral_bonus = total_referral_bonus_pool
        self.referral_bonus_history: List[float] = []  # Track remaining bonus over time

        self.nodes: Dict[int, Node] = {}
        self.next_id = 0
        
        self.tvl_history: List[float] = []
        self.rewards_history: List[float] = []
        self.referral_rewards_history: List[float] = []

        self.tvl_goals = sorted(tvl_goals)
        self.withdrawal_prob = withdrawal_prob
        self.max_withdrawal_pct = max_withdrawal_pct
        self.treasury = 0
        self.current_goal_idx = 0

        self.treasury_history: List[float] = []
        self.active_nodes_history: List[int] = []
        self.referral_nodes_history: List[int] = []
        self.haircut_history: List[float] = []
        self.withdrawal_amount_history: List[float] = []
        self.withdrawal_count_history: List[int] = []
        self.total_haircut_collected_history: List[float] = []
        self.net_withdrawn_history: List[float] = []
        self._initialize(initial_nodes)

    def _generate_deposit(self) -> float:
        return np.random.uniform(self.min_deposit, self.max_deposit)

    def _generate_lock_period(self) -> int:
        return np.random.randint(self.min_lock, self.max_lock)

    def _add_node(self, referrer_id: Optional[int] = None) -> Node:
        deposit = self._generate_deposit()
        node = Node(
            id=self.next_id,
            deposit=deposit,
            join_time=self.current_time,
            lock_period=self._generate_lock_period(),
            active_until=self.current_time + self._generate_lock_period(),
            referrer_id=referrer_id
        )
        
        if referrer_id is not None:
            referrer = self.nodes[referrer_id]
            referrer.referred_ids.add(node.id)
            
            # Only pay referral bonus if we have enough in the pool
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
                'rewards': node.rewards_claimed,
                'referral_rewards': node.referral_rewards,
                'total_rewards': node.rewards_claimed + node.referral_rewards,
                'withdrawn': node.withdrawn_amount,
                'net_withdrawn': node.net_withdrawn,
                'haircut_paid': node.withdrawn_amount - node.net_withdrawn,
                'is_active': node.join_time <= self.current_time < node.active_until,
                'num_referrals': len(node.referred_ids),
                'successful_referrals': len([rid for rid in node.referred_ids 
                                          if self.nodes[rid].referrer_id == node.id])
            })
        return pd.DataFrame(data)

    def _initialize(self, initial_nodes: int):
        self.current_time = 0
        for _ in range(initial_nodes):
            self._add_node()

    def calculate_tvl(self) -> float:
        return sum(node.deposit for node in self.nodes.values() 
                  if node.join_time <= self.current_time < node.active_until)
    
    def get_active_nodes(self) -> Set[int]:
        return {
            id for id, node in self.nodes.items() 
            if node.join_time <= self.current_time < node.active_until
        }
    
    def distribute_rewards(self):
        active_nodes = self.get_active_nodes()
        if not active_nodes:
            return 0
            
        total_tvl = self.calculate_tvl()
        if total_tvl == 0:
            return 0
            
        current_rewards = self.daily_rewards * (self.reward_decay_rate ** self.current_time)
        
        for node_id in active_nodes:
            node = self.nodes[node_id]
            node.tvl_share = node.deposit / total_tvl
            reward = current_rewards * node.tvl_share
            node.rewards_claimed += reward
            
        return current_rewards
    
    def calculate_haircut(self) -> float:
        if self.current_goal_idx >= len(self.tvl_goals):
            return 0
            
        current_tvl = self.calculate_tvl()
        target_tvl, start_haircut = self.tvl_goals[self.current_goal_idx]
        
        if current_tvl >= target_tvl:
            self.current_goal_idx += 1
            return 0
            
        return start_haircut * (1 - current_tvl / target_tvl)
    
    def process_withdrawals(self):
        active_nodes = self.get_active_nodes()
        haircut = self.calculate_haircut()
        self.haircut_history.append(haircut)
        
        total_withdrawals = 0
        withdrawal_count = 0
        total_haircut_collected = 0
        total_net_withdrawn = 0
        
        for node_id in active_nodes:
            if np.random.random() < self.withdrawal_prob:
                node = self.nodes[node_id]
                withdrawal_amount = node.deposit * np.random.uniform(0, self.max_withdrawal_pct)
                haircut_amount = withdrawal_amount * haircut
                net_received = withdrawal_amount - haircut_amount
                
                # Update node's deposit and withdrawal tracking
                node.deposit -= withdrawal_amount
                node.withdrawn_amount += withdrawal_amount
                node.net_withdrawn += net_received
                
                # Update protocol treasury
                self.treasury += haircut_amount
                
                # Track withdrawal event
                node.withdrawal_history.append((
                    self.current_time,
                    withdrawal_amount,
                    haircut_amount,
                    net_received
                ))
                
                # Update aggregate statistics
                total_withdrawals += withdrawal_amount
                withdrawal_count += 1
                total_haircut_collected += haircut_amount
                total_net_withdrawn += net_received

        self.withdrawal_amount_history.append(total_withdrawals)
        self.withdrawal_count_history.append(withdrawal_count)
        self.total_haircut_collected_history.append(total_haircut_collected)
        self.net_withdrawn_history.append(total_net_withdrawn)

    def get_withdrawal_metrics(self) -> pd.DataFrame:
        withdrawal_data = []
        for node in self.nodes.values():
            for time, amount, haircut, net in node.withdrawal_history:
                withdrawal_data.append({
                    'time': time,
                    'node_id': node.id,
                    'withdrawal_amount': amount,
                    'haircut_amount': haircut,
                    'net_withdrawn': net,
                    'haircut_percentage': (haircut / amount * 100) if amount > 0 else 0
                })
        return pd.DataFrame(withdrawal_data)

    def step(self):
        self.current_time += 1
        remaining = self.total_population - len(self.nodes)
        
        if remaining > 0:
            self._process_network_growth(remaining)

        self.tvl_history.append(self.calculate_tvl())
        rewards = self.distribute_rewards()
        self.rewards_history.append(rewards)

        # Track remaining referral bonus
        self.referral_bonus_history.append(self.remaining_referral_bonus)
        
        total_referral_rewards = sum(node.referral_rewards for node in self.nodes.values())
        self.referral_rewards_history.append(total_referral_rewards)
        
        self.active_nodes_history.append(len(self.get_active_nodes()))
        self.referral_nodes_history.append(
            sum(1 for n in self.nodes.values() if n.referrer_id is not None)
        )
        
        self.process_withdrawals()
        self.treasury_history.append(self.treasury)

    def _process_network_growth(self, remaining: int):
        active_nodes = self.get_active_nodes()
        pop_factor = remaining / self.total_population
        
        # Spontaneous growth
        new_spontaneous = np.random.poisson(self.spontaneous_rate * pop_factor)
        new_spontaneous = min(new_spontaneous, remaining)
        
        for _ in range(new_spontaneous):
            self._add_node()
            remaining -= 1
            
        if remaining <= 0:
            return
            
        # Referral growth
        if active_nodes:
            referral_rate = self.referral_rate * (remaining / self.total_population) 
            new_referrals = np.random.poisson(referral_rate * len(active_nodes))
            new_referrals = min(new_referrals, remaining)
            
            for _ in range(new_referrals):
                referrer = np.random.choice(list(active_nodes))
                self._add_node(referrer)