import pytest
from coinshift import CoinshiftNetwork
from dataclasses import dataclass, field
from typing import Set

def test_network_initialization():
   network = CoinshiftNetwork(
       referral_rate=0.3,
       spontaneous_rate=0.2,
       initial_nodes=5,
       total_population=100,
       min_deposit=1000,
       max_deposit=2000,
       daily_rewards=1000,
       reward_decay_rate=0.95,
       tvl_goals=[(1000, 0.5), (2000, 0.3)],
   )
   
   assert len(network.nodes) == 5
   assert network.current_time == 0
   assert len(network.tvl_history) == 0
   assert network.daily_rewards == 1000
   assert network.reward_decay_rate == 0.95
   assert all(1000 <= node.deposit <= 2000 for node in network.nodes.values())
   assert all(isinstance(node.referred_ids, set) for node in network.nodes.values())

def test_network_growth():
   network = CoinshiftNetwork(
       referral_rate=1.0,
       spontaneous_rate=1.0,
       initial_nodes=2,
       total_population=10,
       daily_rewards=1000,
       tvl_goals=[(1000, 0.5), (2000, 0.3)],
   )
   
   initial_nodes = len(network.nodes)
   network.step()
   
   assert len(network.nodes) > initial_nodes
   assert network.current_time == 1
   assert len(network.tvl_history) == 1
   assert len(network.rewards_history) == 1

def test_rewards_distribution():
   network = CoinshiftNetwork(
       referral_rate=0.0,
       spontaneous_rate=0.0,
       initial_nodes=2,
       total_population=10,
       daily_rewards=1000,
       reward_decay_rate=1.0,
       min_deposit=1000,
       max_deposit=1000,
       tvl_goals=[(1000, 0.5), (2000, 0.3)],
   )
   
   network.step()
   total_rewards = sum(node.rewards_claimed for node in network.nodes.values())
   assert abs(total_rewards - 1000) < 0.01
   assert network.rewards_history[0] == 1000
   
   total_tvl_share = sum(node.tvl_share for node in network.nodes.values())
   assert abs(total_tvl_share - 1.0) < 0.01

def test_referral_tracking():
   network = CoinshiftNetwork(
       referral_rate=1.0,
       spontaneous_rate=0.0,
       initial_nodes=1,
       total_population=10,
       tvl_goals=[(1000, 0.5), (2000, 0.3)],
   )
   
   network.step()
   referred = [n for n in network.nodes.values() if n.referrer_id is not None]
   
   for node in referred:
       referrer = network.nodes[node.referrer_id]
       assert node.id in referrer.referred_ids
       assert isinstance(referrer.referred_ids, set)

def test_withdrawal_and_haircut():
    network = CoinshiftNetwork(
        referral_rate=0.0,
        spontaneous_rate=0.0,
        initial_nodes=2,
        total_population=10,
        tvl_goals=[(1000, 0.5), (2000, 0.3)],
        min_deposit=500,
        max_deposit=500,
        withdrawal_prob=1.0,
        max_withdrawal_pct=0.5
    )
    
    initial_tvl = network.calculate_tvl()
    assert initial_tvl == 1000
    
    haircut = network.calculate_haircut()
    assert haircut == 0.5 * (1 - 1000/1000) == 0
    
    network.process_withdrawals()
    
    for node in network.nodes.values():
        assert node.withdrawn_amount > 0
        assert len(node.withdrawal_history) == 1
        assert node.withdrawal_history[0][0] == network.current_time

def test_tvl_goals_progression():
    network = CoinshiftNetwork(
        referral_rate=0.0,
        spontaneous_rate=0.0,
        initial_nodes=4,
        total_population=10,
        tvl_goals=[(1000, 0.5), (2000, 0.3)],
        min_deposit=500,
        max_deposit=500
    )
    
    assert network.current_goal_idx == 0
    assert network.calculate_tvl() == 2000
    assert network.calculate_haircut() == 0
    assert network.current_goal_idx == 1