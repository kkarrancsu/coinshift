from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Protocol, Callable
import numpy as np
from mc import MonteCarloSimulator

class ExchangeRateProvider(Protocol):
    def get_rate(self, time: int) -> float:
        """Returns SHIFT/csUSDL exchange rate at given time"""
        pass

class ConstantExchangeRate:
    def __init__(self, rate: float):
        self.rate = rate
        
    def get_rate(self, time: int) -> float:
        return self.rate

@dataclass
class OptimizationResult:
    target_tvl: float
    haircut: float
    expected_days: float
    success_rate: float
    confidence_interval: Tuple[float, float]
    expected_roi: float
    shift_penalty_ratio: float  # Ratio of penalties to total SHIFT rewards
    avg_daily_rewards: float
    shift_price_at_target: float

class TVLOptimizer:
    def __init__(
        self,
        network_params: dict,
        target_days: int = 90,
        min_success_rate: float = 0.7,
        n_simulations: int = 100,
        max_steps: int = 365,
        min_roi: float = 0.15,
        target_percentile: float = 50,
        exchange_rate_provider: Optional[ExchangeRateProvider] = None
    ):
        self.base_params = network_params
        self.target_days = target_days
        self.min_success_rate = min_success_rate
        self.n_simulations = n_simulations
        self.max_steps = max_steps
        self.min_roi = min_roi
        self.target_percentile = target_percentile
        self.exchange_rate_provider = exchange_rate_provider or ConstantExchangeRate(1.0)
        self.milestone_data = None
        self.simulator = None

    def _run_mc_simulation(
        self, 
        target_tvl: float, 
        haircut: float,
        n_sims: Optional[int] = None
    ) -> Tuple[List[float], Dict[str, float]]:
        params = self.base_params.copy()
        params["tvl_goals"] = [(target_tvl, haircut)]
        
        self.simulator = MonteCarloSimulator(
            n_simulations=n_sims or self.n_simulations,
            max_steps=self.max_steps,
            network_params=params
        )
        
        results = self.simulator.run_simulation()
        result = results[0]
        
        if result.times_reached == 0:
            return [], {}

        times = self.simulator.milestone_data[target_tvl]
        success_rate = result.times_reached / (n_sims or self.n_simulations)
        median_time = result.median_time or 0

        shift_price = self.exchange_rate_provider.get_rate(median_time)
        total_shift_rewards = result.cumulative_rewards
        shift_penalties = result.mean_treasury
        rewards_in_csusdl = total_shift_rewards * shift_price
        locked_csusdl = result.locked_principal
        roi = rewards_in_csusdl / locked_csusdl if locked_csusdl > 0 else 0
        penalty_ratio = shift_penalties / total_shift_rewards if total_shift_rewards > 0 else 0
        avg_timeframe = np.mean(times)
        daily_shift_rewards = total_shift_rewards / avg_timeframe if avg_timeframe > 0 else 0

        self.milestone_data = {
            'time_at_milestone': self.simulator.milestone_data,
            'nodes_at_milestone': self.simulator.nodes_at_milestone,
            'treasury_at_milestone': self.simulator.treasury_at_milestone,
            'referral_rewards_at_milestone': self.simulator.referral_rewards_at_milestone,
            'locked_principal_at_milestone': self.simulator.locked_principal_at_milestone,
            'haircuts_collected_at_milestone': self.simulator.haircuts_collected_at_milestone,
            'rewards_haircut_at_milestone': self.simulator.rewards_haircut_at_milestone,
            'total_nodes_at_milestone': self.simulator.total_nodes_at_milestone,
            'node_trajectories': self.simulator.node_trajectories
        }

        metrics = {
            "success_rate": success_rate,
            "roi": roi,
            "penalty_ratio": penalty_ratio,
            "mean_time": result.mean_time,
            "median_time": result.median_time,
            "p25_time": result.p25_time,
            "p75_time": result.p75_time,
            "daily_rewards": daily_shift_rewards,
            "shift_price": shift_price,
            "total_shift_rewards": total_shift_rewards,
            "shift_penalties": shift_penalties
        }
        
        return times, metrics

    def _evaluate_combination(
        self, 
        target_tvl: float, 
        haircut: float,
        current_tvl: float
    ) -> Tuple[float, Dict[str, float]]:
        times, metrics = self._run_mc_simulation(target_tvl, haircut)
        
        if not times or metrics["success_rate"] < self.min_success_rate:
            return float('-inf'), metrics

        time_at_percentile = np.percentile(times, self.target_percentile)
        
        # Scoring components
        time_score = -abs(time_at_percentile - self.target_days) / self.target_days
        roi_score = max(0, min(1, metrics["roi"] / self.min_roi - 1))
        
        # Penalty ratio score - we want sufficient penalties to discourage early withdrawal
        # but not so high that it discourages participation
        optimal_penalty_ratio = 0.3  # Target 30% of rewards going to penalties
        penalty_score = 1 - abs(metrics["penalty_ratio"] - optimal_penalty_ratio)
        
        # Growth reasonableness
        growth_ratio = target_tvl / current_tvl
        growth_score = max(0, min(1, 2 - (growth_ratio / 3)))
        
        # Combined score
        total_score = (
            time_score * 0.4 +      # Time to reach target
            roi_score * 0.3 +       # ROI adequacy
            penalty_score * 0.2 +   # Penalty effectiveness
            growth_score * 0.1      # Growth reasonableness
        )
        
        return total_score, metrics

    def optimize(self, current_tvl: float) -> OptimizationResult:
        best_score = float('-inf')
        best_result = None
        best_metrics = None
        
        tvl_ratios = np.linspace(1.2, 3.0, 10)
        haircuts = np.linspace(0.1, 0.9, 9)
        
        for _ in range(2):
            for tvl_ratio in tvl_ratios:
                target_tvl = current_tvl * tvl_ratio
                
                for haircut in haircuts:
                    score, metrics = self._evaluate_combination(target_tvl, haircut, current_tvl)
                    
                    if score > best_score:
                        best_score = score
                        best_result = (target_tvl, haircut)
                        best_metrics = metrics
            
            if best_result:
                tvl_base = best_result[0]
                haircut_base = best_result[1]
                tvl_ratios = np.linspace(0.9 * tvl_base/current_tvl, 1.1 * tvl_base/current_tvl, 5)
                haircuts = np.linspace(max(0.1, haircut_base - 0.1), min(0.9, haircut_base + 0.1), 5)
        
        if not best_result or not best_metrics:
            raise ValueError("Failed to find valid optimization result")
            
        # Run one final simulation with more samples for better distribution analysis
        _, final_metrics = self._run_mc_simulation(
            best_result[0], 
            best_result[1],
            n_sims=500  # Increase number of simulations for final analysis
        )
            
        return OptimizationResult(
            target_tvl=best_result[0],
            haircut=best_result[1],
            expected_days=best_metrics["mean_time"],
            success_rate=best_metrics["success_rate"],
            confidence_interval=(best_metrics["p25_time"], best_metrics["p75_time"]),
            expected_roi=best_metrics["roi"],
            shift_penalty_ratio=best_metrics["penalty_ratio"],
            avg_daily_rewards=best_metrics["daily_rewards"],
            shift_price_at_target=best_metrics["shift_price"]
        )