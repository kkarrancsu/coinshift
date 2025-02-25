from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from mc import MonteCarloSimulator

class ExchangeRateProvider:
    def get_rate(self, time: int) -> float:
        """Returns token exchange rate at given time"""
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
    shift_penalty_ratio: float
    avg_daily_rewards: float
    shift_price_at_target: float
    growth_phase: str
    estimated_growth_rate: float

class GrowthPhaseEstimator:
    def __init__(self, network_params: dict, current_time: int):
        self.params = network_params
        self.current_time = current_time
        
    def estimate_growth_phase(self) -> Tuple[str, float]:
        """Determine the current growth phase based on time and params"""
        t = self.current_time
        t0 = self.params["time_to_peak"]
        k = self.params["growth_curve_steepness"]
        initial_rate = self.params["initial_growth_rate"]
        peak_rate = self.params["peak_growth_rate"]
        
        # Calculate current position on the logistic curve
        logistic_factor = 1 / (1 + np.exp(-k * (t - t0)))
        current_growth_rate = initial_rate + (peak_rate - initial_rate) * logistic_factor
        
        # Apply population constraint
        total_population = self.params["total_population"]
        initial_nodes = self.params["initial_nodes"]
        
        # Estimate current population based on time
        # This is a rough estimate; in production you'd use actual node count
        estimated_population = initial_nodes + \
            (self.current_time * (initial_rate + current_growth_rate) / 2)
        estimated_population = min(estimated_population, total_population)
        
        population_factor = max(0, (total_population - estimated_population) / total_population)
        adjusted_growth_rate = current_growth_rate * population_factor
        
        # Determine phase based on position relative to peak
        if t < t0 * 0.5:
            phase = "Early Growth"
        elif t < t0 * 0.9:
            phase = "Accelerating Growth" 
        elif t < t0 * 1.1:
            phase = "Peak Growth"
        elif t < t0 * 1.5:
            phase = "Decelerating Growth"
        else:
            phase = "Maturity"
            
        return phase, adjusted_growth_rate

class TVLOptimizer:
    def __init__(
        self,
        network_params: dict,
        current_time: int = 0,
        target_days: int = 90,
        min_success_rate: float = 0.7,
        n_simulations: int = 25,
        max_steps: int = 365,
        target_percentile: float = 50,
        exchange_rate_provider: Optional[ExchangeRateProvider] = None,
        growth_phase_override: Optional[str] = None
    ):
        self.base_params = network_params
        self.current_time = current_time
        self.target_days = target_days
        self.min_success_rate = min_success_rate
        self.n_simulations = n_simulations
        self.max_steps = max_steps
        self.target_percentile = target_percentile
        self.exchange_rate_provider = exchange_rate_provider or ConstantExchangeRate(1.0)
        self.milestone_data = None
        self.simulator = None
        
        # Growth phase estimation
        self.growth_estimator = GrowthPhaseEstimator(network_params, current_time)
        if growth_phase_override:
            self.growth_phase = growth_phase_override
            _, self.estimated_growth_rate = self.growth_estimator.estimate_growth_phase()
        else:
            self.growth_phase, self.estimated_growth_rate = self.growth_estimator.estimate_growth_phase()
            
    def _get_growth_adjusted_tvl_ratios(self) -> np.ndarray:
        """Return appropriate TVL ratio ranges based on growth phase"""
        if self.growth_phase == "Early Growth":
            return np.asarray([1.1, 1.2, 1.3, 1.4, 1.5])
        elif self.growth_phase == "Accelerating Growth":
            return np.asarray([1.3, 1.5, 1.7, 1.9, 2.1])
        elif self.growth_phase == "Peak Growth":
            return np.asarray([1.5, 1.8, 2.1, 2.4, 2.7])
        elif self.growth_phase == "Decelerating Growth":
            return np.asarray([1.3, 1.5, 1.7, 1.9, 2.0])
        else:  # Maturity
            return np.asarray([1.1, 1.2, 1.3, 1.4, 1.5])
            
    def _get_growth_adjusted_haircuts(self) -> np.ndarray:
        """Return appropriate haircut ranges based on growth phase"""
        if self.growth_phase == "Early Growth":
            return np.asarray([0.7, 0.75, 0.8, 0.85, 0.9])
        elif self.growth_phase == "Accelerating Growth":
            return np.asarray([0.6, 0.65, 0.7, 0.75, 0.8])
        elif self.growth_phase == "Peak Growth":
            return np.asarray([0.5, 0.55, 0.6, 0.65, 0.7])
        elif self.growth_phase == "Decelerating Growth":
            return np.asarray([0.6, 0.65, 0.7, 0.75, 0.8])
        else:  # Maturity
            return np.asarray([0.7, 0.75, 0.8, 0.85, 0.9])

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
        penalty_ratio = shift_penalties / total_shift_rewards if total_shift_rewards > 0 else 0
        avg_timeframe = np.mean(times)
        daily_shift_rewards = total_shift_rewards / avg_timeframe if avg_timeframe > 0 else 0

        self.milestone_data = {
            'time_at_milestone': self.simulator.milestone_data,
            'nodes_at_milestone': self.simulator.nodes_at_milestone,
            'treasury_at_milestone': self.simulator.treasury_at_milestone,
            'locked_principal_at_milestone': self.simulator.locked_principal_at_milestone,
            'haircuts_collected_at_milestone': self.simulator.haircuts_collected_at_milestone,
            'rewards_haircut_at_milestone': self.simulator.rewards_haircut_at_milestone,
            'total_nodes_at_milestone': self.simulator.total_nodes_at_milestone,
            'node_trajectories': self.simulator.node_trajectories,
            'final_tvl_at_milestone': self.simulator.final_tvl_at_milestone
        }

        metrics = {
            "success_rate": success_rate,
            "penalty_ratio": penalty_ratio,
            "mean_time": result.mean_time,
            "median_time": result.median_time,
            "p25_time": result.p25_time,
            "p75_time": result.p75_time,
            "daily_rewards": daily_shift_rewards,
            "shift_price": shift_price,
            "total_shift_rewards": total_shift_rewards,
            "shift_penalties": shift_penalties,
            "final_tvl_at_milestone": result.final_tvl
        }
        
        return times, metrics

    def _evaluate_combination(
        self, 
        target_tvl: float, 
        haircut: float,
    ) -> Tuple[float, Dict[str, float]]:
        times, metrics = self._run_mc_simulation(target_tvl, haircut)
        
        if not times or metrics["success_rate"] < self.min_success_rate:
            return float('-inf'), metrics

        time_at_percentile = np.percentile(times, self.target_percentile)
        
        # Scoring components
        time_score = -abs(time_at_percentile - self.target_days) / self.target_days
        
        # Adjust scoring based on growth phase
        growth_multiplier = 1.0
        if self.growth_phase == "Early Growth":
            growth_multiplier = 0.9  # More conservative in early growth
        elif self.growth_phase == "Accelerating Growth":
            growth_multiplier = 1.1  # More aggressive in accelerating growth
        elif self.growth_phase == "Peak Growth":
            growth_multiplier = 1.2  # Most aggressive at peak growth
        elif self.growth_phase == "Decelerating Growth":
            growth_multiplier = 1.0  # Balanced in decelerating growth
        else:  # Maturity
            growth_multiplier = 0.8  # Most conservative in maturity
        
        # Penalty ratio score - we want sufficient penalties to discourage early withdrawal
        # but not so high that it discourages participation
        optimal_penalty_ratio = 0.3  # Target 30% of rewards going to penalties
        penalty_score = 1 - abs(metrics["penalty_ratio"] - optimal_penalty_ratio)
        
        # Growth reasonableness
        tvl_ratio = target_tvl / metrics["final_tvl_at_milestone"]
        tvl_error = tvl_ratio - 1.0
        tvl_score = max(0, 1 - (tvl_error * tvl_error / 4))
        
        # Combined score with growth phase adjustment
        total_score = (
            time_score * 0.4 * growth_multiplier +  # Time to reach target
            penalty_score * 0.2 +                   # Penalty effectiveness
            tvl_score * 0.4                         # Growth reasonableness
        )
        
        return total_score, metrics

    def optimize(self, current_tvl: float, progress_callback: Callable[[int, int, str], None] = None) -> OptimizationResult:
        best_score = float('-inf')
        best_result = None
        best_metrics = None
        
        # Growth-adjusted values
        tvl_ratios = self._get_growth_adjusted_tvl_ratios()
        haircuts = self._get_growth_adjusted_haircuts()
        
        total_iterations = 2 * len(tvl_ratios) * len(haircuts)
        current_iteration = 0
        
        for iteration in range(2):
            for tvl_ratio in tvl_ratios:
                target_tvl = current_tvl * tvl_ratio
                
                for haircut in haircuts:
                    if progress_callback:
                        current_iteration += 1
                        progress_callback(
                            current_iteration, 
                            total_iterations,
                            f"Testing TVL ratio: {tvl_ratio:.2f}, Haircut: {haircut:.2f}"
                        )
                        
                    score, metrics = self._evaluate_combination(target_tvl, haircut)
                    
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
        if progress_callback:
            progress_callback(total_iterations, total_iterations, "Running final detailed analysis...")
            
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
            shift_penalty_ratio=best_metrics["penalty_ratio"],
            avg_daily_rewards=best_metrics["daily_rewards"],
            shift_price_at_target=best_metrics["shift_price"],
            growth_phase=self.growth_phase,
            estimated_growth_rate=self.estimated_growth_rate
        )