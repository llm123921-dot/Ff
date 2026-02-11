"""
HyperStability Core v2.0
Adaptive AI Stability System
Author: 15-year-old AI developer
License: MIT
"""

import numpy as np
from collections import deque
import pickle
import csv

# ==========================
# HyperStabilityCoreAdvanced
# ==========================

class HyperStabilityCoreAdvanced:
    """
    A single adaptive core that monitors and stabilizes AI behavior.
    Acts as a "health monitor" + "self-healing engine" for AI models.
    """
    
    def __init__(self, num_variables=50, memory_factor=0.88, short_alpha=0.22, long_alpha=0.01,
                 error_clamp=3.0, max_freeze_duration=120, noise_level=0.02, experience_maxlen=2000,
                 warmup_steps=20):
        """
        Initialize the core with adaptive parameters.
        
        Args:
            num_variables: Dimensionality of the state space
            memory_factor: How much to retain previous state
            short_alpha: Learning rate for short-term error memory
            long_alpha: Learning rate for long-term error memory  
            error_clamp: Maximum allowed error value
            max_freeze_duration: Steps before force-unfreezing
            noise_level: Amount of noise for robustness training
            experience_maxlen: Size of experience buffer
            warmup_steps: Steps to ignore freezing (prevents early false alarms)
        """
        
        # Core state
        self.num_variables = num_variables
        self.state = np.random.uniform(0.3, 0.7, num_variables)  # Initial state between 0.3-0.7
        
        # Error tracking
        self.short_error = np.zeros(num_variables)  # Recent errors (fast adaptation)
        self.long_error = np.zeros(num_variables)   # Historical errors (slow adaptation)
        
        # Adaptive parameters
        self.memory_factor = memory_factor
        self.short_alpha = short_alpha
        self.long_alpha = long_alpha
        self.error_clamp = error_clamp
        
        # Safety modes
        self.freeze_mode = False      # Pauses adaptation when unstable
        self.emergency_mode = False   # Extreme caution mode
        self.freeze_counter = 0
        self.max_freeze_duration = max_freeze_duration
        
        # Health metrics
        self.health_score = 1.0       # 1.0 = perfect health, 0.0 = dead
        self.best_health = 1.0
        
        # Training enhancements
        self.noise_level = noise_level  # For robustness
        self.warmup_steps = warmup_steps  # Prevents early freezing
        self.global_trend = 0.0
        
        # Memory buffers
        self.experience = deque(maxlen=experience_maxlen)  # Stores experiences
        self.decision_log = deque(maxlen=1000)  # Stores decisions for analysis
    
    # --------------------
    # Core Update Logic
    # --------------------
    
    def update(self, input_vector: np.ndarray):
        """
        Main update loop. Processes new input and adapts state.
        
        Args:
            input_vector: New data from environment/AI model
        """
        
        # 1. Calculate error between input and current state
        error = np.clip(input_vector - self.state, -self.error_clamp, self.error_clamp)
        abs_error = np.abs(error)
        
        # 2. Update error memories
        self.short_error = (1 - self.short_alpha) * self.short_error + self.short_alpha * abs_error
        self.long_error = (1 - self.long_alpha) * self.long_error + self.long_alpha * abs_error
        
        # 3. Calculate instability trend
        mean_short = np.mean(self.short_error)
        mean_long = np.mean(self.long_error)
        self.global_trend = mean_short - mean_long  # Positive = getting worse
        
        # 4. Safety mode activation logic
        if self.warmup_steps > 0:
            # Warmup phase: ignore instability, let system initialize
            self.warmup_steps -= 1
            self.freeze_mode = False
            self.emergency_mode = False
        else:
            # Normal operation: activate safety modes when trend exceeds threshold
            if self.global_trend > 0.04:
                self.freeze_mode = True
                self.emergency_mode = mean_short > 0.45  # Severe instability
                self.freeze_counter += 1
            else:
                self.freeze_mode = False
                self.emergency_mode = False
                self.freeze_counter = 0
            
            # Force unfreeze if stuck too long
            if self.freeze_counter > self.max_freeze_duration:
                self.freeze_mode = False
                self.emergency_mode = False
                self.freeze_counter = 0
        
        # 5. Dynamic learning rate adjustment
        lr = 0.01 + 0.2 * np.exp(-3 * mean_short)  # Higher error = lower learning rate
        if self.freeze_mode:
            lr *= 0.15  # Drastically reduce learning in freeze mode
        elif self.emergency_mode:
            lr *= 0.05  # Almost zero learning in emergency mode
        
        # 6. Update state with memory factor and error correction
        self.state = self.memory_factor * self.state + (1 - self.memory_factor) * (self.state + lr * error)
        
        # 7. Add noise for robustness (helps escape local minima)
        self.state += np.random.normal(0, self.noise_level, self.num_variables)
        
        # 8. Constrain state to valid range
        self.state = np.clip(self.state, 0.0, 1.0)
        
        # 9. Update health score
        self.health_score += 0.008 - 0.006 * self.freeze_mode
        self.health_score = np.clip(self.health_score, 0.0, 1.0)
        
        # 10. Store experience and decision
        self.experience.append({
            'input': input_vector.copy(), 
            'state': self.state.copy(), 
            'error': error.copy()
        })
        self.decision_log.append({
            'trend': self.global_trend, 
            'freeze': self.freeze_mode, 
            'emergency': self.emergency_mode
        })
        
        # 11. Auto-checkpoint: save if health improved
        if self.health_score > self.best_health:
            self.best_health = self.health_score
            self.save_model(f"best_core_health_{self.best_health:.3f}.pkl")
    
    # --------------------
    # Parameter Adaptation
    # --------------------
    
    def adapt_parameters(self):
        """
        Dynamically adjust internal parameters based on long-term drift.
        Called periodically to improve adaptation.
        """
        drift = np.mean(self.long_error)
        
        # Widen error clamp when drift is high (more tolerance)
        self.error_clamp = np.clip(self.error_clamp * (1 + 0.02 * drift), 2.0, 6.0)
        
        # Slow down short-term adaptation when drifting
        self.short_alpha = np.clip(self.short_alpha * (1 - 0.01 * drift), 0.15, 0.4)
        
        # Speed up long-term adaptation when drifting
        self.long_alpha = np.clip(self.long_alpha * (1 + 0.01 * drift), 0.005, 0.05)
    
    # --------------------
    # Performance Metrics
    # --------------------
    
    def mse(self, target_vector):
        """
        Calculate Mean Squared Error against target.
        
        Args:
            target_vector: Desired state
            
        Returns:
            float: MSE value
        """
        return np.mean((self.state - target_vector)**2)
    
    def compute_reward(self, target_vector):
        """
        Calculate reward signal for reinforcement learning.
        
        Args:
            target_vector: Desired state
            
        Returns:
            float: Reward between 0 and 1 (higher = better)
        """
        mse = self.mse(target_vector)
        return 1.0 / (1.0 + mse)  # Inverse relationship: lower MSE = higher reward
    
    def stability_score(self):
        """
        Calculate overall system stability.
        
        Returns:
            float: Stability score between 0 and 1
        """
        state_std = np.std(self.state)
        health_stability = 1 - abs(self.health_score - self.best_health)
        return 0.7 * (1 - state_std) + 0.3 * health_stability
    
    # --------------------
    # Model Persistence
    # --------------------
    
    def save_model(self, filename):
        """
        Save entire core to file using pickle.
        
        Args:
            filename: Path to save file
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filename):
        """
        Load core from pickle file.
        
        Args:
            filename: Path to load file
            
        Returns:
            HyperStabilityCoreAdvanced: Loaded core instance
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    # --------------------
    # Data Export
    # --------------------
    
    def export_metrics_csv(self, filename):
        """
        Append current metrics to CSV file.
        
        Args:
            filename: Path to CSV file
        """
        metrics = self.metrics()
        
        # Check if file exists to write headers
        try:
            with open(filename, 'r') as f:
                file_exists = True
        except FileNotFoundError:
            file_exists = False
        
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
    
    # --------------------
    # Reporting
    # --------------------
    
    def metrics(self):
        """
        Get current performance metrics.
        
        Returns:
            dict: Current metrics
        """
        return {
            "health": self.health_score,
            "freeze": self.freeze_mode,
            "emergency": self.emergency_mode,
            "trend": self.global_trend,
            "experience_size": len(self.experience),
            "stability": self.stability_score()
        }
    
    def print_performance(self):
        """
        Print current performance to console.
        """
        m = self.metrics()
        print(f"Health: {m['health']:.3f} | Trend: {m['trend']:.4f} | "
              f"Freeze: {m['freeze']} | Emergency: {m['emergency']} | "
              f"Stability: {m['stability']:.3f}")


# ================================
# HyperStabilityEnsembleAdvanced
# ================================

class HyperStabilityEnsembleAdvanced:
    """
    Ensemble of multiple cores for higher reliability.
    Different cores may specialize in different patterns.
    """
    
    def __init__(self, num_variables=50, ensemble_size=3):
        """
        Initialize ensemble with multiple cores.
        
        Args:
            num_variables: Dimensionality of state space
            ensemble_size: Number of cores in ensemble
        """
        self.cores = [HyperStabilityCoreAdvanced(num_variables) for _ in range(ensemble_size)]
        self.step = 0
        self.events = deque(maxlen=500)
    
    def update(self, input_vector: np.ndarray):
        """
        Update all cores with new input.
        
        Args:
            input_vector: New data from environment
        """
        self.step += 1
        
        for core in self.cores:
            core.update(input_vector)
            
            # Periodic parameter adaptation
            if self.step % 50 == 0:
                core.adapt_parameters()
        
        # Monitor ensemble health
        self._monitor_variance()
    
    def _monitor_variance(self):
        """
        Check if cores are diverging too much.
        High variance indicates disagreement between cores.
        """
        states = np.stack([c.state for c in self.cores])
        variance = np.mean(np.var(states, axis=0))
        
        if variance > 0.08:
            self.events.append({
                "step": self.step, 
                "type": "HIGH_CORE_DIVERGENCE", 
                "variance": variance
            })
    
    def state(self):
        """
        Get weighted average state of all cores.
        Healthier cores have more influence.
        
        Returns:
            np.ndarray: Weighted average state
        """
        healths = np.array([c.health_score for c in self.cores])
        states = np.stack([c.state for c in self.cores])
        
        # Weight by health score
        return np.average(states, axis=0, weights=healths)
    
    def report(self):
        """
        Print ensemble status report.
        """
        healths = [c.health_score for c in self.cores]
        freezes = sum(c.freeze_mode for c in self.cores)
        
        print(f"\n{'='*50}")
        print(f" HYPER STABILITY ADVANCED ENSEMBLE ")
        print(f"{'='*50}")
        print(f"Step                : {self.step}")
        print(f"Mean health         : {np.mean(healths):.3f}")
        print(f"Min health          : {np.min(healths):.3f}")
        print(f"Freeze cores        : {freezes}/{len(self.cores)}")
        print(f"State mean          : {self.state().mean():.4f}")
        print(f"State std           : {self.state().std():.4f}")
        print(f"Status              : {'🟢 STABLE' if np.mean(healths) > 0.75 else '🟡 MONITOR'}")
        print(f"Events logged       : {len(self.events)}")
        print(f"{'='*50}\n")


# ================================
# Example Usage
# ================================

if __name__ == "__main__":
    """
    Example: Running the ensemble with stress test.
    10% of inputs are anomalies to test robustness.
    """
    
    print("🚀 Starting HyperStability Core v2.0")
    print("="*50)
    
    # Initialize ensemble
    ensemble = HyperStabilityEnsembleAdvanced(num_variables=50, ensemble_size=3)
    
    # Run simulation for 100 steps
    for step in range(100):
        
        # Stress test: 10% chance of anomalous input
        if np.random.rand() < 0.1:
            input_vector = np.random.rand(50) * 10  # Anomaly (large values)
        else:
            input_vector = np.random.rand(50)       # Normal input
        
        ensemble.update(input_vector)
        
        # Print report every 10 steps
        if step % 10 == 0:
            ensemble.report()
            
            # Print individual core performance
            for i, core in enumerate(ensemble.cores):
                print(f"--- Core {i+1} ---")
                core.print_performance()
    
    # Save first core
    ensemble.cores[0].save_model("core0.pkl")
    print("💾 Saved core0.pkl")
    
    # Calculate MSE against random target
    target_vector = np.random.rand(50)
    mse_value = ensemble.cores[0].mse(target_vector)
    print(f"📊 Core 0 MSE: {mse_value:.4f}")
    
    # Export metrics
    ensemble.cores[0].export_metrics_csv("metrics.csv")
    print("📈 Exported metrics to metrics.csv")
    
    print("\n✅ Simulation complete!")
