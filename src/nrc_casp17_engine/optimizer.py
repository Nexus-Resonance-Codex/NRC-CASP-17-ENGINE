"""
optimizer.py — Optimization algorithms (L-BFGS-B and Simulated Annealing)
"""

import numpy as np
from scipy.optimize import minimize


class NRCOptimizer:
    """
    Handles energy minimization using gradient-based L-BFGS-B
    and Monte Carlo Simulated Annealing.
    """

    def __init__(self, energy_and_gradient_fn):
        """
        energy_and_gradient_fn: callable taking a flat coordinate array (3N,)
                                and returning (energy_val, gradient_array)
        """
        self.energy_and_gradient_fn = energy_and_gradient_fn

    def minimize_lbfgs(self, x0: np.ndarray, max_iter: int = 500) -> np.ndarray:
        """Run standard L-BFGS-B optimization with tightened tolerances."""
        res = minimize(
            self.energy_and_gradient_fn,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "gtol": 1e-7, "ftol": 1e-10},
        )
        return res.x

    def minimize_annealing(
        self,
        x0: np.ndarray,
        cycles: int = 5,
        steps_per_cycle: int = 50,
        t_start: float = 1.0,
        t_end: float = 0.01,
    ) -> np.ndarray:
        """
        Run Simulated Annealing (Monte Carlo with Minimization) to traverse local minima.
        """
        current_x = np.copy(x0)
        current_e, _ = self.energy_and_gradient_fn(current_x)

        best_x = np.copy(current_x)
        best_e = current_e

        # Exponential cooling schedule
        temp = t_start
        alpha = (t_end / t_start) ** (1.0 / max(1, cycles - 1))

        for cycle in range(cycles):
            for step in range(steps_per_cycle):
                # Propose perturbation (scale by current temperature)
                perturbation = np.random.normal(0, 0.1 * temp, size=x0.shape)
                proposed_x = current_x + perturbation

                # Perform a short L-BFGS-B relaxation to bring it to local minimum
                relaxed_x = self.minimize_lbfgs(proposed_x, max_iter=10)
                relaxed_e, _ = self.energy_and_gradient_fn(relaxed_x)

                # Metropolis criteria
                d_energy = relaxed_e - current_e
                if d_energy < 0 or np.random.rand() < np.exp(
                    -d_energy / max(1e-9, temp)
                ):
                    current_x = relaxed_x
                    current_e = relaxed_e

                    if current_e < best_e:
                        best_x = np.copy(current_x)
                        best_e = current_e

            temp *= alpha

        # Final intensive minimization on best found state
        return self.minimize_lbfgs(best_x, max_iter=200)
