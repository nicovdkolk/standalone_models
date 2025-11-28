"""Interpolation helpers for user-supplied propeller KT/KQ curves."""

import numpy as np
from scipy.interpolate import interp1d
from typing import List, Union, Optional


class CustomCurves:
    """Interpolate KT(J) and KQ(J) from tabulated advance coefficients."""

    # Store KT/KQ data and construct interpolators for user-supplied curves.

    def __init__(
        self,
        J_values: List[float],
        KT_values: List[float],
        KQ_values: List[float],
        interpolation_kind: str = 'cubic'
        ):

        self.J_values = np.array(J_values, dtype=float)
        self.KT_values = np.array(KT_values, dtype=float)
        self.KQ_values = np.array(KQ_values, dtype=float)

        self._validate_inputs()

        self.J_min = float(self.J_values[0])
        self.J_max = self._find_kt_zero_crossing()

        self.interpolation_kind = interpolation_kind
        self._kt_interpolator = interp1d(
            self.J_values,
            self.KT_values,
            kind=interpolation_kind,
            bounds_error=False,
            fill_value=(self.KT_values[0], self.KT_values[-1])
        )
        self._kq_interpolator = interp1d(
            self.J_values,
            self.KQ_values,
            kind=interpolation_kind,
            bounds_error=False,
            fill_value=(self.KQ_values[0], self.KQ_values[-1])
        )


    def _validate_inputs(self):
        """Ensure curves are aligned, monotonic, and valid."""
        if len(self.J_values) != len(self.KT_values) or len(self.J_values) != len(self.KQ_values):
            raise ValueError("J_values, KT_values, and KQ_values must have the same length")

        if len(self.J_values) < 2:
            raise ValueError("At least 2 data points required for interpolation")

        if not np.all(np.diff(self.J_values) > 0):
            raise ValueError("J_values must be strictly monotonic increasing")

        if np.any(self.KQ_values < 0):
            raise ValueError("KQ values must be non-negative")

    def _find_kt_zero_crossing(self) -> float:
        """Locate the J value where KT first crosses zero (or max J if not)."""
        negative_mask = self.KT_values <= 0
        if np.any(negative_mask):
            # Find the first negative point
            first_negative_idx = np.where(negative_mask)[0][0]
            if first_negative_idx > 0:
                # Interpolate to find exact zero crossing
                J_low = self.J_values[first_negative_idx - 1]
                J_high = self.J_values[first_negative_idx]
                KT_low = self.KT_values[first_negative_idx - 1]
                KT_high = self.KT_values[first_negative_idx]
                
                # Linear interpolation to find zero crossing
                if KT_high != KT_low:
                    J_zero = J_low - KT_low * (J_high - J_low) / (KT_high - KT_low)
                    return float(J_zero)
                else:
                    # KT values are equal (shouldn't happen with monotonic J)
                    return float(J_low)
            else:
                # KT is negative from the start, return 0
                return 0.0
        else:
            # KT never crosses zero, return maximum J from input data
            return float(self.J_values[-1])

    def _evaluate_curve(self, interpolator: interp1d, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate a curve interpolator while preserving scalar inputs."""
        J_array = np.asarray(J)
        is_scalar = J_array.ndim == 0
        if is_scalar:
            J_array = np.array([J_array])

        values = interpolator(J_array)
        if is_scalar:
            return float(values[0]) if hasattr(values, '__len__') else float(values)
        return values

    def calculate_kt(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Return interpolated thrust coefficient KT for the given advance coefficient(s).
        """
        return self._evaluate_curve(self._kt_interpolator, J)

    def calculate_kq(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Return interpolated torque coefficient KQ for the given advance coefficient(s).
        """
        return self._evaluate_curve(self._kq_interpolator, J)

    def calculate_efficiency(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate open-water efficiency η₀ = (J / 2π) × (KT / KQ).

        Efficiency is zeroed where KT < 0 or KQ is near zero to avoid invalid values.
        """
        J = np.asarray(J)
        is_scalar = J.ndim == 0
        if is_scalar:
            J = np.array([J])

        kt = np.asarray(self.calculate_kt(J))
        kq = np.asarray(self.calculate_kq(J))
        eta_0 = np.zeros_like(J, dtype=float)

        efficiency_mask = (kt >= 0) & (kq > 1e-6)
        eta_0[efficiency_mask] = (J[efficiency_mask] / (2 * np.pi)) * (kt[efficiency_mask] / kq[efficiency_mask])

        if is_scalar:
            return float(eta_0[0])
        else:
            return eta_0

    def max_efficiency_J(self) -> float:
        """
        Return the advance coefficient that maximizes η₀ over the reference range.
        """
        J_range = np.linspace(self.J_min, self.J_max, 100)
        efficiencies = self.calculate_efficiency(J_range)
        max_idx = np.argmax(efficiencies)
        return J_range[max_idx]

    @property
    def J_range(self) -> tuple[float, float]:
        """
        Reference J domain [J_min, J_max] based on the KT zero crossing.
        """
        return self.J_min, self.J_max

    @property
    def max_efficiency(self) -> float:
        """
        Maximum efficiency value across the reference J range.
        """
        J_opt = self.max_efficiency_J()
        return float(self.calculate_efficiency(J_opt))
