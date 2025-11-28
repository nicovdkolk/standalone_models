"""
Custom propeller curves handler.

This module implements interpolation of user-provided KT and KQ curves
as functions of advance coefficient J.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import List, Union, Optional


class CustomCurves:
    """
    Custom propeller curves from user-provided tabulated data.

    Interpolates thrust coefficient KT(J) and torque coefficient KQ(J)
    from user-provided data points. Supports both linear and cubic
    interpolation for smooth curves.

    Parameters:
    -----------
    J_values : List[float]
        Advance coefficient values (must be monotonic) [-]
    KT_values : List[float]
        Thrust coefficient values corresponding to J_values [-]
    KQ_values : List[float]
        Torque coefficient values corresponding to J_values [-]
    interpolation_kind : str, default 'cubic'
        Interpolation method: 'linear' or 'cubic'
    """

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

        # Validation
        self._validate_inputs()

        # Create interpolators
        self.interpolation_kind = interpolation_kind

        # KT interpolator
        self._kt_interpolator = interp1d(
            self.J_values, self.KT_values,
            kind=interpolation_kind,
            bounds_error=False,
            fill_value=(self.KT_values[0], self.KT_values[-1])
        )

        # KQ interpolator
        self._kq_interpolator = interp1d(
            self.J_values, self.KQ_values,
            kind=interpolation_kind,
            bounds_error=False,
            fill_value=(self.KQ_values[0], self.KQ_values[-1])
        )


    def _validate_inputs(self):
        """Validate input data."""
        if len(self.J_values) != len(self.KT_values) or len(self.J_values) != len(self.KQ_values):
            raise ValueError("J_values, KT_values, and KQ_values must have the same length")

        if len(self.J_values) < 2:
            raise ValueError("At least 2 data points required for interpolation")

        # Check monotonicity of J
        if not np.all(np.diff(self.J_values) > 0):
            raise ValueError("J_values must be strictly monotonic increasing")

        # Check for reasonable coefficient ranges
        if np.any(self.KT_values < 0):
            raise ValueError("KT values must be non-negative")
        if np.any(self.KQ_values < 0):
            raise ValueError("KQ values must be non-negative")

    def calculate_kt(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate thrust coefficient KT at given advance coefficient(s).

        Uses interpolation of provided data points.

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]

        Returns:
        --------
        float or np.ndarray
            Thrust coefficient KT [-]
        """
        J = np.asarray(J)
        kt = self._kt_interpolator(J)

        # Ensure non-negative results
        kt = np.maximum(kt, 0.0)

        return float(kt) if J.ndim == 0 else kt

    def calculate_kq(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate torque coefficient KQ at given advance coefficient(s).

        Uses interpolation of provided data points.

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]

        Returns:
        --------
        float or np.ndarray
            Torque coefficient KQ [-]
        """
        J = np.asarray(J)
        kq = self._kq_interpolator(J)

        # Ensure non-negative results
        kq = np.maximum(kq, 0.0)

        return float(kq) if J.ndim == 0 else kq

    def calculate_efficiency(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate propeller efficiency η₀.

        η₀ = (J / 2π) × (KT / KQ)

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]

        Returns:
        --------
        float or np.ndarray
            Propeller efficiency η₀ [-]
        """
        J = np.asarray(J)
        is_scalar = J.ndim == 0
        
        # Ensure J is at least 1D for array operations
        J_work = J if not is_scalar else np.array([J])
        
        # Calculate KT and KQ - ensure they're arrays
        kt = np.asarray(self.calculate_kt(J_work))
        kq = np.asarray(self.calculate_kq(J_work))

        # Calculate efficiency
        eta_0 = np.zeros_like(J_work, dtype=float)
        nonzero_mask = kq > 1e-10  # Avoid division by very small numbers
        eta_0[nonzero_mask] = (J_work[nonzero_mask] / (2 * np.pi)) * (kt[nonzero_mask] / kq[nonzero_mask])

        # Clamp efficiency to reasonable range
        eta_0 = np.clip(eta_0, 0.0, 1.0)

        return float(eta_0[0]) if is_scalar else eta_0

    def max_efficiency_J(self) -> float:
        """
        Find J value that maximizes efficiency.

        Returns:
        --------
        float
            Advance coefficient at maximum efficiency [-]
        """
        # Sample J range based on data
        J_min, J_max = self.J_values[0], self.J_values[-1]
        J_range = np.linspace(J_min, J_max, 100)

        # Calculate efficiencies
        efficiencies = self.calculate_efficiency(J_range)

        # Find maximum
        max_idx = np.argmax(efficiencies)
        return J_range[max_idx]

    @property
    def J_range(self) -> tuple[float, float]:
        """
        Get the valid J range for interpolation.

        Returns:
        --------
        tuple[float, float]
            (J_min, J_max) range [-]
        """
        return float(self.J_values[0]), float(self.J_values[-1])

    @property
    def max_efficiency(self) -> float:
        """
        Get maximum achievable efficiency.

        Returns:
        --------
        float
            Maximum efficiency [-]
        """
        J_opt = self.max_efficiency_J()
        return float(self.calculate_efficiency(J_opt))
