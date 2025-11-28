"""
Custom propeller curves handler.

This module implements interpolation of user-provided KT and KQ curves
as functions of advance coefficient J.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
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

        # Validation
        self._validate_inputs()

        # Find J_max where KT crosses zero
        self.J_min = float(self.J_values[0])
        self.J_max = self._find_kt_zero_crossing()

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
        # Confirm curve arrays are aligned, monotonic, and non-negative.
        if len(self.J_values) != len(self.KT_values) or len(self.J_values) != len(self.KQ_values):
            raise ValueError("J_values, KT_values, and KQ_values must have the same length")

        if len(self.J_values) < 2:
            raise ValueError("At least 2 data points required for interpolation")

        # Check monotonicity of J
        if not np.all(np.diff(self.J_values) > 0):
            raise ValueError("J_values must be strictly monotonic increasing")

        # Check for reasonable coefficient ranges (KQ must be non-negative)
        if np.any(self.KQ_values < 0):
            raise ValueError("KQ values must be non-negative")
        # Note: KT can be negative (we'll find where it crosses zero)

    def _find_kt_zero_crossing(self) -> float:
        """
        Find the J value where KT crosses zero in the input data.
        
        Uses linear interpolation between data points to find the exact
        zero crossing. If KT never crosses zero, returns the maximum J
        from the input data.
        
        This is a reference value - KT can be calculated for any J value,
        but efficiency requires KT >= 0.
        
        Returns:
        --------
        float
            Reference J value where KT typically crosses zero
        """
        # Find where KT crosses zero
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

    def calculate_kt(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate thrust coefficient KT at given advance coefficient(s).

        Uses interpolation of provided data points.
        KT can be negative outside the valid operating range. J_max is a reference value
        where KT typically crosses zero, but interpolation is performed for all J values.

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]

        Returns:
        --------
        float or np.ndarray
            Thrust coefficient KT [-] (may be negative outside valid operating range)
        """
        # Interpolate KT for all J values (no range clamping)
        J = np.asarray(J)
        is_scalar = J.ndim == 0
        if is_scalar:
            J = np.array([J])
        
        # Use interpolator which handles extrapolation via fill_value
        kt = self._kt_interpolator(J)

        if is_scalar:
            return float(kt[0]) if hasattr(kt, '__len__') else float(kt)
        else:
            return kt

    def calculate_kq(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate torque coefficient KQ at given advance coefficient(s).

        Uses interpolation of provided data points.
        KQ can be negative outside the valid operating range. J_max is a reference value
        where KT typically crosses zero, but interpolation is performed for all J values.

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]

        Returns:
        --------
        float or np.ndarray
            Torque coefficient KQ [-] (may be negative outside valid operating range)
        """
        # Interpolate KQ for all J values (no range clamping)
        J = np.asarray(J)
        is_scalar = J.ndim == 0
        if is_scalar:
            J = np.array([J])
        
        # Use interpolator which handles extrapolation via fill_value
        kq = self._kq_interpolator(J)

        if is_scalar:
            return float(kq[0]) if hasattr(kq, '__len__') else float(kq)
        else:
            return kq

    def calculate_efficiency(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate propeller efficiency η₀.

        η₀ = (J / 2π) × (KT / KQ)

        Efficiency is only calculated where KT >= 0. KT and KQ may be negative outside
        the valid operating range, but efficiency requires KT >= 0 for physical validity.

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]

        Returns:
        --------
        float or np.ndarray
            Propeller efficiency η₀ [-] (zero where KT < 0)
        """
        # Derive KT/KQ at requested J values and combine into η₀.
        J = np.asarray(J)
        is_scalar = J.ndim == 0
        if is_scalar:
            J = np.array([J])
        
        # Calculate KT and KQ for all J values
        kt = np.asarray(self.calculate_kt(J))
        kq = np.asarray(self.calculate_kq(J))
        
        # Initialize result array with zeros
        eta_0 = np.zeros_like(J, dtype=float)

        # Only calculate efficiency where KT >= 0 (enforce strict constraint)
        # KQ > 1e-6 to avoid division by zero
        efficiency_mask = (kt >= 0) & (kq > 1e-6)
        eta_0[efficiency_mask] = (J[efficiency_mask] / (2 * np.pi)) * (kt[efficiency_mask] / kq[efficiency_mask])

        if is_scalar:
            return float(eta_0[0])
        else:
            return eta_0

    def max_efficiency_J(self) -> float:
        """
        Find J value that maximizes efficiency.

        Returns:
        --------
        float
            Advance coefficient at maximum efficiency [-]
        """
        # Sample the provided J domain to locate the efficiency peak.
        # Sample J range based on valid J range (up to KT zero crossing)
        J_range = np.linspace(self.J_min, self.J_max, 100)

        # Calculate efficiencies
        efficiencies = self.calculate_efficiency(J_range)

        # Find maximum
        max_idx = np.argmax(efficiencies)
        return J_range[max_idx]

    @property
    def J_range(self) -> tuple[float, float]:
        """
        Get the reference J range.
        
        Returns the range [J_min, J_max] where J_max is where KT typically crosses zero.
        This is a reference value - KT and KQ can be calculated for any J value,
        but efficiency requires KT >= 0.

        Returns:
        --------
        tuple[float, float]
            (J_min, J_max) range [-]
        """
        # Report the reference J range (KT zero crossing is a reference, not a hard limit).
        return self.J_min, self.J_max

    @property
    def max_efficiency(self) -> float:
        """
        Get maximum achievable efficiency.

        Returns:
        --------
        float
            Maximum efficiency [-]
        """
        # Compute η₀ at the optimal J for convenience.
        J_opt = self.max_efficiency_J()
        return float(self.calculate_efficiency(J_opt))
