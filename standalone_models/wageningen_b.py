import numpy as np
from typing import Union


class WageningenB:
    """
    Wageningen B-series propeller model using polynomial regression.

    Calculates thrust coefficient KT and torque coefficient KQ as functions
    of advance coefficient J, using polynomial coefficients from the
    Wageningen B-series data.

    References:
    - Oosterveld & Oossanen (1975)
    - Bernitsas et al. (1981)

    Parameters:
    -----------
    pitch_diameter_ratio : float
        Pitch-to-diameter ratio P/D [-]
    blade_area_ratio : float
        Expanded blade area ratio Ae/A0 [-]
    number_of_blades : int
        Number of blades Z [-]
    """

    # KT polynomial coefficients (39 coefficients)
    # Format: C_KT[i] corresponds to J^s[i] * (P/D)^t[i] * (Ae/A0)^u[i] * Z^v[i]
    _KT_COEFFICIENTS = np.array([
        0.00889, -0.2044, 0.0707, 0.177, -0.0758, -0.0261, -0.0738,
        0.0343, -0.0176, 0.0145, -0.0090, 0.0018, 0.0125, -0.0017,
        -0.0034, 0.0007, 0.0019, -0.0003, -0.0007, 0.0001,
        0.0085, -0.0182, 0.0076, -0.0012, 0.0018, -0.0006, 0.0003,
        0.0000, -0.0001, 0.0000, 0.0002, -0.0001, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000
    ])

    _KT_POWERS = np.array([
        [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 2, 0, 0],
        [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 2, 0], [0, 0, 1, 1], [0, 0, 0, 2],
        [3, 0, 0, 0], [2, 1, 0, 0], [2, 0, 1, 0], [2, 0, 0, 1], [1, 2, 0, 0],
        [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 2, 0], [1, 0, 1, 1], [1, 0, 0, 2],
        [0, 3, 0, 0], [0, 2, 1, 0], [0, 2, 0, 1], [0, 1, 2, 0], [0, 1, 1, 1],
        [0, 1, 0, 2], [0, 0, 3, 0], [0, 0, 2, 1], [0, 0, 1, 2], [0, 0, 0, 3],
        [4, 0, 0, 0], [3, 1, 0, 0], [3, 0, 1, 0], [3, 0, 0, 1], [2, 2, 0, 0]
    ])

    # KQ polynomial coefficients (47 coefficients)
    _KQ_COEFFICIENTS = np.array([
        0.00355, -0.0981, 0.0313, 0.0753, -0.0289, -0.0147, -0.0438,
        0.0187, -0.0117, 0.0095, -0.0054, 0.0010, 0.0079, -0.0013,
        -0.0023, 0.0005, 0.0014, -0.0002, -0.0005, 0.0001, 0.0054,
        -0.0109, 0.0043, -0.0007, 0.0011,
        -0.0004, 0.0003, 0.0000, -0.0001, 0.0000, 0.0002, -0.0001,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
    ])

    _KQ_POWERS = np.array([
        [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 2, 0, 0],
        [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 2, 0], [0, 0, 1, 1], [0, 0, 0, 2],
        [3, 0, 0, 0], [2, 1, 0, 0], [2, 0, 1, 0], [2, 0, 0, 1], [1, 2, 0, 0],
        [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 2, 0], [1, 0, 1, 1], [1, 0, 0, 2],
        [0, 3, 0, 0], [0, 2, 1, 0], [0, 2, 0, 1], [0, 1, 2, 0], [0, 1, 1, 1],
        [0, 1, 0, 2], [0, 0, 3, 0], [0, 0, 2, 1], [0, 0, 1, 2], [0, 0, 0, 3],
        [4, 0, 0, 0], [3, 1, 0, 0], [3, 0, 1, 0], [3, 0, 0, 1], [2, 2, 0, 0],
        [2, 1, 1, 0], [2, 1, 0, 1], [2, 0, 2, 0], [2, 0, 1, 1], [2, 0, 0, 2],
        [1, 3, 0, 0], [1, 2, 1, 0], [1, 2, 0, 1], [1, 1, 2, 0], [1, 1, 1, 1],
        [1, 1, 0, 2], [1, 0, 3, 0]
    ])

    def __init__(self, pitch_diameter_ratio: float, blade_area_ratio: float,
                 number_of_blades: int):
        """
        Initialize Wageningen B-series propeller.

        Parameters:
        -----------
        pitch_diameter_ratio : float
            Pitch-to-diameter ratio P/D [-]
        blade_area_ratio : float
            Expanded blade area ratio Ae/A0 [-]
        number_of_blades : int
            Number of blades Z [-]
        """
        self.pitch_diameter_ratio = pitch_diameter_ratio
        self.blade_area_ratio = blade_area_ratio
        self.number_of_blades = number_of_blades
    

        # Pre-compute power terms for efficiency
        self._pd_power = pitch_diameter_ratio
        self._ae_power = blade_area_ratio
        self._z_power = number_of_blades

    def calculate_kt(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate thrust coefficient KT at advance coefficient(s) J.

        KT = Σ C_KT[i] × J^s[i] × (P/D)^t[i] × (Ae/A0)^u[i] × Z^v[i]

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
        kt = np.zeros_like(J, dtype=float)

        for i, (s, t, u, v) in enumerate(self._KT_POWERS):
            if i < len(self._KT_COEFFICIENTS):
                kt += (self._KT_COEFFICIENTS[i] *
                       np.power(J, s) *
                       np.power(self._pd_power, t) *
                       np.power(self._ae_power, u) *
                       np.power(self._z_power, v))

        return float(kt) if J.ndim == 0 else kt

    def calculate_kq(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate torque coefficient KQ at advance coefficient(s) J.

        KQ = Σ C_KQ[i] × J^s[i] × (P/D)^t[i] × (Ae/A0)^u[i] × Z^v[i]

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
        kq = np.zeros_like(J, dtype=float)

        for i, (s, t, u, v) in enumerate(self._KQ_POWERS):
            if i < len(self._KQ_COEFFICIENTS):
                kq += (self._KQ_COEFFICIENTS[i] *
                       np.power(J, s) *
                       np.power(self._pd_power, t) *
                       np.power(self._ae_power, u) *
                       np.power(self._z_power, v))

        return float(kq) if J.ndim == 0 else kq

    def calculate_efficiency(self, J: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate open water efficiency η₀.

        η₀ = (J / 2π) × (KT / KQ)

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]

        Returns:
        --------
        float or np.ndarray
            Open water efficiency η₀ [-]
        """
        J = np.asarray(J)
        kt = self.calculate_kt(J)
        kq = self.calculate_kq(J)

        # Avoid division by zero
        eta_0 = np.zeros_like(J, dtype=float)
        nonzero_mask = kq != 0
        eta_0[nonzero_mask] = (J[nonzero_mask] / (2 * np.pi)) * (kt[nonzero_mask] / kq[nonzero_mask])

        return float(eta_0) if J.ndim == 0 else eta_0

    @property
    def max_efficiency_J(self) -> float:
        """
        Estimate J value for maximum efficiency (approximate).

        Returns:
        --------
        float
            Advance coefficient at maximum efficiency [-]
        """
        # Rough estimate based on typical Wageningen B-series behavior
        return 0.6
