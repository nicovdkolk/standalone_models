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

    # KT polynomial coefficients (39 coefficients) reproduced from Table 1
    # (Bernitsas et al., Rn = 2 × 10^6).
    # Format: C_KT[i] corresponds to J^s[i] * (P/D)^t[i] * (Ae/A0)^u[i] * Z^v[i]
    _KT_COEFFICIENTS = np.array(
        [
            0.00880496,
            -0.204554,
            0.166351,
            0.158114,
            -0.147581,
            -0.481497,
            0.415437,
            0.0144043,
            -0.0530054,
            0.0143481,
            0.0606826,
            -0.0125894,
            0.0109689,
            -0.133698,
            0.00638407,
            -0.00132718,
            0.168496,
            -0.0507214,
            0.0854559,
            -0.0504475,
            0.010465,
            -0.00648272,
            -0.00841728,
            0.0168424,
            -0.00102296,
            -0.0317791,
            0.018604,
            -0.00410798,
            -0.000606848,
            -0.0049819,
            0.0025983,
            -0.000560528,
            -0.00163652,
            -0.000328787,
            0.000116502,
            0.000690904,
            0.00421749,
            0.0000565229,
            -0.00146564,
        ]
    )

    _KT_POWERS = np.array(
        [
            # s, t, u, v
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 2, 0, 0),
            (2, 0, 1, 0),
            (1, 1, 1, 0),
            (0, 2, 1, 0),
            (0, 0, 0, 1),
            (2, 0, 0, 1),
            (0, 1, 0, 1),
            (1, 1, 0, 1),
            (0, 0, 1, 1),
            (1, 0, 1, 1),
            (0, 3, 0, 0),
            (0, 6, 0, 0),
            (2, 6, 0, 0),
            (3, 0, 1, 0),
            (0, 0, 2, 0),
            (2, 0, 2, 0),
            (3, 0, 2, 0),
            (1, 6, 2, 0),
            (2, 6, 2, 0),
            (0, 3, 0, 1),
            (1, 3, 0, 1),
            (3, 3, 0, 1),
            (0, 3, 1, 1),
            (1, 0, 2, 1),
            (0, 2, 2, 1),
            (0, 0, 0, 2),
            (1, 0, 0, 2),
            (2, 0, 0, 2),
            (3, 0, 0, 2),
            (1, 2, 1, 2),
            (1, 6, 1, 2),
            (2, 6, 1, 2),
            (0, 0, 1, 2),
            (0, 3, 1, 2),
            (3, 6, 2, 2),
            (0, 3, 2, 2),
        ],
        dtype=int,
    )

    # KQ polynomial coefficients (47 coefficients) reproduced from Table 1
    # (Bernitsas et al., Rn = 2 × 10^6).
    _KQ_COEFFICIENTS = np.array(
        [
            0.00379368,
            0.00886523,
            -0.032241,
            0.00344778,
            -0.0408811,
            -0.108009,
            -0.0885381,
            0.188561,
            -0.00370871,
            0.00513696,
            0.0209449,
            0.00474319,
            -0.00723408,
            0.00438388,
            -0.0269403,
            0.0558082,
            0.0161886,
            0.00318086,
            0.015896,
            0.0471729,
            0.0196283,
            -0.0502782,
            -0.030055,
            0.0417122,
            -0.0397722,
            -0.00350024,
            -0.0106854,
            0.00110903,
            -0.000313912,
            0.0035985,
            -0.00142121,
            -0.00383637,
            0.0126803,
            -0.00318278,
            0.00334268,
            -0.00183491,
            -0.000112451,
            -0.0000297228,
            0.000269551,
            0.00083265,
            0.00155334,
            0.000302683,
            -0.0001843,
            -0.000425399,
            0.000869243,
            -0.0004659,
            0.0000554194,
        ]
    )

    _KQ_POWERS = np.array(
        [
            # s, t, u, v
            (0, 0, 0, 0),
            (2, 0, 0, 0),
            (1, 1, 0, 0),
            (0, 2, 0, 0),
            (0, 1, 1, 0),
            (1, 1, 1, 0),
            (2, 1, 1, 0),
            (0, 2, 1, 0),
            (1, 0, 0, 1),
            (0, 1, 0, 1),
            (1, 1, 0, 1),
            (2, 1, 0, 1),
            (2, 0, 1, 1),
            (1, 1, 1, 1),
            (0, 2, 1, 1),
            (3, 0, 1, 0),
            (0, 3, 1, 0),
            (1, 3, 1, 0),
            (0, 0, 2, 0),
            (1, 0, 2, 0),
            (3, 0, 2, 0),
            (0, 1, 2, 0),
            (3, 1, 2, 0),
            (2, 2, 2, 0),
            (0, 3, 2, 0),
            (0, 6, 2, 0),
            (3, 0, 0, 1),
            (3, 3, 0, 1),
            (0, 6, 0, 1),
            (3, 0, 1, 1),
            (0, 6, 1, 1),
            (1, 0, 2, 1),
            (0, 2, 2, 1),
            (2, 3, 2, 1),
            (0, 6, 2, 1),
            (1, 1, 0, 2),
            (1, 2, 0, 2),
            (1, 6, 0, 2),
            (0, 0, 1, 2),
            (1, 1, 1, 2),
            (2, 1, 1, 2),
            (2, 2, 1, 2),
            (0, 0, 2, 2),
            (0, 3, 2, 2),
            (3, 3, 2, 2),
            (0, 6, 2, 2),
            (1, 6, 2, 2),
        ],
        dtype=int,
    )

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

    @staticmethod
    def _log_reynolds_term(reynolds_number: float) -> float:
        """Return log10(Re) - log10(2) for Reynolds corrections."""

        return np.log10(reynolds_number) - 0.301

    def _delta_kt(self, J: np.ndarray, reynolds_number: float) -> np.ndarray:
        """Compute ΔKT correction for Reynolds numbers above 2×10^6."""

        log_r = self._log_reynolds_term(reynolds_number)

        return (
            0.000353485
            - 0.0033758 * log_r * np.power(J, 2)
            + 0.00112016 * np.power(self._pd_power, 2)
            + 0.00255833 * self._ae_power
            - 0.0027623 * self._z_power
            + 0.000061636 * np.power(log_r, 2) * np.power(self._pd_power, 2)
            + 0.000148383 * log_r * np.power(self._pd_power, 2)
            + 0.000364374 * np.power(log_r, 2) * self._ae_power
            + 0.0000749837 * log_r * self._ae_power
            + 0.000422515 * np.power(log_r, 2) * self._z_power
            + 0.000100988 * log_r * self._z_power
        )

    def _delta_kq(self, reynolds_number: float) -> float:
        """Compute ΔKQ correction for Reynolds numbers above 2×10^6."""

        log_r = self._log_reynolds_term(reynolds_number)

        return (
            -0.0005914124
            + 0.00696898 * self._pd_power
            - 0.0008845 * np.power(self._pd_power, 2)
            - 0.00015027 * np.power(self._pd_power, 6)
            - 0.0005593 * log_r
            + 0.00011727 * np.power(log_r, 2)
            + 0.000594341 * np.power(log_r, 3)
            - 0.0000220915 * np.power(log_r, 2) * np.power(self._pd_power, 2)
            + 0.0000220915 * np.power(log_r, 4) * np.power(self._pd_power, 2)
            + 0.00000220915 * np.power(log_r, 3) * np.power(self._pd_power, 3)
            - 0.000220915 * np.power(log_r, 2) * self._ae_power
            + 0.0000220915 * np.power(log_r, 3) * np.power(self._ae_power, 2)
        )

    def calculate_kt(
        self,
        J: Union[float, np.ndarray],
        reynolds_number: Union[None, float] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate thrust coefficient KT at advance coefficient(s) J.

        KT = Σ C_KT[i] × J^s[i] × (P/D)^t[i] × (Ae/A0)^u[i] × Z^v[i]

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]
        reynolds_number : float, optional
            Reynolds number. If provided and greater than 2×10^6, a ΔKT correction
            from Table 2 is added.

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

        if reynolds_number is not None and reynolds_number > 2_000_000:
            kt += self._delta_kt(J, reynolds_number)

        # Ensure non-negative results (Kt must be positive)
        kt = np.maximum(kt, 0.0)

        return float(kt) if J.ndim == 0 else kt

    def calculate_kq(
        self,
        J: Union[float, np.ndarray],
        reynolds_number: Union[None, float] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate torque coefficient KQ at advance coefficient(s) J.

        KQ = Σ C_KQ[i] × J^s[i] × (P/D)^t[i] × (Ae/A0)^u[i] × Z^v[i]

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]
        reynolds_number : float, optional
            Reynolds number. If provided and greater than 2×10^6, a ΔKQ correction
            from Table 2 is added.

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

        if reynolds_number is not None and reynolds_number > 2_000_000:
            kq += self._delta_kq(reynolds_number)

        # Ensure non-negative results (Kq must be positive)
        kq = np.maximum(kq, 0.0)

        return float(kq) if J.ndim == 0 else kq

    def calculate_efficiency(
        self,
        J: Union[float, np.ndarray],
        reynolds_number: Union[None, float] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate open water efficiency η₀.

        η₀ = (J / 2π) × (KT / KQ)

        Parameters:
        -----------
        J : float or np.ndarray
            Advance coefficient(s) [-]
        reynolds_number : float, optional
            Reynolds number. If provided and greater than 2×10^6, ΔKT and ΔKQ
            corrections from Table 2 are applied.

        Returns:
        --------
        float or np.ndarray
            Open water efficiency η₀ [-]
        """
        J = np.asarray(J)
        kt = self.calculate_kt(J, reynolds_number=reynolds_number)
        kq = self.calculate_kq(J, reynolds_number=reynolds_number)

        # Ensure kt and kq are arrays for indexing
        kt = np.asarray(kt)
        kq = np.asarray(kq)

        # Avoid division by zero and very small numbers
        # Only calculate efficiency where both Kt and Kq are positive
        eta_0 = np.zeros_like(J, dtype=float)
        valid_mask = (kt > 0) & (kq > 1e-6)  # Both Kt and Kq must be positive
        eta_0[valid_mask] = (J[valid_mask] / (2 * np.pi)) * (kt[valid_mask] / kq[valid_mask])

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
