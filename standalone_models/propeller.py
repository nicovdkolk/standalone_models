"""
Main propeller class supporting multiple models.

This module implements a unified Propeller class that can use different
propeller models (simple thrust, Wageningen B-series, custom curves)

and supports multiple optimization modes (speed-driven, RPM-driven, power-driven).
"""

from typing import Optional, Union, Dict, Any, Callable
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


# Simple LoadCase class for compatibility
class LoadCase:
    """Simple load case class for propeller calculations."""
    def __init__(self, speed: float):
        self.speed = speed


class Propeller:
    """
    Propeller models for converting shaft power (kW, Torque, RPM) to propeller delivered thrust
    (kN) and vice versa.

    Handles backwards and forwards-facing propeller models:
    Backwards: Propeller delivered thrust (kN) >> Shaft power (kW, Torque, RPM)
    Forward: Shaft power (kW, Torque, RPM) >> Propeller delivered thrust
    
    Supports three propeller models:
    - Thrust model (current)
    - Wageningen B-series polynomial model
    - Custom KT/KQ curves

    Supports three solving modes:
    - Speed mode (current, legacy): Given speed → find thrust and power
    - Power mode (future): Given power → find RPM and thrust at speed
    - RPM mode (future): Given RPM → find thrust and power at speed
    """

    def __init__(
        self,

        propeller_model: Optional[Union[str, object]] = None,  # propeller model

        dia_prop: float = None,  # prop diameter
        wake_fraction: float = None,  # wake fraction
        thrust_deduction: float = None,  # thrust deduction
        n_prop: float = None,  # number of propellers

        eta_hull: float = None,  # hull efficiency
        eta_rotative: float = None,  # rotative efficiency
        eta_open_water: float = None,  # open water efficiency
        eta_D: float = None,  # delivered power efficiency (legacy)
        eta_delivered_power: Union[float, dict, None] = None,  # delivered power efficiency

        pitch_diameter_ratio: float = None,  # pitch diameter ratio
        blade_area_ratio: float = None,  # blade area ratio
        number_of_blades: float = None,  # number of blades

        max_rpm: float = None,  # maximum RPM
        nominal_rpm: float = None,  # design RPM

        kt_curve: dict = None,  # KT curve
        kq_curve: dict = None,  # KQ curve

        physics: Optional[object] = None,  # Physics object for calculations

    ):

        self.propeller_model = propeller_model

        self.dia_prop = dia_prop
        self.wake_fraction = wake_fraction
        self.thrust_deduction = thrust_deduction
        self.n_prop = n_prop
        
        self.hull_efficiency = eta_hull
        self.rotative_efficiency = eta_rotative
        self.open_water_efficiency = eta_open_water
        
        # Handle eta_delivered_power: can be float, dict (for interpolation), or callable
        self.eta_delivered_power = eta_delivered_power if eta_delivered_power is not None else eta_D
        self.eta_delivered_power_interpolator = None
        
        if isinstance(self.eta_delivered_power, dict):
            self.eta_delivered_power_interpolator = interp1d(
                self.eta_delivered_power["speed"],
                self.eta_delivered_power["eta_delivered_power"],
                kind='linear',
                fill_value=(self.eta_delivered_power["eta_delivered_power"][0], self.eta_delivered_power["eta_delivered_power"][-1]),
                bounds_error=False
            )

        self.pitch_diameter_ratio = pitch_diameter_ratio
        self.blade_area_ratio = blade_area_ratio
        self.number_of_blades = number_of_blades

        self.max_rpm = max_rpm
        self.nominal_rpm = nominal_rpm

        self.kt_curve = kt_curve
        self.kq_curve = kq_curve

        # Initialize model and physics (will be set if needed)
        self.model = None
        self.physics = physics

        # Initialize propeller model if specified
        if propeller_model == "wageningen_b":
            self._init_wageningen_b()
        elif propeller_model == "custom_curves":
            self._init_custom_curves()

    def _init_wageningen_b(self):
        """Initialize Wageningen B model."""
        from .wageningen_b import WageningenB

        self.model = WageningenB(
            pitch_diameter_ratio=self.pitch_diameter_ratio,
            blade_area_ratio=self.blade_area_ratio,
            number_of_blades=self.number_of_blades
        )

    def _init_custom_curves(self):
        """Initialize custom curves model."""
        from .custom_curves import CustomCurves

        if self.kt_curve is None or self.kq_curve is None:
            raise ValueError("Custom curves model requires 'kt_curve' and 'kq_curve'")

        # Extract J, KT, KQ from curve dictionaries
        J_kt = self.kt_curve.get("J", [])
        KT = self.kt_curve.get("KT", [])
        J_kq = self.kq_curve.get("J", [])
        KQ = self.kq_curve.get("KQ", [])

        if J_kt != J_kq:
            raise ValueError("KT and KQ curves must have the same J values")

        self.model = CustomCurves(
            J_values=J_kt,
            KT_values=KT,
            KQ_values=KQ
        )

    # === Speed-Driven Mode (Current) ===

    def eta_D_at_speed(self, loadcase: LoadCase) -> float:
        """
        Get eta_D (delivered power efficiency) at a given speed.
        
        Supports:
        - float: constant efficiency
        - dict: interpolated efficiency based on speed
        - callable: function that takes speed and returns efficiency
        
        Parameters:
        -----------
        speed : float
            Ship speed [m/s]
            
        Returns:
        --------
        float
            Delivered power efficiency [-]
        """
        if isinstance(self.eta_delivered_power, (int, float)):
            return float(self.eta_delivered_power)
        elif self.eta_delivered_power_interpolator is not None:
            return float(self.eta_delivered_power_interpolator(loadcase.speed))
        else:
            raise ValueError("eta_delivered_power is not set")

    # === Helpers ===

    def _evaluate_mapping(self, value: Union[float, Callable[[LoadCase], float]], loadcase: LoadCase) -> float:
        """Evaluate scalar or callable mapping against a loadcase."""
        if callable(value):
            return float(value(loadcase))
        return float(value)

    def _wake_fraction_at_loadcase(self, loadcase: LoadCase) -> float:
        """Return wake fraction for the loadcase, defaulting to 0.0 if missing."""
        if self.wake_fraction is None:
            return 0.0
        return self._evaluate_mapping(self.wake_fraction, loadcase)

    def _thrust_deduction_at_loadcase(self, loadcase: LoadCase) -> float:
        """Return thrust deduction for the loadcase, defaulting to 0.0 if missing."""
        if self.thrust_deduction is None:
            return 0.0
        return self._evaluate_mapping(self.thrust_deduction, loadcase)


    def effective_power(self, loadcase: LoadCase, thrust: float) -> float:
        """Calculate the effective power for this loadcase.
        thrust is actually the residual in x, which can be +/- or zero
        effective power can only be greater or equal to zero
        """
        return max(0, thrust * loadcase.speed)

    def delivered_power(self, loadcase: LoadCase, effective_power: float) -> float:
        """Calculate the delivered power for this loadcase."""
        return effective_power / self.eta_D_at_speed(loadcase)


    def thrust_from_power(self, Pe: float, speed: float,
                         wake_fraction: float, thrust_deduction: float) -> Optional[float]:
        """
        Calculate thrust from delivered power, auto-selecting model priority.

        Priority (first available):
        1. Simple thrust model using η_D
        2. Wageningen B polynomial model
        3. Custom KT/KQ curves
        """
        loadcase = LoadCase(speed)
        wake_fraction = self._evaluate_mapping(wake_fraction, loadcase)
        thrust_deduction = self._evaluate_mapping(thrust_deduction, loadcase)

        # 1) Simple thrust model
        if self.eta_delivered_power is not None:
            v_a = speed * (1 - wake_fraction)
            if v_a <= 0:
                return 0.0

            eta_D_value = self.eta_D_at_speed(loadcase)
            T_gross = Pe * eta_D_value / v_a
            return T_gross * (1 - thrust_deduction)

        # 2/3) Advanced models: use RPM inverse to maintain consistency
        if self.model is None or self.physics is None:
            return None

        rpm_solution = self.rpm_from_power_speed(Pe, speed, wake_fraction, initial_rpm=self.nominal_rpm or 100.0)
        if rpm_solution is None:
            return None

        return self.thrust_from_rpm(rpm_solution, speed, wake_fraction, thrust_deduction)

    def delivered_power_from_thrust(self, thrust: float, speed: float) -> Optional[float]:
        """Unified solver: thrust (net, after thrust deduction) → delivered power."""
        loadcase = LoadCase(speed)
        wake_fraction = self._wake_fraction_at_loadcase(loadcase)
        thrust_deduction = self._thrust_deduction_at_loadcase(loadcase)

        # 1) Simple thrust model using eta_D
        if self.eta_delivered_power is not None:
            v_a = speed * (1 - wake_fraction)
            if v_a <= 0:
                return 0.0

            eta_D_value = self.eta_D_at_speed(loadcase)
            gross_thrust = thrust / (1 - thrust_deduction) if thrust_deduction < 1 else float('inf')
            return gross_thrust * v_a / eta_D_value

        # 2/3) Advanced models require physics and hydrodynamic model
        if self.model is None or self.physics is None:
            return None

        rpm_solution = self.rpm_from_thrust_speed(
            thrust=thrust,
            speed=speed,
            wake_fraction=wake_fraction,
            thrust_deduction=thrust_deduction,
            initial_rpm=self.nominal_rpm or 100.0,
        )
        if rpm_solution is None:
            return None

        return self.power_from_rpm(rpm_solution, speed, wake_fraction)

    # === RPM-Driven Mode (Future) ===

    def advance_coefficient(self, rpm: float, speed: float,
                           wake_fraction: float) -> float:
        """
        Calculate advance coefficient J = v_a / (n × D).

        Parameters:
        -----------
        rpm : float
            Propeller revolutions per minute [rev/min]
        speed : float
            Ship speed [m/s]
        wake_fraction : float
            Wake fraction w [-]

        Returns:
        --------
        float
            Advance coefficient J [-]
        """
        n_rps = rpm / 60  # Convert to rev/s
        v_a = speed * (1 - wake_fraction)
        return v_a / (n_rps * self.dia_prop) if n_rps > 0 else 0.0

    def thrust_from_rpm(self, rpm: float, speed: float,
                       wake_fraction: float, thrust_deduction: float) -> Optional[float]:
        """
        Calculate thrust from RPM using KT/KQ curves.

        Formula: T = KT(J) × ρ × n² × D⁴ / (1-t)

        Parameters:
        -----------
        rpm : float
            Propeller RPM [rev/min]
        speed : float
            Ship speed [m/s]
        wake_fraction : float
            Wake fraction w [-]
        thrust_deduction : float
            Thrust deduction t [-]

        Returns:
        --------
        Optional[float]
            Thrust [N], or None if no KT/KQ model available
        """
        if self.model is None or self.physics is None:
            return None

        J = self.advance_coefficient(rpm, speed, wake_fraction)
        KT = self.model.calculate_kt(J)
        n_rps = rpm / 60

        T_gross = KT * self.physics.rho_w * (n_rps**2) * (self.dia_prop**4)
        return T_gross / (1 - thrust_deduction)

    def torque_from_rpm(self, rpm: float, speed: float,
                       wake_fraction: float) -> Optional[float]:
        """
        Calculate torque from RPM using KT/KQ curves.

        Formula: Q = KQ(J) × ρ × n² × D⁵

        Parameters:
        -----------
        rpm : float
            Propeller RPM [rev/min]
        speed : float
            Ship speed [m/s]
        wake_fraction : float
            Wake fraction w [-]

        Returns:
        --------
        Optional[float]
            Torque [N⋅m], or None if no KT/KQ model available
        """
        if self.model is None or self.physics is None:
            return None

        J = self.advance_coefficient(rpm, speed, wake_fraction)
        KQ = self.model.calculate_kq(J)
        n_rps = rpm / 60

        return KQ * self.physics.rho_w * (n_rps**2) * (self.dia_prop**5)

    def power_from_rpm(self, rpm: float, speed: float,
                      wake_fraction: float) -> Optional[float]:
        """
        Calculate delivered power from RPM.

        P_delivered = Q × ω = Q × 2π × n

        Parameters:
        -----------
        rpm : float
            Propeller RPM [rev/min]
        speed : float
            Ship speed [m/s]
        wake_fraction : float
            Wake fraction w [-]

        Returns:
        --------
        Optional[float]
            Delivered power [W], or None if no KT/KQ model available
        """
        Q = self.torque_from_rpm(rpm, speed, wake_fraction)
        if Q is None:
            return None

        n_rps = rpm / 60
        return Q * 2 * np.pi * n_rps

    # === Inverse Methods (for optimization) ===

    def rpm_from_thrust_speed(self, thrust: float, speed: float,
                             wake_fraction: float, thrust_deduction: float,
                             initial_rpm: float = 100.0) -> Optional[float]:
        """
        Calculate RPM needed to produce given thrust at given speed.

        Solves: thrust = thrust_from_rpm(rpm, speed, w, t) for rpm

        Parameters:
        -----------
        thrust : float
            Required thrust [N]
        speed : float
            Ship speed [m/s]
        wake_fraction : float
            Wake fraction w [-]
        thrust_deduction : float
            Thrust deduction t [-]
        initial_rpm : float, default 100.0
            Initial RPM guess [rev/min]

        Returns:
        --------
        Optional[float]
            RPM [rev/min], or None if no KT/KQ model available
        """
        if self.model is None or self.physics is None:
            return None

        def residual(rpm):
            T_calc = self.thrust_from_rpm(rpm[0], speed, wake_fraction, thrust_deduction)
            return T_calc - thrust if T_calc is not None else float('inf')

        try:
            result = fsolve(residual, [initial_rpm], xtol=1e-6, maxfev=100)
            rpm_solution = result[0]

            # Validate solution is reasonable
            if rpm_solution > 0 and rpm_solution < (self.max_rpm or 1000):
                return rpm_solution
            else:
                return None
        except:
            return None

    def rpm_from_power_speed(self, power: float, speed: float,
                            wake_fraction: float, initial_rpm: float = 100.0) -> Optional[float]:
        """
        Calculate RPM needed to absorb given power at given speed.

        Solves: power = power_from_rpm(rpm, speed, w) for rpm

        Parameters:
        -----------
        power : float
            Delivered power [W]
        speed : float
            Ship speed [m/s]
        wake_fraction : float
            Wake fraction w [-]
        initial_rpm : float, default 100.0
            Initial RPM guess [rev/min]

        Returns:
        --------
        Optional[float]
            RPM [rev/min], or None if no KT/KQ model available
        """
        if self.model is None or self.physics is None:
            return None

        def residual(rpm):
            P_calc = self.power_from_rpm(rpm[0], speed, wake_fraction)
            return P_calc - power if P_calc is not None else float('inf')

        try:
            result = fsolve(residual, [initial_rpm], xtol=1e-6, maxfev=100)
            rpm_solution = result[0]

            # Validate solution
            if rpm_solution > 0 and rpm_solution < (self.max_rpm or 1000):
                return rpm_solution
            else:
                return None
        except:
            return None

    # === Utility Methods ===

    @property
    def has_model(self) -> bool:
        """Check if advanced propeller model is available."""
        return self.model is not None

    def get_optimal_J(self) -> Optional[float]:
        """
        Get J value for optimal efficiency.

        Returns:
        --------
        Optional[float]
            Optimal advance coefficient, or None if no model
        """
        if self.model is None:
            return None

        if hasattr(self.model, 'max_efficiency_J'):
            return self.model.max_efficiency_J()
        else:
            return None
