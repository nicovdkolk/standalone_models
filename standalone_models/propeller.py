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
        # Store ship speed for reuse in solver routines.
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

        # Instantiate polynomial model with provided geometric ratios.
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

        # Build CustomCurves interpolator using the aligned KT/KQ tables.
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
        # Evaluate delivered power efficiency via scalar, lookup table, or callable.
        if isinstance(self.eta_delivered_power, (int, float)):
            return float(self.eta_delivered_power)
        elif self.eta_delivered_power_interpolator is not None:
            return float(self.eta_delivered_power_interpolator(loadcase.speed))
        else:
            raise ValueError("eta_delivered_power is not set")

    # === Helpers ===

    def _evaluate_mapping(self, value: Union[float, Callable[[LoadCase], float]], loadcase: LoadCase) -> float:
        """Evaluate scalar or callable mapping against a loadcase."""
        # Call provided mapping when necessary to resolve speed-dependent inputs.
        if callable(value):
            return float(value(loadcase))
        return float(value)

    def _wake_fraction_at_loadcase(self, loadcase: LoadCase) -> float:
        """Return wake fraction for the loadcase, defaulting to 0.0 if missing."""
        # Use default wake fraction when none is supplied.
        if self.wake_fraction is None:
            return 0.0
        return self._evaluate_mapping(self.wake_fraction, loadcase)

    def _thrust_deduction_at_loadcase(self, loadcase: LoadCase) -> float:
        """Return thrust deduction for the loadcase, defaulting to 0.0 if missing."""
        # Use default thrust deduction when none is supplied.
        if self.thrust_deduction is None:
            return 0.0
        return self._evaluate_mapping(self.thrust_deduction, loadcase)

    def hull_efficiency_at_loadcase(self, loadcase: LoadCase) -> float:
        """
        Calculate hull efficiency η_H from wake fraction and thrust deduction.
        
        η_H = (1 - t) / (1 - w)
        
        where:
        - t = thrust deduction factor
        - w = wake fraction
        
        This expresses how hull–propeller interaction modifies the mapping between
        thrust power P_T and effective power P_E. Hull efficiency accounts for:
        - Wake fraction w: reduces inflow speed V_A = V(1-w)
        - Thrust deduction factor t: relates gross thrust T to net resistance R = (1-t)T
        
        The relationship: P_E = η_H · P_T = η_H · T · V_A
        
        Parameters:
        -----------
        loadcase : LoadCase
            Load case with ship speed [m/s] (for speed-dependent wake/thrust deduction)
            
        Returns:
        --------
        float
            Hull efficiency η_H [-] (dimensionless, typically 0.9-1.1)
            
        Raises:
        -------
        ValueError
            If wake_fraction >= 1.0 (invalid, would cause division by zero or negative)
        """
        # Priority: explicit value > calculated value > default
        # If hull efficiency is explicitly set, use that (allows manual override)
        if self.hull_efficiency is not None:
            return float(self.hull_efficiency)
        
        # Calculate from wake fraction and thrust deduction
        wake_fraction = self._wake_fraction_at_loadcase(loadcase)
        thrust_deduction = self._thrust_deduction_at_loadcase(loadcase)
        
        # Handle default case: if both are None/zero, no interaction, efficiency is 1.0
        if self.wake_fraction is None and self.thrust_deduction is None:
            return 1.0
        
        # Validate wake fraction (must be < 1.0 to avoid division by zero or negative values)
        if wake_fraction >= 1.0:
            raise ValueError(
                f"Invalid wake_fraction value: {wake_fraction}. "
                "Wake fraction must be < 1.0 to calculate hull efficiency. "
                "Values >= 1.0 would result in non-positive inflow speed V_A."
            )
        
        # Calculate hull efficiency: η_H = (1 - t) / (1 - w)
        hull_efficiency = (1 - thrust_deduction) / (1 - wake_fraction)
        
        return float(hull_efficiency)

    def effective_power(self, loadcase: LoadCase, thrust: float) -> float:
        """
        Calculate the effective power P_E for this loadcase.
        
        P_E = R · V
        
        where:
        - R = total resistance (net thrust after thrust deduction, i.e., R = (1-t)T)
        - V = ship speed through water
        
        Parameters:
        -----------
        loadcase : LoadCase
            Load case with ship speed V [m/s]
        thrust : float
            Net thrust (residual force in x-direction) [kN]
            This represents the total resistance R after accounting for thrust deduction.
            Can be positive (forward) or zero (zero thrust). May be negative (regeneration) in the future.
            
        Returns:
        --------
        float
            Effective power P_E [kW]. Always >= 0 (negative thrust results in 0 power).
        """
        # P_E = R · V, where R is net thrust (resistance) and V is ship speed
        # Translate net thrust demand into effective power while preventing negative outputs.
        return max(0, thrust * loadcase.speed)

    def thrust_power(self, thrust: float, speed: float, wake_fraction: float) -> float:
        """
        Calculate thrust power P_T from gross thrust and inflow speed.
        
        P_T = T · V_A
        
        where:
        - T = propeller thrust (gross thrust, before thrust deduction)
        - V_A = inflow speed to propeller disc = V(1-w)
        - V = ship speed through water
        - w = wake fraction
        
        Parameters:
        -----------
        thrust : float
            Gross propeller thrust T [kN] (before thrust deduction)
        speed : float
            Ship speed V [m/s]
        wake_fraction : float
            Wake fraction w [-]
            
        Returns:
        --------
        float
            Thrust power P_T [kW]
        """
        # V_A = V(1-w) - inflow speed reduced by wake
        v_a = speed * (1 - wake_fraction)
        # P_T = T · V_A - rate at which propeller thrust does work on the flow
        return thrust * v_a

    def delivered_power(self, loadcase: LoadCase, effective_power: float) -> float:
        """
        Calculate the delivered power P_D for this loadcase.
        
        P_D = P_E / η_D (where η_D is quasi-propulsive coefficient)
        
        Delivered power P_D is the power delivered to the propeller at the propeller plane,
        after mechanical losses in transmission (P_D = η_Tr · P_B).
        
        Parameters:
        -----------
        loadcase : LoadCase
            Load case with ship speed [m/s]
        effective_power : float
            Effective power P_E [kW]
            
        Returns:
        --------
        float
            Delivered power P_D [kW]
        """
        # Simple efficiency model: P_D = P_E / η_D
        if self.eta_delivered_power is not None:
            return effective_power / self.eta_D_at_speed(loadcase)
        
        # Model-based calculation: work backwards from effective power to thrust, then to delivered power
        if self.model is None or self.physics is None:
            raise ValueError("eta_delivered_power is not set and no model available")
        
        # Handle zero speed or zero effective power
        if loadcase.speed <= 0 or effective_power <= 0:
            return 0.0
        
        net_thrust = effective_power / loadcase.speed
        
        # Use delivered_power_from_thrust to calculate delivered power using the model
        delivered_power_kw = self.delivered_power_from_thrust(net_thrust, loadcase.speed)
        if delivered_power_kw is None:
            raise ValueError("eta_delivered_power is not set and model calculation failed")

        return delivered_power_kw


    def thrust_from_power(self, Pe: float, speed: float,
                         wake_fraction: float, thrust_deduction: float) -> Optional[float]:
        """
        Calculate thrust from delivered power, auto-selecting model priority.

        Priority (first available):
        1. Simple thrust model using η_D
        2. Wageningen B polynomial model
        3. Custom KT/KQ curves
        """
        # Route delivered power through the best available model to estimate thrust.
        loadcase = LoadCase(speed)
        wake_fraction = self._evaluate_mapping(wake_fraction, loadcase)
        thrust_deduction = self._evaluate_mapping(thrust_deduction, loadcase)

        # 1) Simple thrust model
        if self.eta_delivered_power is not None:
            # V_A = V(1-w) - inflow speed reduced by wake
            v_a = speed * (1 - wake_fraction)
            if v_a <= 0:
                return 0.0

            eta_D_value = self.eta_D_at_speed(loadcase)
            # From P_D and η_D, calculate gross thrust: T_gross = P_D · η_D / V_A
            T_gross = Pe * eta_D_value / v_a
            # Return net thrust: R = (1-t)T (after thrust deduction)
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
        # Select the appropriate model path to back-calculate delivered power from thrust.
        loadcase = LoadCase(speed)
        wake_fraction = self._wake_fraction_at_loadcase(loadcase)
        thrust_deduction = self._thrust_deduction_at_loadcase(loadcase)

        # 1) Simple thrust model using eta_D
        if self.eta_delivered_power is not None:
            # V_A = V(1-w) - inflow speed reduced by wake
            v_a = speed * (1 - wake_fraction)
            if v_a <= 0:
                return 0.0

            eta_D_value = self.eta_D_at_speed(loadcase)
            # Convert net thrust (R) to gross thrust: T = R/(1-t)
            gross_thrust = thrust / (1 - thrust_deduction) if thrust_deduction < 1 else float('inf')
            # P_D = P_T / η_D = (T · V_A) / η_D
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
        Calculate advance coefficient J = V_A / (n × D).

        J = V_A / (nD)
        
        where:
        - V_A = inflow speed = V(1-w)
        - n = propeller rotational speed [rev/s]
        - D = propeller diameter [m]

        Parameters:
        -----------
        rpm : float
            Propeller revolutions per minute [rev/min]
        speed : float
            Ship speed V [m/s]
        wake_fraction : float
            Wake fraction w [-]

        Returns:
        --------
        float
            Advance coefficient J [-]
        """
        # Translate RPM and inflow speed into the non-dimensional advance coefficient.
        n_rps = rpm / 60  # Convert to rev/s
        # V_A = V(1-w) - inflow speed reduced by wake
        v_a = speed * (1 - wake_fraction)
        # J = V_A / (nD) - non-dimensional advance ratio
        return v_a / (n_rps * self.dia_prop) if n_rps > 0 else 0.0

    def thrust_from_rpm(self, rpm: float, speed: float,
                       wake_fraction: float, thrust_deduction: float) -> Optional[float]:
        """
        Calculate thrust from RPM using KT/KQ curves.

        Formula: T = KT(J) × ρ × n² × D⁴ / (1-t)

        Note: RPM-based calculations are only available when eta_delivered_power is None.
        Models with eta_delivered_power set (float or dict) use simple efficiency model
        and cannot perform RPM-based calculations.

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
            Thrust [kN], or None if eta_delivered_power is set or no KT/KQ model available
        """
        if self.eta_delivered_power is not None:
            return None  # RPM calculations not available when using eta_D efficiency model
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

        Note: RPM-based calculations are only available when eta_delivered_power is None.
        Models with eta_delivered_power set (float or dict) use simple efficiency model
        and cannot perform RPM-based calculations.

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
            Torque [kN⋅m], or None if eta_delivered_power is set or no KT/KQ model available
        """
        if self.eta_delivered_power is not None:
            return None  # RPM calculations not available when using eta_D efficiency model
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

        Note: RPM-based calculations are only available when eta_delivered_power is None.
        Models with eta_delivered_power set (float or dict) use simple efficiency model
        and cannot perform RPM-based calculations.

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
            Delivered power [kW], or None if eta_delivered_power is set or no KT/KQ model available
        """
        if self.eta_delivered_power is not None:
            return None  # RPM calculations not available when using eta_D efficiency model
        Q = self.torque_from_rpm(rpm, speed, wake_fraction)
        if Q is None:
            return None

        n_rps = rpm / 60
        # P_D = 2πnQ - delivered power from torque and rotational speed
        # Convert torque at the shaft into delivered mechanical power.
        return Q * 2 * np.pi * n_rps

    # === Inverse Methods (for optimization) ===

    def rpm_from_thrust_speed(self, thrust: float, speed: float,
                             wake_fraction: float, thrust_deduction: float,
                             initial_rpm: float = 100.0) -> Optional[float]:
        """
        Calculate RPM needed to produce given thrust at given speed.

        Solves: thrust = thrust_from_rpm(rpm, speed, w, t) for rpm

        Note: RPM-based calculations are only available when eta_delivered_power is None.
        Models with eta_delivered_power set (float or dict) use simple efficiency model
        and cannot perform RPM-based calculations.

        Parameters:
        -----------
        thrust : float
            Required thrust [kN]
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
            RPM [rev/min], or None if eta_delivered_power is set or no KT/KQ model available
        """
        if self.eta_delivered_power is not None:
            return None  # RPM calculations not available when using eta_D efficiency model
        if self.model is None or self.physics is None:
            return None

        # Solve for RPM that satisfies the target thrust using KT curves.
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

        Note: RPM-based calculations are only available when eta_delivered_power is None.
        Models with eta_delivered_power set (float or dict) use simple efficiency model
        and cannot perform RPM-based calculations.

        Parameters:
        -----------
        power : float
            Delivered power [kW]
        speed : float
            Ship speed [m/s]
        wake_fraction : float
            Wake fraction w [-]
        initial_rpm : float, default 100.0
            Initial RPM guess [rev/min]

        Returns:
        --------
        Optional[float]
            RPM [rev/min], or None if eta_delivered_power is set or no KT/KQ model available
        """
        if self.eta_delivered_power is not None:
            return None  # RPM calculations not available when using eta_D efficiency model
        if self.model is None or self.physics is None:
            return None

        # Solve for RPM that absorbs the requested delivered power.
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
        # Indicate whether KT/KQ model support is configured.
        return self.model is not None

    def get_optimal_J(self) -> Optional[float]:
        """
        Get J value for optimal efficiency.

        Returns:
        --------
        Optional[float]
            Optimal advance coefficient, or None if no model
        """
        # Defer to the model to report its preferred operating advance coefficient.
        if self.model is None:
            return None

        if hasattr(self.model, 'max_efficiency_J'):
            # Handle both property and method cases
            max_eff_J = self.model.max_efficiency_J
            if callable(max_eff_J):
                return max_eff_J()
            else:
                return max_eff_J
        else:
            return None
