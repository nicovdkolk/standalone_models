<!-- a2d7f67d-6016-4173-80e9-d1d8775ceae3 93725bb2-9c95-4f19-9b4d-7e4e2d0e6141 -->
# Wageningen B-Series Propeller Model Implementation

## Overview

Develop a modular Propeller class supporting multiple propeller models and designed for future optimization modes:

- Simple thrust model (default fallback, current speed-driven mode)
- Wageningen B-series polynomial model (optional, enables RPM mode)
- User-defined Kt/Kq curves (optional, enables RPM mode)

CLI-based development in Shipyard core only. API inputs and GUI integration deferred.

## Architecture for Future Optimization Modes

Propulsion optimization can operate in different modes depending on what variables are controlled/solved. These modes are independent of powering architecture (conventional, DE, PTI/PTO).

### Mode 1: Speed-Driven (Current)

```
Given: speed → Solve: resistance → thrust_needed → power_required
Propeller: thrust = Pe × η_D / speed (simple model)
Used by: Current optimizer, all ship types
```

### Mode 2: RPM-Driven (Future)

```
Given: rpm (or rpm range) → Solve: thrust(rpm, speed) = resistance(speed) for speed
Requires implicit solve because: J = v_a/(n×D) couples rpm and speed
Propeller provides: thrust_from_rpm(rpm, speed) for evaluator
Optimizer finds equilibrium speed
Used by: Ships with Kt/Kq data, any powering architecture
```

### Mode 3: Power-Driven (Future)

```
Given: power_available → Solve: rpm from power → thrust(rpm, speed) = resistance(speed)
Power constraint: P = Q(rpm, speed) × 2π × n
Optimizer finds feasible rpm and speed within power limit
Used by: Any ship with power constraints (MCR limits, available gensets, etc.)
```

Note: Modes 2 and 3 apply to any powering architecture. DE ships may use power-driven mode more commonly, but conventional ships can also optimize with RPM or power constraints.

### Key Propeller Class Design Principles

1. **Evaluation Methods (not solvers):** Propeller class evaluates characteristics at given rpm/speed/power, does NOT solve implicit equations
2. **Bidirectional Support:** Methods support both forward (rpm→thrust) and inverse (thrust→rpm) for optimizer use
3. **J-coupling Awareness:** All methods accept both rpm AND speed because J couples them
4. **Fallback Chain:** Wageningen B / Custom Curves → Simple model fallback if insufficient data
5. **Architecture Agnostic:** Propeller model works with any powering architecture (conventional, DE, PTI/PTO)

## Implementation Steps

### 1. Create Wageningen B Helper Module

**File:** `Shipyard/src/core/forces/propeller_models/__init__.py` (new)

Empty init file for propeller_models package.

**File:** `Shipyard/src/core/forces/propeller_models/wageningen_b.py` (new)

Python translation of MATLAB Wageningen B-series code with polynomial coefficient evaluation:

```python
class WageningenB:
    """
    Wageningen B-series propeller model using polynomial regression.
    
    References:
    - Oosterveld & Oossanen (1975)
    - Bernitsas et al. (1981)
    """
    
    # Class constants: coefficient arrays (C_KT, s_KT, t_KT, u_KT, v_KT)
    # ... (39 coefficients from MATLAB code for KT)
    # ... (47 coefficients from MATLAB code for KQ)
    
    def __init__(self, P_D: float, Ae_A0: float, z: int, diameter: float):
        """
        Initialize Wageningen B-series propeller.
        
        Parameters:
        - P_D: Pitch-to-diameter ratio [-]
        - Ae_A0: Expanded blade area ratio [-]
        - z: Number of blades [-]
        - diameter: Propeller diameter [m]
        """
        
    def calculate_kt(self, J: float | np.ndarray) -> float | np.ndarray:
        """Calculate thrust coefficient KT at advance coefficient(s) J."""
        # KT = Σ C_KT[i] × J^s[i] × (P/D)^t[i] × (Ae/A0)^u[i] × z^v[i]
        
    def calculate_kq(self, J: float | np.ndarray) -> float | np.ndarray:
        """Calculate torque coefficient KQ at advance coefficient(s) J."""
        # KQ = Σ C_KQ[i] × J^s[i] × (P/D)^t[i] × (Ae/A0)^u[i] × z^v[i]
        
    def calculate_efficiency(self, J: float | np.ndarray) -> float | np.ndarray:
        """Calculate open water efficiency η₀ = (J / 2π) × (KT / KQ)."""
```

Use numpy for vectorized evaluation (supports J as scalar or array).

### 2. Create Custom Curve Helper Module

**File:** `Shipyard/src/core/forces/propeller_models/custom_curves.py` (new)

Handler for user-provided Kt/Kq curves with validation and interpolation:

```python
class CustomCurves:
    """
    Custom propeller curves from user-provided data.
    Interpolates KT(J) and KQ(J) from tabulated data.
    """
    
    def __init__(self, J_values: list[float], KT_values: list[float], 
                 KQ_values: list[float], diameter: float):
        """
        Initialize custom propeller curves.
        
        Parameters:
        - J_values: Advance coefficient values (must be monotonic) [-]
        - KT_values: Thrust coefficient values [-]
        - KQ_values: Torque coefficient values [-]
        - diameter: Propeller diameter [m]
        
        Raises ValueError if arrays mismatched or J not monotonic.
        """
        # Validation
        # Create scipy interpolators (linear or cubic)
        
    def calculate_kt(self, J: float | np.ndarray) -> float | np.ndarray:
        """Interpolate KT at given J value(s)."""
        
    def calculate_kq(self, J: float | np.ndarray) -> float | np.ndarray:
        """Interpolate KQ at given J value(s)."""
        
    def calculate_efficiency(self, J: float | np.ndarray) -> float | np.ndarray:
        """Calculate η₀ from interpolated KT and KQ."""
```

### 3. Create Base Propeller Class

**File:** `Shipyard/src/core/forces/propeller.py` (new)

Main Propeller class with model selection and methods supporting future optimization modes:

```python
class Propeller:
    """
    Propeller model with support for simple thrust, Wageningen B-series,
    and custom Kt/Kq curves. Designed for speed-driven, RPM-driven, and
    power-driven optimization modes.
    
    Works with any powering architecture (conventional, DE, PTI/PTO).
    """
    
    def __init__(self, config: dict, physics: Physics = None):
        """
        Initialize propeller from configuration dict.
        
        Config keys:
        - propeller_model: "wageningen_b", "custom_curves", or None (simple)
        - dia_prop: diameter [m]
        - n_prop: number of propellers [-]
        - eta_D: propeller efficiency for simple model [-]
        
        Wageningen B config:
        - pitch_diameter_ratio: P/D [-]
        - blade_area_ratio: Ae/A0 [-]
        - number_of_blades: z [-]
        
        Custom curves config:
        - kt_curve: {"J": [...], "KT": [...]}
        - kq_curve: {"J": [...], "KQ": [...]}
        
        Future optimization modes:
        - max_rpm: maximum RPM [rev/min]
        - nominal_rpm: design RPM [rev/min]
        """
        self.physics = physics
        self.diameter = config.get("dia_prop")
        self.n_prop = config.get("n_prop", 1)
        self.eta_D = config.get("eta_D")
        
        # Model selection
        model_type = config.get("propeller_model")
        if model_type == "wageningen_b":
            self.model = WageningenB(...)
        elif model_type == "custom_curves":
            self.model = CustomCurves(...)
        else:
            self.model = None  # Simple model
    
    # === Speed-Driven Mode (Current) ===
    
    def thrust_from_power(self, Pe: float, speed: float, 
                         wake_fraction: float, thrust_deduction: float) -> float:
        """
        Calculate thrust from delivered power (current simple model).
        Used in speed-driven mode.
        
        Simple model: T = Pe × η_D / (v_a × (1-t))
        Falls back to this if no Kt/Kq model available.
        
        Parameters:
        - Pe: Delivered power [W]
        - speed: Ship speed [m/s]
        - wake_fraction: w [-]
        - thrust_deduction: t [-]
        
        Returns: Thrust [N]
        """
    
    # === RPM-Driven Mode (Future) ===
    
    def advance_coefficient(self, rpm: float, speed: float, 
                           wake_fraction: float) -> float:
        """
        Calculate advance coefficient J = v_a / (n × D).
        
        Parameters:
        - rpm: Propeller revolutions per minute [rev/min]
        - speed: Ship speed [m/s]
        - wake_fraction: w [-]
        
        Returns: J [-]
        """
        n_rps = rpm / 60  # Convert to rev/s
        v_a = speed * (1 - wake_fraction)
        return v_a / (n_rps * self.diameter) if n_rps > 0 else 0
    
    def thrust_from_rpm(self, rpm: float, speed: float, 
                       wake_fraction: float, thrust_deduction: float) -> float:
        """
        Calculate thrust from RPM using Kt/Kq curves.
        Used in RPM-driven mode for evaluating thrust at a given rpm and speed.
        
        Formula: T = KT(J) × ρ × n² × D⁴ / (1-t)
        Falls back to thrust_from_power if no Kt/Kq model.
        
        Parameters:
        - rpm: Propeller RPM [rev/min]
        - speed: Ship speed [m/s]
        - wake_fraction: w [-]
        - thrust_deduction: t [-]
        
        Returns: Thrust [N]
        """
        if self.model is None:
            # Fallback: estimate from simple model (requires eta_D)
            # This is approximate - ideally RPM mode requires Kt/Kq
            return None  # Or raise NotImplementedError
        
        J = self.advance_coefficient(rpm, speed, wake_fraction)
        KT = self.model.calculate_kt(J)
        n_rps = rpm / 60
        T_gross = KT * self.physics.rho_w * (n_rps**2) * (self.diameter**4)
        return T_gross / (1 - thrust_deduction)
    
    def torque_from_rpm(self, rpm: float, speed: float, 
                       wake_fraction: float) -> float:
        """
        Calculate torque from RPM using Kt/Kq curves.
        Used for power calculation: P = Q × 2π × n
        
        Formula: Q = KQ(J) × ρ × n² × D⁵
        
        Parameters:
        - rpm: Propeller RPM [rev/min]
        - speed: Ship speed [m/s]
        - wake_fraction: w [-]
        
        Returns: Torque [N⋅m]
        """
        if self.model is None:
            return None
        
        J = self.advance_coefficient(rpm, speed, wake_fraction)
        KQ = self.model.calculate_kq(J)
        n_rps = rpm / 60
        return KQ * self.physics.rho_w * (n_rps**2) * (self.diameter**5)
    
    def power_from_rpm(self, rpm: float, speed: float, 
                      wake_fraction: float) -> float:
        """
        Calculate delivered power from RPM.
        P_delivered = Q × ω = Q × 2π × n
        
        Used in power-driven mode to check power constraints.
        """
        Q = self.torque_from_rpm(rpm, speed, wake_fraction)
        if Q is None:
            return None
        n_rps = rpm / 60
        return Q * 2 * np.pi * n_rps
    
    # === Inverse Methods (for optimization) ===
    
    def rpm_from_thrust_speed(self, thrust: float, speed: float,
                             wake_fraction: float, thrust_deduction: float,
                             initial_rpm: float = 100) -> float:
        """
        Calculate RPM needed to produce given thrust at given speed.
        Solves: thrust = thrust_from_rpm(rpm, speed, w, t) for rpm.
        
        Used by optimizer to find feasible rpm for a thrust target.
        Requires iterative solve (scipy.optimize.fsolve).
        
        Returns: RPM [rev/min] or None if no Kt/Kq model
        """
        if self.model is None:
            return None
        
        # Use scipy.optimize.fsolve or newton
        # def residual(rpm): 
        #     return self.thrust_from_rpm(rpm, speed, w, t) - thrust
        # return fsolve(residual, initial_rpm)[0]
```

### 4. Update Hull Class Integration

**File:** `Shipyard/src/core/forces/hull.py`

Integrate Propeller class while maintaining backward compatibility:

**Changes to Hull.__init__() (around line 351):**

```python
# Existing code
self.powering = Powering(self.physics, **powering) if powering is not None else None

# NEW: Add propeller instance
if powering is not None:
    from .propeller import Propeller
    try:
        self.propeller = Propeller(powering, self.physics)
    except (KeyError, ValueError, TypeError):
        # If propeller config incomplete/invalid, skip
        self.propeller = None
else:
    self.propeller = None
```

**Changes to Powering class (around line 206-219):**

Update `propellor_thrust()` method to optionally delegate to Propeller:

```python
def propellor_thrust(self, loadcase: LoadCase, thrust):
    """thrust is actually the residual in x, which can be +/- or zero
    propellor_thrust can only be greater or equal to zero
    >> thrust should be limited to max MCR"""

    eff_thrust = thrust / (1 - self.t)
    
    if loadcase.speed == 0:
        prop_thrust = eff_thrust
    else:
        # Calculate Pe (delivered power) needed
        Pe = eff_thrust * loadcase.speed / (1 - self.t) / self.eta_D
        
        # Check MCR limit
        max_Pe = self.eta_D * self.eta_Tr * self.me_mcr / self.n_prop
        Pe_limited = min(max_Pe, Pe)
        
        # If propeller model available, could use it here
        # (but current speed-driven mode doesn't need it)
        prop_thrust = Pe_limited * self.eta_D / loadcase.speed * self.n_prop
    
    return max(0, prop_thrust)
```

NOTE: Current speed-driven mode doesn't require propeller model. Integration point is prepared for future RPM/power modes.

### 5. Create CLI Test Script

**File:** `Shipyard/test_propeller_cli.py` (new)

CLI script for manual testing during development:

```python
#!/usr/bin/env python3
"""
CLI test script for propeller models.
Run with: python test_propeller_cli.py
"""

from src.core.forces.propeller import Propeller
from src.core.forces.propeller_models.wageningen_b import WageningenB
from src.core.forces.propeller_models.custom_curves import CustomCurves
from src.application.settings import Physics
import numpy as np

def test_wageningen_b():
    """Test Wageningen B model with known propeller."""
    print("=" * 60)
    print("Testing Wageningen B-series Model")
    print("=" * 60)
    
    # Example: 4-blade propeller, P/D=1.0, Ae/A0=0.55
    config = {
        "propeller_model": "wageningen_b",
        "dia_prop": 6.5,  # m
        "n_prop": 1,
        "pitch_diameter_ratio": 1.0,
        "blade_area_ratio": 0.55,
        "number_of_blades": 4,
    }
    
    physics = Physics()
    prop = Propeller(config, physics)
    
    # Test thrust at various RPM and speeds
    rpms = [80, 90, 100, 110, 120]
    speed = 7.0  # m/s (~13.6 knots)
    w = 0.3
    t = 0.2
    
    print(f"\nSpeed: {speed:.1f} m/s, Wake: {w}, Thrust deduction: {t}")
    print(f"{'RPM':>6} {'J':>8} {'KT':>8} {'Thrust(kN)':>12} {'Power(MW)':>12}")
    print("-" * 60)
    
    for rpm in rpms:
        J = prop.advance_coefficient(rpm, speed, w)
        thrust = prop.thrust_from_rpm(rpm, speed, w, t)
        power = prop.power_from_rpm(rpm, speed, w)
        kt = prop.model.calculate_kt(J)
        
        print(f"{rpm:6.0f} {J:8.3f} {kt:8.4f} {thrust/1000:12.1f} {power/1e6:12.3f}")

def test_custom_curves():
    """Test custom curve model."""
    print("\n" + "=" * 60)
    print("Testing Custom Curves Model")
    print("=" * 60)
    
    # Example custom data
    J_data = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    KT_data = [0.45, 0.42, 0.36, 0.28, 0.18, 0.08]
    KQ_data = [0.065, 0.062, 0.055, 0.045, 0.032, 0.018]
    
    config = {
        "propeller_model": "custom_curves",
        "dia_prop": 6.0,
        "n_prop": 1,
        "kt_curve": {"J": J_data, "KT": KT_data},
        "kq_curve": {"J": J_data, "KQ": KQ_data},
    }
    
    physics = Physics()
    prop = Propeller(config, physics)
    
    # Plot efficiency curve
    J_range = np.linspace(0.1, 0.9, 20)
    print(f"\n{'J':>8} {'KT':>8} {'KQ':>8} {'η₀':>8}")
    print("-" * 40)
    for J in J_range:
        kt = prop.model.calculate_kt(J)
        kq = prop.model.calculate_kq(J)
        eta = prop.model.calculate_efficiency(J)
        print(f"{J:8.3f} {kt:8.4f} {kq:8.5f} {eta:8.4f}")

if __name__ == "__main__":
    test_wageningen_b()
    test_custom_curves()
    print("\n" + "=" * 60)
    print("CLI Tests Complete")
    print("=" * 60)
```

### 6. Create Unit Tests

**File:** `Shipyard/test/test_propeller_models.py` (new)

Comprehensive unit tests:

```python
import pytest
import numpy as np
from src.core.forces.propeller_models.wageningen_b import WageningenB
from src.core.forces.propeller_models.custom_curves import CustomCurves

class TestWageningenB:
    def test_kt_calculation(self):
        """Test KT calculation against known values."""
        # Use published data for validation
        
    def test_kq_calculation(self):
        """Test KQ calculation against known values."""
        
    def test_efficiency_calculation(self):
        """Test efficiency η₀ = (J/2π)(KT/KQ)."""
        
    def test_vectorized_evaluation(self):
        """Test that J can be array for batch evaluation."""
        
class TestCustomCurves:
    def test_interpolation(self):
        """Test interpolation of custom curves."""
        
    def test_validation(self):
        """Test validation of input data."""
        # Non-monotonic J should raise ValueError
        # Mismatched array lengths should raise ValueError

class TestPropeller:
    def test_model_selection(self):
        """Test propeller model selection logic."""
        
    def test_simple_model_fallback(self):
        """Test fallback to simple thrust model."""
        
    def test_advance_coefficient(self):
        """Test J = v_a / (n × D) calculation."""
        
    def test_thrust_from_rpm(self):
        """Test thrust calculation from RPM."""
        
    def test_power_from_rpm(self):
        """Test power calculation from RPM."""
```

**File:** `Shipyard/test/test_propeller_integration.py` (new)

Integration tests with Hull:

```python
class TestHullPropellerIntegration:
    def test_hull_with_propeller(self):
        """Test Hull initialization with propeller config."""
        
    def test_hull_without_propeller(self):
        """Test Hull still works without propeller config (backward compat)."""
        
    def test_powering_delegation(self):
        """Test that Powering can access propeller if needed."""
```

## Key Design Decisions

1. **Architecture Agnostic:** RPM/power modes work with any powering system (conventional, DE, PTI/PTO)
2. **Propeller = Evaluator, Not Solver:** Propeller class evaluates thrust/power/torque at given conditions; optimizer handles implicit solves
3. **J-Coupling Explicit:** All RPM-related methods require BOTH rpm and speed because J couples them
4. **Bidirectional Methods:** Support rpm→thrust (forward) and thrust→rpm (inverse via solver) for optimizer flexibility
5. **Graceful Degradation:** Missing Kt/Kq model → fallback to simple model where possible, None otherwise
6. **Future-Ready:** API designed for optimization modes, but current implementation focuses on evaluation methods
7. **CLI Development:** Test via CLI scripts first, API/GUI integration later

## Out of Scope (Deferred)

- API input schema (PoweringInput extensions)
- GraphQL mutations for propeller configuration
- AlbatrosGUI template updates
- Input mapping routines
- Full RPM-driven/power-driven optimizer implementation (only propeller evaluation methods created here)

### To-dos

- [ ] Create WageningenB helper with polynomial KT/KQ calculations and vectorized numpy evaluation
- [ ] Create CustomCurves helper with scipy interpolation and validation
- [ ] Create Propeller class with model selection, thrust_from_rpm, power_from_rpm, and advance_coefficient methods
- [ ] Update Hull class to instantiate Propeller and maintain backward compatibility
- [ ] Create CLI test script for manual testing of Wageningen B and custom curves
- [ ] Create unit tests for WageningenB, CustomCurves, and Propeller classes
- [ ] Create integration tests for Hull-Propeller interaction and backward compatibility

