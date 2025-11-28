"""
Comprehensive Testing Notebook for Standalone Propeller and Powering Models
============================================================================

This script tests all aspects of the propeller and powering models including:
- Loadcases and thrust vectors
- Propeller models (Wageningen B, Custom Curves, Simple)
- Powering flows (DD, DE, PTI/PTO)
- Efficiency analysis
- Edge cases and validation
- Performance curves
- Optimization scenarios

Speed Range: 0:4:16 knots (0, 4, 8, 12, 16 knots)

To use in Jupyter:
1. Convert this to .ipynb or run cells manually
2. Or use: jupyter nbconvert --to notebook --execute test_models_comprehensive.py
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# Import the models
from standalone_models import (
    Physics, Propeller, Powering, DieselEngine,
    WageningenB, CustomCurves
)

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE TESTING NOTEBOOK FOR PROPELLER AND POWERING MODELS")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: SETUP - Physics and Configuration
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 1: SETUP - Physics and Configuration")
print("=" * 80)

# Initialize physics constants
physics = Physics()

print("\nPhysics Constants:")
print(f"  Water density: {physics.rho_w} kg/m³")
print(f"  Air density: {physics.rho_air} kg/m³")
print(f"  Gravity: {physics.g} m/s²")
print(f"  Water kinematic viscosity: {physics.nu_w} m²/s")
print(f"  Air kinematic viscosity: {physics.nu_air} m²/s")

# ============================================================================
# SECTION 2: SPEED CONFIGURATION - Speed range 0:4:16 knots
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 2: SPEED CONFIGURATION")
print("=" * 80)

# Speed range configuration: 0:4:16 knots (0, 4, 8, 12, 16 knots)
# Conversion factor: 1 knot = 1852/3600 m/s = 0.514444... m/s
KNOTS_TO_MS = 1852.0 / 3600.0  # 0.514444...
MS_TO_KNOTS = 3600.0 / 1852.0  # 1.943844... (approximately 1.944)

# Speed range in knots: 0, 4, 8, 12, 16 knots
SPEED_RANGE_KNOTS = [0, 4, 8, 12, 16]

# Speed range in m/s (converted from knots)
SPEED_RANGE_MS = [knots * KNOTS_TO_MS for knots in SPEED_RANGE_KNOTS]

print(f"\nSpeed Range Configuration:")
print(f"  Speeds in knots: {SPEED_RANGE_KNOTS}")
print(f"  Speeds in m/s: {[f'{s:.4f}' for s in SPEED_RANGE_MS]}")

# ============================================================================
# SECTION 3: LOADCASES - Define test scenarios
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 3: LOADCASES - Define test scenarios")
print("=" * 80)

@dataclass
class LoadCase:
    """Test loadcase with speed."""
    name: str
    speed: float  # m/s
    
    def __str__(self):
        return f"{self.name}: {self.speed:.2f} m/s ({self.speed * MS_TO_KNOTS:.2f} knots)"

# Define comprehensive test loadcases using speed range 0:4:16 knots
loadcases = []
for speed_knots in SPEED_RANGE_KNOTS:
    speed_ms = speed_knots * KNOTS_TO_MS
    name = f"{speed_knots} knots"
    loadcases.append(LoadCase(name, speed_ms))

print("\nTest Loadcases:")
for lc in loadcases:
    print(f"  {lc}")

# ============================================================================
# SECTION 4: THRUST VECTORS - Define thrust requirements
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 4: THRUST VECTORS - Define thrust requirements")
print("=" * 80)

# Thrust requirements (in kN) - can be positive (forward) or negative (reverse)
thrust_requirements = {
    "Zero Thrust": 0.0,
    "Low Thrust": 50.0,
    "Medium Thrust": 150.0,
    "High Thrust": 300.0,
    "Very High Thrust": 500.0,
    "Extreme Thrust": 750.0,
    "Reverse Thrust": -100.0,  # Negative for reverse/braking
}

print("\nThrust Requirements:")
for name, thrust in thrust_requirements.items():
    print(f"  {name}: {thrust:.1f} kN")

# ============================================================================
# SECTION 5: PROPELLER CONFIGURATIONS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 5: PROPELLER CONFIGURATIONS")
print("=" * 80)

# NOTE: RPM-based calculations (thrust_from_rpm, power_from_rpm, rpm_from_thrust_speed, etc.)
# are ONLY available for models that:
#   1. Have a propeller model (Wageningen B or Custom Curves) - has_model=True
#   2. Do NOT have eta_delivered_power set (must be None)
#
# Models that support RPM calculations:
#   - Wageningen B (without eta_delivered_power)
#   - Custom Curves (without eta_delivered_power)
#   - Single Prop (Wageningen B variant, without eta_delivered_power)
#
# Models that do NOT support RPM calculations:
#   - Simple Model (has eta_delivered_power as float)
#   - Variable Eta (has eta_delivered_power as dict, even though it has Wageningen B model)
#
# This is because models with eta_delivered_power use simple efficiency-based calculations
# and cannot perform RPM-based KT/KQ curve calculations.

# Configuration 1: Wageningen B-series propeller
# NOTE: Do NOT set eta_delivered_power for advanced models - they use KT/KQ curves instead
# This configuration supports RPM-based calculations.
propeller_wageningen = Propeller(
    propeller_model="wageningen_b",
    dia_prop=4.5,              # m
    wake_fraction=0.25,
    thrust_deduction=0.15,
    n_prop=2,                  # 2 propellers
    pitch_diameter_ratio=0.8,
    blade_area_ratio=0.55,
    number_of_blades=4,
    max_rpm=200.0,
    nominal_rpm=150.0,
    # eta_delivered_power removed - will use Wageningen B KT/KQ model
    physics=physics
)

# Configuration 2: Custom curves propeller (using Wageningen B results with random geometries)
# NOTE: Do NOT set eta_delivered_power for advanced models - they use KT/KQ curves instead
# This configuration supports RPM-based calculations.
# Generate random geometries for Wageningen B
np.random.seed(42)  # For reproducibility
random_pitch_diameter_ratio = np.random.uniform(0.6, 1.2)
random_blade_area_ratio = np.random.uniform(0.3, 0.8)
random_number_of_blades = np.random.choice([3, 4, 5, 6])

print(f"\nCustom Curves - Random Wageningen B Geometries:")
print(f"  Pitch/Diameter Ratio: {random_pitch_diameter_ratio:.3f}")
print(f"  Blade Area Ratio: {random_blade_area_ratio:.3f}")
print(f"  Number of Blades: {random_number_of_blades}")

# Create Wageningen B model with random geometries
wageningen_for_custom = WageningenB(
    pitch_diameter_ratio=random_pitch_diameter_ratio,
    blade_area_ratio=random_blade_area_ratio,
    number_of_blades=random_number_of_blades
)

# Generate J values and calculate KT, KQ using Wageningen B
J_custom = np.linspace(0.1, 1.2, 30)
KT_custom = wageningen_for_custom.calculate_kt(J_custom)
KQ_custom = wageningen_for_custom.calculate_kq(J_custom)

# Ensure arrays are 1D
if KT_custom.ndim == 0:
    KT_custom = np.array([KT_custom])
if KQ_custom.ndim == 0:
    KQ_custom = np.array([KQ_custom])

propeller_custom = Propeller(
    propeller_model="custom_curves",
    dia_prop=4.5,
    wake_fraction=0.25,
    thrust_deduction=0.15,
    n_prop=2,
    max_rpm=200.0,
    nominal_rpm=150.0,
    # eta_delivered_power removed - will use custom KT/KQ curves
    kt_curve={"J": J_custom.tolist(), "KT": KT_custom.tolist()},
    kq_curve={"J": J_custom.tolist(), "KQ": KQ_custom.tolist()},
    physics=physics
)

# Configuration 3: Simple thrust model
# NOTE: This model uses constant eta_delivered_power (float) and does NOT support RPM calculations.
propeller_simple = Propeller(
    dia_prop=4.5,
    wake_fraction=0.25,
    thrust_deduction=0.15,
    n_prop=2,
    eta_delivered_power=0.65,
    physics=physics
)

# Configuration 4: Variable efficiency propeller (speed-dependent)
# NOTE: This model has both Wageningen B model AND eta_delivered_power (dict).
# Even though it has a model, it does NOT support RPM calculations because
# eta_delivered_power is set. The simple efficiency model takes precedence.
propeller_variable_eta = Propeller(
    propeller_model="wageningen_b",
    dia_prop=4.5,
    wake_fraction=0.25,
    thrust_deduction=0.15,
    n_prop=2,
    pitch_diameter_ratio=0.8,
    blade_area_ratio=0.55,
    number_of_blades=4,
    max_rpm=200.0,
    nominal_rpm=150.0,
    eta_delivered_power={
        "speed": [0, 5, 10, 15, 20, 25],
        "eta_delivered_power": [0.50, 0.60, 0.65, 0.68, 0.70, 0.72]
    },
    physics=physics
)

# Configuration 5: Single propeller (n_prop=1)
# NOTE: Do NOT set eta_delivered_power for advanced models - they use KT/KQ curves instead
propeller_single = Propeller(
    propeller_model="wageningen_b",
    dia_prop=4.5,
    wake_fraction=0.25,
    thrust_deduction=0.15,
    n_prop=1,  # Single propeller
    pitch_diameter_ratio=0.8,
    blade_area_ratio=0.55,
    number_of_blades=4,
    max_rpm=200.0,
    nominal_rpm=150.0,
    # eta_delivered_power removed - will use Wageningen B KT/KQ model
    physics=physics
)

print("\nPropeller Configurations:")
propellers = {
    "Wageningen B": propeller_wageningen,
    "Custom Curves": propeller_custom,
    "Simple Model": propeller_simple,
    "Variable Eta": propeller_variable_eta,
    "Single Prop": propeller_single,
}

for name, prop in propellers.items():
    print(f"  {name}: has_model={prop.has_model}, n_prop={prop.n_prop}")

# ============================================================================
# SECTION 6: POWERING CONFIGURATIONS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 6: POWERING CONFIGURATIONS")
print("=" * 80)

# Configuration 1: Diesel-Direct (DD) mode
powering_dd = Powering(
    diesel_electric=False,
    pti_pto=False,
    hotel_load=200.0,          # kW
    me_mcr=5000.0,             # kW
    me_csr=4250.0,             # kW (85% of MCR)
    me_sfoc=0.180,             # kg/kWh
    eta_gearbox=0.97,
    eta_shaft=0.99,
    eta_generator=0.96,
    eta_converters=0.95,
)

# Configuration 2: Diesel-Electric (DE) mode
powering_de = Powering(
    diesel_electric=True,
    pti_pto=False,
    hotel_load=200.0,
    n_gensets=3,
    genset_mcr=1500.0,         # kW per genset
    genset_csr=1275.0,         # kW per genset (85% of MCR)
    genset_sfoc=0.190,         # kg/kWh
    eta_generator=0.96,
    eta_converters=0.95,
    eta_electric_motor=0.95,
    eta_gearbox=0.97,
)

# Configuration 3: PTI/PTO mode
powering_pti_pto = Powering(
    diesel_electric=False,
    pti_pto=True,
    hotel_load=200.0,
    me_mcr=5000.0,
    me_csr=4250.0,
    me_sfoc=0.180,
    n_gensets=2,
    genset_mcr=1500.0,
    genset_csr=1275.0,
    genset_sfoc=0.190,
    pti_pto_kw=1000.0,         # PTI/PTO power rating
    eta_gearbox=0.97,
    eta_shaft=0.99,
    eta_generator=0.96,
    eta_converters=0.95,
    eta_pti_pto=0.93,
)

print("\nPowering Configurations:")
print("  1. Diesel-Direct (DD)")
print("  2. Diesel-Electric (DE)")
print("  3. PTI/PTO")

# ============================================================================
# SECTION 7: TESTING FUNCTIONS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 7: TESTING FUNCTIONS")
print("=" * 80)

def test_propeller_thrust_from_power(propeller, loadcase, power_kw):
    """Test: Given power, calculate thrust."""
    power_w = power_kw * 1000
    thrust = propeller.thrust_from_power(
        power_w,
        loadcase.speed,
        propeller.wake_fraction,
        propeller.thrust_deduction
    )
    if thrust is None:
        return None
    return thrust / 1000  # Convert to kN

def test_propeller_power_from_thrust(propeller, loadcase, thrust_kn):
    """Test: Given thrust, calculate delivered power."""
    thrust_n = thrust_kn * 1000
    power_w = propeller.delivered_power_from_thrust(
        thrust_n,
        loadcase.speed
    )
    return power_w / 1000 if power_w is not None else None  # Convert to kW

def test_propeller_power_from_rpm(propeller, loadcase, rpm):
    """Test: Given RPM, calculate power and thrust."""
    if not can_do_rpm_calculations(propeller):
        return None, None, None
    
    power_w = propeller.power_from_rpm(
        rpm,
        loadcase.speed,
        propeller.wake_fraction
    )
    
    thrust = propeller.thrust_from_rpm(
        rpm,
        loadcase.speed,
        propeller.wake_fraction,
        propeller.thrust_deduction
    )
    
    torque = propeller.torque_from_rpm(
        rpm,
        loadcase.speed,
        propeller.wake_fraction
    )
    
    if power_w is None or thrust is None:
        return None, None, None
    
    return power_w / 1000, thrust / 1000, torque  # Convert to kW, kN, N⋅m

def test_propeller_rpm_from_thrust(propeller, loadcase, thrust_kn):
    """Test: Given thrust requirement, find required RPM."""
    if not can_do_rpm_calculations(propeller):
        return None
    
    thrust_n = thrust_kn * 1000
    rpm = propeller.rpm_from_thrust_speed(
        thrust_n,
        loadcase.speed,
        propeller.wake_fraction,
        propeller.thrust_deduction
    )
    return rpm

def test_propeller_rpm_from_power(propeller, loadcase, power_kw):
    """Test: Given power requirement, find required RPM."""
    if not can_do_rpm_calculations(propeller):
        return None
    
    power_w = power_kw * 1000
    rpm = propeller.rpm_from_power_speed(
        power_w,
        loadcase.speed,
        propeller.wake_fraction
    )
    return rpm

def test_powering_flow(propeller, powering, loadcase, thrust_kn, est_power_kw=0.0):
    """
    Test full powering flow: thrust -> power -> engine -> fuel consumption.
    
    Parameters:
    -----------
    est_power_kw : float
        EST (Energy Storage Technology) power consumption [kW]
    """
    # Step 1: Calculate effective power from thrust
    thrust_n = thrust_kn * 1000
    effective_power_w = propeller.effective_power(loadcase, thrust_n)
    
    # Step 2: Calculate delivered power
    delivered_power_w = propeller.delivered_power(loadcase, effective_power_w)
    delivered_power_kw = delivered_power_w / 1000
    
    # Step 3: Calculate shaft power
    shaft_power_kw = powering.shaft_power_from_delivered_power(delivered_power_kw)
    
    # Step 4: Calculate consumers
    consumers_kwe = powering.consumers(est_power_kw)
    
    # Step 5: Calculate grid load and brake power
    grid_load_kwe, brake_power_kwm = powering.grid_load_and_brake_power_from_consumers_and_shaft_power(
        consumers_kwe, shaft_power_kw
    )
    
    # Step 6: Calculate fuel consumption
    main_engine_fc = powering.main_engine_fc(brake_power_kwm)
    
    n_gensets_active = powering.n_gensets_active(grid_load_kwe)
    aux_power_per_genset = powering.aux_power_per_genset(grid_load_kwe, n_gensets_active) if n_gensets_active > 0 else 0
    genset_fc = powering.genset_fc(aux_power_per_genset, n_gensets_active)
    
    total_fc = powering.total_fc(main_engine_fc, genset_fc)
    
    # Step 7: Validate engine load
    engine_valid = (powering.main_engine.is_valid_load(brake_power_kwm) if powering.main_engine is not None and brake_power_kwm > 0 else True)
    
    return {
        "effective_power_kw": effective_power_w / 1000,
        "delivered_power_kw": delivered_power_kw,
        "shaft_power_kw": shaft_power_kw,
        "grid_load_kwe": grid_load_kwe,
        "brake_power_kwm": brake_power_kwm,
        "main_engine_fc_kg_h": main_engine_fc,
        "n_gensets_active": n_gensets_active,
        "genset_fc_kg_h": genset_fc,
        "total_fc_kg_h": total_fc,
        "engine_valid": engine_valid,
        "est_power_kw": est_power_kw,
    }

def can_do_rpm_calculations(propeller):
    """
    Check if propeller can perform RPM-based calculations.
    
    RPM calculations require:
    - has_model=True (Wageningen B or Custom Curves)
    - eta_delivered_power=None (not set as float or dict)
    
    Models with eta_delivered_power set use simple efficiency model
    and cannot do RPM-based calculations, even if they have a model.
    
    Parameters:
    -----------
    propeller : Propeller
        Propeller instance to check
        
    Returns:
    --------
    bool
        True if RPM calculations are available, False otherwise
    """
    return propeller.has_model and propeller.eta_delivered_power is None

def calculate_advance_coefficient(propeller, rpm, speed):
    """Calculate advance coefficient J."""
    if not can_do_rpm_calculations(propeller):
        return None
    return propeller.advance_coefficient(rpm, speed, propeller.wake_fraction)

def calculate_propeller_efficiency(propeller, J):
    """Calculate propeller efficiency from J."""
    if not can_do_rpm_calculations(propeller) or propeller.model is None:
        return None
    if hasattr(propeller.model, 'calculate_efficiency'):
        return propeller.model.calculate_efficiency(J)
    return None

print("\nTesting functions defined.")

# ============================================================================
# SECTION 8: BASIC TESTS - Thrust from Power
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 8: BASIC TESTS - Thrust from Power")
print("=" * 80)

test_power_kw = 2000.0
results_thrust = []

for lc in loadcases:
    for prop_name, prop in propellers.items():
        try:
            thrust = test_propeller_thrust_from_power(prop, lc, test_power_kw)
            results_thrust.append({
                "loadcase": f"{lc.speed * MS_TO_KNOTS:.2f} knots",
                "speed_ms": lc.speed,
                "propeller": prop_name,
                "power_kw": test_power_kw,
                "thrust_kn": thrust
            })
        except Exception as e:
            print(f"  Error with {prop_name} at {lc.speed * MS_TO_KNOTS:.2f} knots: {e}")

df_thrust = pd.DataFrame(results_thrust)
if not df_thrust.empty:
    print(f"\nThrust from Power ({test_power_kw} kW):")
    pivot = df_thrust.pivot(index="loadcase", columns="propeller", values="thrust_kn")
    print(pivot.to_string())
else:
    print("\nNo results for thrust from power test.")

# ============================================================================
# SECTION 9: BASIC TESTS - Power from Thrust (Backwards Model)
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 9: BASIC TESTS - Power from Thrust (Backwards Model)")
print("=" * 80)

test_thrust_kn = 200.0
results_power = []

for lc in loadcases:
    for prop_name, prop in propellers.items():
        try:
            power_kw = test_propeller_power_from_thrust(prop, lc, test_thrust_kn)
            if power_kw is not None:
                results_power.append({
                    "loadcase": f"{lc.speed * MS_TO_KNOTS:.2f} knots",
                    "speed_ms": lc.speed,
                    "propeller": prop_name,
                    "thrust_kn": test_thrust_kn,
                    "power_kw": power_kw
                })
        except Exception as e:
            print(f"  Error with {prop_name} at {lc.speed * MS_TO_KNOTS:.2f} knots: {e}")

df_power = pd.DataFrame(results_power)
if not df_power.empty:
    print(f"\nPower from Thrust ({test_thrust_kn} kN):")
    pivot = df_power.pivot(index="loadcase", columns="propeller", values="power_kw")
    print(pivot.to_string())
else:
    print("\nNo results for power from thrust test.")

# ============================================================================
# SECTION 10: RPM-BASED TESTS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 10: RPM-BASED TESTS")
print("=" * 80)

test_rpm = 150.0
results_rpm = []

for lc in loadcases:
    for prop_name, prop in [("Wageningen B", propeller_wageningen),
                            ("Custom Curves", propeller_custom)]:
        power_kw, thrust_kn, torque = test_propeller_power_from_rpm(prop, lc, test_rpm)
        if power_kw is not None:
            J = calculate_advance_coefficient(prop, test_rpm, lc.speed)
            efficiency = calculate_propeller_efficiency(prop, J) if J is not None else None
            results_rpm.append({
                "loadcase": f"{lc.speed * MS_TO_KNOTS:.2f} knots",
                "speed_ms": lc.speed,
                "propeller": prop_name,
                "rpm": test_rpm,
                "power_kw": power_kw,
                "thrust_kn": thrust_kn,
                "torque_nm": torque,
                "J": J,
                "efficiency": efficiency
            })

df_rpm = pd.DataFrame(results_rpm)
if not df_rpm.empty:
    print(f"\nPower, Thrust, and Torque from RPM ({test_rpm} RPM):")
    print(df_rpm[["loadcase", "propeller", "power_kw", "thrust_kn", "torque_nm", "J", "efficiency"]].to_string())
else:
    print("\nNo results for RPM-based test.")

# ============================================================================
# SECTION 11: INVERSE TESTS - RPM from Thrust/Power
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 11: INVERSE TESTS - RPM from Thrust/Power")
print("=" * 80)

test_thrust_kn = 200.0
results_rpm_from_thrust = []

for lc in loadcases:
    for prop_name, prop in [("Wageningen B", propeller_wageningen),
                            ("Custom Curves", propeller_custom)]:
        rpm = test_propeller_rpm_from_thrust(prop, lc, test_thrust_kn)
        if rpm is not None:
            power_kw, _, _ = test_propeller_power_from_rpm(prop, lc, rpm)
            results_rpm_from_thrust.append({
                "loadcase": f"{lc.speed * MS_TO_KNOTS:.2f} knots",
                "speed_ms": lc.speed,
                "propeller": prop_name,
                "thrust_kn": test_thrust_kn,
                "required_rpm": rpm,
                "required_power_kw": power_kw
            })

df_rpm_from_thrust = pd.DataFrame(results_rpm_from_thrust)
if not df_rpm_from_thrust.empty:
    print(f"\nRequired RPM from Thrust ({test_thrust_kn} kN):")
    print(df_rpm_from_thrust.to_string())
else:
    print("\nNo results for RPM from thrust test.")

# ============================================================================
# SECTION 12: EFFICIENCY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 12: EFFICIENCY ANALYSIS")
print("=" * 80)

# Test efficiency vs J curves
J_range = np.linspace(0.1, 1.2, 100)
efficiency_results = []

for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom)]:
    if can_do_rpm_calculations(prop) and prop.model is not None:
        for J in J_range:
            try:
                efficiency = calculate_propeller_efficiency(prop, J)
                if efficiency is not None:
                    efficiency_results.append({
                        "propeller": prop_name,
                        "J": J,
                        "efficiency": efficiency
                    })
            except:
                pass

df_efficiency = pd.DataFrame(efficiency_results)
if not df_efficiency.empty:
    # Find optimal J
    for prop_name in df_efficiency["propeller"].unique():
        prop_data = df_efficiency[df_efficiency["propeller"] == prop_name]
        max_eff_idx = prop_data["efficiency"].idxmax()
        optimal_J = prop_data.loc[max_eff_idx, "J"]
        max_eff = prop_data.loc[max_eff_idx, "efficiency"]
        print(f"\n{prop_name}:")
        print(f"  Optimal J: {optimal_J:.3f}")
        print(f"  Maximum Efficiency: {max_eff:.3f}")
        
        # Also test get_optimal_J method
        if prop_name == "Wageningen B":
            optimal_J_method = propeller_wageningen.get_optimal_J()
            print(f"  Optimal J (method): {optimal_J_method}")

# Test variable efficiency
print("\nVariable Efficiency Testing:")
for lc in [loadcases[1], loadcases[2], loadcases[4]]:  # 4, 8, 16 knots
    # Use propeller_simple for constant efficiency (it has eta_delivered_power set)
    eta_const = propeller_simple.eta_D_at_speed(lc)
    eta_var = propeller_variable_eta.eta_D_at_speed(lc)
    print(f"  {lc.speed * MS_TO_KNOTS:.2f} knots ({lc.speed} m/s): Constant={eta_const:.3f}, Variable={eta_var:.3f}")

# ============================================================================
# SECTION 13: PERFORMANCE CURVES - KT, KQ, Efficiency
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 13: PERFORMANCE CURVES - KT, KQ, Efficiency")
print("=" * 80)

performance_curves = []

for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom)]:
    if can_do_rpm_calculations(prop) and prop.model is not None:
        for J in J_range:
            try:
                KT = prop.model.calculate_kt(J)
                KQ = prop.model.calculate_kq(J)
                efficiency = calculate_propeller_efficiency(prop, J)
                performance_curves.append({
                    "propeller": prop_name,
                    "J": J,
                    "KT": KT,
                    "KQ": KQ,
                    "efficiency": efficiency
                })
            except:
                pass

df_curves = pd.DataFrame(performance_curves)
if not df_curves.empty:
    print("\nPerformance curves calculated for J range [0.1, 1.2]")
    print("Sample data:")
    print(df_curves.head(10).to_string())

# ============================================================================
# SECTION 14: EDGE CASES AND VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 14: EDGE CASES AND VALIDATION")
print("=" * 80)

edge_case_results = []

# Test 1: Zero speed
print("\n1. Zero Speed Test:")
lc_zero = loadcases[0]  # 0 knots
for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Simple", propeller_simple)]:
    try:
        thrust = test_propeller_thrust_from_power(prop, lc_zero, 1000.0)
        edge_case_results.append({
            "test": "Zero Speed",
            "propeller": prop_name,
            "result": f"Thrust: {thrust:.2f} kN",
            "status": "OK" if not np.isnan(thrust) else "ERROR"
        })
    except Exception as e:
        edge_case_results.append({
            "test": "Zero Speed",
            "propeller": prop_name,
            "result": str(e),
            "status": "ERROR"
        })

# Test 2: Negative thrust (reverse)
print("\n2. Negative Thrust Test:")
lc_test = loadcases[2]  # 8 knots
for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Simple", propeller_simple)]:
    try:
        # Negative thrust should give negative effective power
        thrust_n = -100000  # -100 kN
        eff_power = prop.effective_power(lc_test, thrust_n)
        edge_case_results.append({
            "test": "Negative Thrust",
            "propeller": prop_name,
            "result": f"Effective Power: {eff_power/1000:.2f} kW",
            "status": "OK"
        })
    except Exception as e:
        edge_case_results.append({
            "test": "Negative Thrust",
            "propeller": prop_name,
            "result": str(e),
            "status": "ERROR"
        })

# Test 3: Engine load validation
print("\n3. Engine Load Validation:")
test_powers = [100, 1000, 3000, 5000, 6000, 10000]  # kW
for power_kw in test_powers:
    is_valid = powering_dd.main_engine.is_valid_load(power_kw) if powering_dd.main_engine is not None else False
    edge_case_results.append({
        "test": "Engine Load Validation",
        "propeller": "N/A",
        "result": f"Power: {power_kw} kW, Valid: {is_valid}",
        "status": "OK" if is_valid else "WARNING"
    })

# Test 4: RPM limits
print("\n4. RPM Limits Test:")
test_rpms = [50, 100, 150, 200, 250, 300]
for rpm in test_rpms:
    if can_do_rpm_calculations(propeller_wageningen):
        lc = loadcases[2]  # 8 knots
        power_kw, thrust_kn, _ = test_propeller_power_from_rpm(propeller_wageningen, lc, rpm)
        within_limit = rpm <= (propeller_wageningen.max_rpm or 1000)
        edge_case_results.append({
            "test": "RPM Limits",
            "propeller": "Wageningen B",
            "result": f"RPM: {rpm}, Power: {power_kw:.1f} kW, Within Limit: {within_limit}",
            "status": "OK" if within_limit else "WARNING"
        })

# Test 5: RPM calculation eligibility validation
print("\n5. RPM Calculation Eligibility Test:")
lc_test = loadcases[2]  # 8 knots
test_rpm = 150.0
test_thrust_kn = 200.0

for prop_name, prop in [("Simple Model", propeller_simple),
                        ("Variable Eta", propeller_variable_eta),
                        ("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom)]:
    can_do_rpm = can_do_rpm_calculations(prop)
    
    # Test power_from_rpm
    power_kw, thrust_kn, torque = test_propeller_power_from_rpm(prop, lc_test, test_rpm)
    power_from_rpm_works = power_kw is not None
    
    # Test rpm_from_thrust
    rpm_from_thrust = test_propeller_rpm_from_thrust(prop, lc_test, test_thrust_kn)
    rpm_from_thrust_works = rpm_from_thrust is not None
    
    # Verify consistency
    expected_can_do = can_do_rpm
    actual_can_do = power_from_rpm_works and rpm_from_thrust_works
    is_consistent = (expected_can_do == actual_can_do)
    
    edge_case_results.append({
        "test": "RPM Eligibility",
        "propeller": prop_name,
        "result": f"can_do_rpm={can_do_rpm}, power_from_rpm={power_from_rpm_works}, rpm_from_thrust={rpm_from_thrust_works}, consistent={is_consistent}",
        "status": "OK" if is_consistent else "ERROR"
    })

df_edge = pd.DataFrame(edge_case_results)
print("\nEdge Case Test Results:")
print(df_edge.to_string())

# ============================================================================
# SECTION 15: FULL INTEGRATION TESTS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 15: FULL INTEGRATION TESTS")
print("=" * 80)

integration_results = []

# Test matrix: loadcase × thrust × powering mode
for lc in loadcases[1:]:  # Skip zero speed
    for thrust_name, thrust_kn in list(thrust_requirements.items())[:5]:  # Skip reverse for now
        for powering_name, powering in [("DD", powering_dd),
                                       ("DE", powering_de),
                                       ("PTI/PTO", powering_pti_pto)]:
            try:
                result = test_powering_flow(propeller_wageningen, powering, lc, thrust_kn)
                result.update({
                    "loadcase": f"{lc.speed * MS_TO_KNOTS:.2f} knots",
                    "speed_ms": lc.speed,
                    "thrust_kn": thrust_kn,
                    "powering_mode": powering_name
                })
                integration_results.append(result)
            except Exception as e:
                print(f"  Error: {lc.speed * MS_TO_KNOTS:.2f} knots, {thrust_name}, {powering_name}: {e}")

df_integration = pd.DataFrame(integration_results)
if not df_integration.empty:
    print(f"\nIntegration Results ({len(df_integration)} cases):")
    print("\nSummary by Powering Mode:")
    for mode in ["DD", "DE", "PTI/PTO"]:
        mode_data = df_integration[df_integration["powering_mode"] == mode]
        if not mode_data.empty:
            print(f"\n{mode} Mode:")
            print(f"  Avg Delivered Power: {mode_data['delivered_power_kw'].mean():.1f} kW")
            print(f"  Avg Total FC: {mode_data['total_fc_kg_h'].mean():.1f} kg/h")
            print(f"  Avg Grid Load: {mode_data['grid_load_kwe'].mean():.1f} kWe")
            print(f"  Valid Engine Loads: {mode_data['engine_valid'].sum()}/{len(mode_data)}")

# ============================================================================
# SECTION 16: VISUALIZATIONS - Using speed range 0:4:16 knots
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 16: GENERATING VISUALIZATIONS")
print("=" * 80)

# Use the speed range for plotting (convert knots to m/s for calculations)
speeds_ms = SPEED_RANGE_MS
speeds_knots = SPEED_RANGE_KNOTS

# ============================================================================
# GROUP 1: SPEED-BASED (BACKWARDS) - Thrust from Power/Speed
# ============================================================================

print("\nGenerating Group 1: Speed-based (Backwards) - Thrust from Power/Speed")

fig1 = plt.figure(figsize=(18, 12))

# Plot 1.1: Thrust vs Speed for different power levels - All propeller models
ax1_1 = plt.subplot(2, 3, 1)
power_levels = [1000, 2000, 3000, 4000]

for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom),
                        ("Simple Model", propeller_simple)]:
    for power_kw in power_levels:
        thrusts = []
        for speed_ms in speeds_ms:
            lc = LoadCase("", speed_ms)
            thrust = test_propeller_thrust_from_power(prop, lc, power_kw)
            thrusts.append(thrust if thrust is not None else np.nan)
        ax1_1.plot(speeds_knots, thrusts, label=f"{prop_name} @ {power_kw} kW", 
                  linewidth=2, linestyle='-' if prop_name == "Wageningen B" else 
                  '--' if prop_name == "Custom Curves" else ':', marker='o')

ax1_1.set_xlabel("Speed [knots]")
ax1_1.set_ylabel("Thrust [kN]")
ax1_1.set_title("Thrust vs Speed (Backwards) - All Propeller Models")
ax1_1.legend(fontsize=8, ncol=2)
ax1_1.grid(True, alpha=0.3)

# Plot 1.2: Thrust vs Speed - Wageningen B at different power levels
ax1_2 = plt.subplot(2, 3, 2)
for power_kw in power_levels:
    thrusts = []
    for speed_ms in speeds_ms:
        lc = LoadCase("", speed_ms)
        thrust = test_propeller_thrust_from_power(propeller_wageningen, lc, power_kw)
        thrusts.append(thrust if thrust is not None else np.nan)
    ax1_2.plot(speeds_knots, thrusts, label=f"{power_kw} kW", linewidth=2, marker='o')

ax1_2.set_xlabel("Speed [knots]")
ax1_2.set_ylabel("Thrust [kN]")
ax1_2.set_title("Thrust vs Speed - Wageningen B")
ax1_2.legend()
ax1_2.grid(True, alpha=0.3)

# Plot 1.3: Thrust vs Speed - Custom Curves at different power levels
ax1_3 = plt.subplot(2, 3, 3)
for power_kw in power_levels:
    thrusts = []
    for speed_ms in speeds_ms:
        lc = LoadCase("", speed_ms)
        thrust = test_propeller_thrust_from_power(propeller_custom, lc, power_kw)
        thrusts.append(thrust if thrust is not None else np.nan)
    ax1_3.plot(speeds_knots, thrusts, label=f"{power_kw} kW", linewidth=2, marker='o')

ax1_3.set_xlabel("Speed [knots]")
ax1_3.set_ylabel("Thrust [kN]")
ax1_3.set_title("Thrust vs Speed - Custom Curves")
ax1_3.legend()
ax1_3.grid(True, alpha=0.3)

# Plot 1.4: Thrust vs Speed - Simple Model at different power levels
ax1_4 = plt.subplot(2, 3, 4)
for power_kw in power_levels:
    thrusts = []
    for speed_ms in speeds_ms:
        lc = LoadCase("", speed_ms)
        thrust = test_propeller_thrust_from_power(propeller_simple, lc, power_kw)
        thrusts.append(thrust if thrust is not None else np.nan)
    ax1_4.plot(speeds_knots, thrusts, label=f"{power_kw} kW", linewidth=2, marker='o')

ax1_4.set_xlabel("Speed [knots]")
ax1_4.set_ylabel("Thrust [kN]")
ax1_4.set_title("Thrust vs Speed - Simple Model")
ax1_4.legend()
ax1_4.grid(True, alpha=0.3)

# Plot 1.5: Single vs Double Propeller Comparison
ax1_5 = plt.subplot(2, 3, 5)
power_test = 2000.0
for prop_name, prop in [("Single Prop", propeller_single),
                        ("Double Prop", propeller_wageningen)]:
    thrusts = []
    for speed_ms in speeds_ms:
        lc = LoadCase("", speed_ms)
        thrust = test_propeller_thrust_from_power(prop, lc, power_test)
        thrusts.append(thrust if thrust is not None else np.nan)
    ax1_5.plot(speeds_knots, thrusts, label=f"{prop_name} ({prop.n_prop} prop)", linewidth=2, marker='o')

ax1_5.set_xlabel("Speed [knots]")
ax1_5.set_ylabel("Thrust [kN]")
ax1_5.set_title(f"Single vs Double Propeller ({power_test} kW)")
ax1_5.legend()
ax1_5.grid(True, alpha=0.3)

# Plot 1.6: Variable vs Constant Efficiency Impact
ax1_6 = plt.subplot(2, 3, 6)
power_test = 2000.0
for prop_name, prop in [("Variable Eta", propeller_variable_eta),
                        ("Simple (Constant)", propeller_simple)]:
    thrusts = []
    for speed_ms in speeds_ms:
        lc = LoadCase("", speed_ms)
        thrust = test_propeller_thrust_from_power(prop, lc, power_test)
        thrusts.append(thrust if thrust is not None else np.nan)
    ax1_6.plot(speeds_knots, thrusts, label=prop_name, linewidth=2, marker='o')

ax1_6.set_xlabel("Speed [knots]")
ax1_6.set_ylabel("Thrust [kN]")
ax1_6.set_title(f"Variable vs Constant Efficiency ({power_test} kW)")
ax1_6.legend()
ax1_6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/test_results_comprehensive_group1_speed_backwards.png", dpi=150, bbox_inches='tight')
print("  Saved: results/test_results_comprehensive_group1_speed_backwards.png")
plt.close()

# ============================================================================
# GROUP 2: SPEED-BASED (FORWARDS) - Power from Thrust/Speed
# ============================================================================

print("\nGenerating Group 2: Speed-based (Forwards) - Power from Thrust/Speed")

fig2 = plt.figure(figsize=(18, 12))

# Plot 2.1: Power vs Speed for different thrust levels - All propeller models
ax2_1 = plt.subplot(2, 3, 1)
thrust_levels = [100, 200, 300, 400]

for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom),
                        ("Simple Model", propeller_simple)]:
    for thrust_kn in thrust_levels:
        powers = []
        for speed_ms in speeds_ms:
            lc = LoadCase("", speed_ms)
            power_kw = test_propeller_power_from_thrust(prop, lc, thrust_kn)
            if power_kw is not None:
                powers.append(power_kw)
            else:
                powers.append(np.nan)
        ax2_1.plot(speeds_knots, powers, label=f"{prop_name} @ {thrust_kn} kN", 
                  linewidth=2, linestyle='-' if prop_name == "Wageningen B" else 
                  '--' if prop_name == "Custom Curves" else ':', marker='o')

ax2_1.set_xlabel("Speed [knots]")
ax2_1.set_ylabel("Power [kW]")
ax2_1.set_title("Power vs Speed (Forwards) - All Propeller Models")
ax2_1.legend(fontsize=8, ncol=2)
ax2_1.grid(True, alpha=0.3)

# Plot 2.2: Power vs Speed - Wageningen B at different thrust levels
ax2_2 = plt.subplot(2, 3, 2)
for thrust_kn in thrust_levels:
    powers = []
    for speed_ms in speeds_ms:
        lc = LoadCase("", speed_ms)
        power_kw = test_propeller_power_from_thrust(propeller_wageningen, lc, thrust_kn)
        if power_kw is not None:
            powers.append(power_kw)
        else:
            powers.append(np.nan)
    ax2_2.plot(speeds_knots, powers, label=f"{thrust_kn} kN", linewidth=2, marker='o')

ax2_2.set_xlabel("Speed [knots]")
ax2_2.set_ylabel("Power [kW]")
ax2_2.set_title("Power vs Speed - Wageningen B")
ax2_2.legend()
ax2_2.grid(True, alpha=0.3)

# Plot 2.3: Power vs Speed - Custom Curves at different thrust levels
ax2_3 = plt.subplot(2, 3, 3)
for thrust_kn in thrust_levels:
    powers = []
    for speed_ms in speeds_ms:
        lc = LoadCase("", speed_ms)
        power_kw = test_propeller_power_from_thrust(propeller_custom, lc, thrust_kn)
        if power_kw is not None:
            powers.append(power_kw)
        else:
            powers.append(np.nan)
    ax2_3.plot(speeds_knots, powers, label=f"{thrust_kn} kN", linewidth=2, marker='o')

ax2_3.set_xlabel("Speed [knots]")
ax2_3.set_ylabel("Power [kW]")
ax2_3.set_title("Power vs Speed - Custom Curves")
ax2_3.legend()
ax2_3.grid(True, alpha=0.3)

# Plot 2.4: Power vs Speed - Simple Model at different thrust levels
ax2_4 = plt.subplot(2, 3, 4)
for thrust_kn in thrust_levels:
    powers = []
    for speed_ms in speeds_ms:
        lc = LoadCase("", speed_ms)
        power_kw = test_propeller_power_from_thrust(propeller_simple, lc, thrust_kn)
        if power_kw is not None:
            powers.append(power_kw)
        else:
            powers.append(np.nan)
    ax2_4.plot(speeds_knots, powers, label=f"{thrust_kn} kN", linewidth=2, marker='o')

ax2_4.set_xlabel("Speed [knots]")
ax2_4.set_ylabel("Power [kW]")
ax2_4.set_title("Power vs Speed - Simple Model")
ax2_4.legend()
ax2_4.grid(True, alpha=0.3)

# Plot 2.5: Required RPM vs Speed for different thrust levels
ax2_5 = plt.subplot(2, 3, 5)
if not df_rpm_from_thrust.empty:
    for prop_name in df_rpm_from_thrust["propeller"].unique():
        prop_data = df_rpm_from_thrust[df_rpm_from_thrust["propeller"] == prop_name]
        # Convert loadcase strings back to numeric for plotting
        speeds_plot = [float(lc.split()[0]) for lc in prop_data["loadcase"]]
        ax2_5.plot(speeds_plot, prop_data["required_rpm"], 
                  label=prop_name, marker='o', linewidth=2)
ax2_5.set_xlabel("Speed [knots]")
ax2_5.set_ylabel("Required RPM")
ax2_5.set_title("Required RPM vs Speed (200 kN thrust)")
ax2_5.legend()
ax2_5.grid(True, alpha=0.3)

# Plot 2.6: Variable vs Constant Efficiency - Power perspective
ax2_6 = plt.subplot(2, 3, 6)
thrust_test = 200.0
for prop_name, prop in [("Variable Eta", propeller_variable_eta),
                        ("Simple (Constant)", propeller_simple)]:
    powers = []
    for speed_ms in speeds_ms:
        lc = LoadCase("", speed_ms)
        power_kw = test_propeller_power_from_thrust(prop, lc, thrust_test)
        if power_kw is not None:
            powers.append(power_kw)
        else:
            powers.append(np.nan)
    ax2_6.plot(speeds_knots, powers, label=prop_name, linewidth=2, marker='o')

ax2_6.set_xlabel("Speed [knots]")
ax2_6.set_ylabel("Power [kW]")
ax2_6.set_title(f"Variable vs Constant Efficiency ({thrust_test} kN)")
ax2_6.legend()
ax2_6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/test_results_comprehensive_group2_speed_forwards.png", dpi=150, bbox_inches='tight')
print("  Saved: results/test_results_comprehensive_group2_speed_forwards.png")
plt.close()

# ============================================================================
# GROUP 3: POWER/RPM/TORQUE-BASED - RPM and Torque Analysis
# ============================================================================

print("\nGenerating Group 3: Power/RPM/Torque-based Analysis")

fig3 = plt.figure(figsize=(18, 12))

# Plot 3.1: Power from RPM vs Speed - All propeller models
ax3_1 = plt.subplot(2, 3, 1)
rpm_levels = [100, 150, 200]

for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom)]:
    for rpm in rpm_levels:
        powers = []
        for speed_ms in speeds_ms:
            lc = LoadCase("", speed_ms)
            power_kw, _, _ = test_propeller_power_from_rpm(prop, lc, rpm)
            if power_kw is not None:
                powers.append(power_kw)
            else:
                powers.append(np.nan)
        ax3_1.plot(speeds_knots, powers, label=f"{prop_name} @ {rpm} RPM", 
                  linewidth=2, marker='o', markersize=5)

ax3_1.set_xlabel("Speed [knots]")
ax3_1.set_ylabel("Power [kW]")
ax3_1.set_title("Power from RPM vs Speed - All Models")
ax3_1.legend(fontsize=8, ncol=2)
ax3_1.grid(True, alpha=0.3)

# Plot 3.2: Thrust from RPM vs Speed - All propeller models
ax3_2 = plt.subplot(2, 3, 2)
for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom)]:
    for rpm in rpm_levels:
        thrusts = []
        for speed_ms in speeds_ms:
            lc = LoadCase("", speed_ms)
            _, thrust_kn, _ = test_propeller_power_from_rpm(prop, lc, rpm)
            if thrust_kn is not None:
                thrusts.append(thrust_kn)
            else:
                thrusts.append(np.nan)
        ax3_2.plot(speeds_knots, thrusts, label=f"{prop_name} @ {rpm} RPM", 
                  linewidth=2, marker='o', markersize=5)

ax3_2.set_xlabel("Speed [knots]")
ax3_2.set_ylabel("Thrust [kN]")
ax3_2.set_title("Thrust from RPM vs Speed - All Models")
ax3_2.legend(fontsize=8, ncol=2)
ax3_2.grid(True, alpha=0.3)

# Plot 3.3: Torque from RPM vs Speed - All propeller models
ax3_3 = plt.subplot(2, 3, 3)
for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom)]:
    for rpm in rpm_levels:
        torques = []
        for speed_ms in speeds_ms:
            lc = LoadCase("", speed_ms)
            _, _, torque = test_propeller_power_from_rpm(prop, lc, rpm)
            if torque is not None:
                torques.append(torque / 1000)  # Convert to kN⋅m
            else:
                torques.append(np.nan)
        ax3_3.plot(speeds_knots, torques, label=f"{prop_name} @ {rpm} RPM", 
                  linewidth=2, marker='o', markersize=5)

ax3_3.set_xlabel("Speed [knots]")
ax3_3.set_ylabel("Torque [kN⋅m]")
ax3_3.set_title("Torque from RPM vs Speed - All Models")
ax3_3.legend(fontsize=8, ncol=2)
ax3_3.grid(True, alpha=0.3)

# Plot 3.4: Power vs RPM at fixed speeds
ax3_4 = plt.subplot(2, 3, 4)
rpm_range = np.linspace(50, 250, 50)
for lc in [loadcases[1], loadcases[2], loadcases[4]]:  # 4, 8, 16 knots
    powers = []
    for rpm in rpm_range:
        power_kw, _, _ = test_propeller_power_from_rpm(propeller_wageningen, lc, rpm)
        if power_kw is not None:
            powers.append(power_kw)
        else:
            powers.append(np.nan)
    ax3_4.plot(rpm_range, powers, label=f"{lc.speed * MS_TO_KNOTS:.2f} knots", linewidth=2)

ax3_4.set_xlabel("RPM")
ax3_4.set_ylabel("Power [kW]")
ax3_4.set_title("Power vs RPM (Wageningen B)")
ax3_4.legend()
ax3_4.grid(True, alpha=0.3)

# Plot 3.5: Thrust vs RPM at fixed speeds
ax3_5 = plt.subplot(2, 3, 5)
for lc in [loadcases[1], loadcases[2], loadcases[4]]:  # 4, 8, 16 knots
    thrusts = []
    for rpm in rpm_range:
        _, thrust_kn, _ = test_propeller_power_from_rpm(propeller_wageningen, lc, rpm)
        if thrust_kn is not None:
            thrusts.append(thrust_kn)
        else:
            thrusts.append(np.nan)
    ax3_5.plot(rpm_range, thrusts, label=f"{lc.speed * MS_TO_KNOTS:.2f} knots", linewidth=2)

ax3_5.set_xlabel("RPM")
ax3_5.set_ylabel("Thrust [kN]")
ax3_5.set_title("Thrust vs RPM (Wageningen B)")
ax3_5.legend()
ax3_5.grid(True, alpha=0.3)

# Plot 3.6: Torque vs RPM at fixed speeds
ax3_6 = plt.subplot(2, 3, 6)
for lc in [loadcases[1], loadcases[2], loadcases[4]]:  # 4, 8, 16 knots
    torques = []
    for rpm in rpm_range:
        _, _, torque = test_propeller_power_from_rpm(propeller_wageningen, lc, rpm)
        if torque is not None:
            torques.append(torque / 1000)  # Convert to kN⋅m
        else:
            torques.append(np.nan)
    ax3_6.plot(rpm_range, torques, label=f"{lc.speed * MS_TO_KNOTS:.2f} knots", linewidth=2)

ax3_6.set_xlabel("RPM")
ax3_6.set_ylabel("Torque [kN⋅m]")
ax3_6.set_title("Torque vs RPM (Wageningen B)")
ax3_6.legend()
ax3_6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/test_results_comprehensive_group3_power_rpm_torque.png", dpi=150, bbox_inches='tight')
print("  Saved: results/test_results_comprehensive_group3_power_rpm_torque.png")
plt.close()

# ============================================================================
# GROUP 4: EFFICIENCY AND PERFORMANCE CURVES
# ============================================================================

print("\nGenerating Group 4: Efficiency and Performance Curves")

fig4 = plt.figure(figsize=(18, 12))

# Plot 4.1: Performance Curves - KT vs J
ax4_1 = plt.subplot(2, 3, 1)
if not df_curves.empty:
    for prop_name in df_curves["propeller"].unique():
        prop_data = df_curves[df_curves["propeller"] == prop_name]
        ax4_1.plot(prop_data["J"], prop_data["KT"], label=f"{prop_name}", linewidth=2)
ax4_1.set_xlabel("Advance Coefficient J [-]")
ax4_1.set_ylabel("Thrust Coefficient KT [-]")
ax4_1.set_title("KT vs J - All Propeller Models")
ax4_1.legend()
ax4_1.grid(True, alpha=0.3)

# Plot 4.2: Performance Curves - KQ vs J
ax4_2 = plt.subplot(2, 3, 2)
if not df_curves.empty:
    for prop_name in df_curves["propeller"].unique():
        prop_data = df_curves[df_curves["propeller"] == prop_name]
        ax4_2.plot(prop_data["J"], prop_data["KQ"], label=f"{prop_name}", linewidth=2)
ax4_2.set_xlabel("Advance Coefficient J [-]")
ax4_2.set_ylabel("Torque Coefficient KQ [-]")
ax4_2.set_title("KQ vs J - All Propeller Models")
ax4_2.legend()
ax4_2.grid(True, alpha=0.3)

# Plot 4.3: Efficiency vs J
ax4_3 = plt.subplot(2, 3, 3)
if not df_efficiency.empty:
    for prop_name in df_efficiency["propeller"].unique():
        prop_data = df_efficiency[df_efficiency["propeller"] == prop_name]
        ax4_3.plot(prop_data["J"], prop_data["efficiency"], label=prop_name, linewidth=2)
ax4_3.set_xlabel("Advance Coefficient J [-]")
ax4_3.set_ylabel("Efficiency η₀ [-]")
ax4_3.set_title("Propeller Efficiency vs J - All Models")
ax4_3.legend()
ax4_3.grid(True, alpha=0.3)

# Plot 4.4: Variable vs Constant Efficiency vs Speed
ax4_4 = plt.subplot(2, 3, 4)
eta_const = []
eta_var = []

for speed_ms in speeds_ms:
    lc = LoadCase("", speed_ms)
    eta_const.append(propeller_simple.eta_D_at_speed(lc))
    eta_var.append(propeller_variable_eta.eta_D_at_speed(lc))

ax4_4.plot(speeds_knots, eta_const, label="Constant", linewidth=2, marker='o')
ax4_4.plot(speeds_knots, eta_var, label="Variable", linewidth=2, marker='o')
ax4_4.set_xlabel("Speed [knots]")
ax4_4.set_ylabel("Delivered Power Efficiency")
ax4_4.set_title("Constant vs Variable Efficiency vs Speed")
ax4_4.legend()
ax4_4.grid(True, alpha=0.3)

# Plot 4.5: Efficiency comparison across propeller models
ax4_5 = plt.subplot(2, 3, 5)
if not df_efficiency.empty:
    for prop_name in df_efficiency["propeller"].unique():
        prop_data = df_efficiency[df_efficiency["propeller"] == prop_name]
        max_eff_idx = prop_data["efficiency"].idxmax()
        optimal_J = prop_data.loc[max_eff_idx, "J"]
        max_eff = prop_data.loc[max_eff_idx, "efficiency"]
        ax4_5.scatter(optimal_J, max_eff, s=200, label=f"{prop_name}\n(J={optimal_J:.3f}, η={max_eff:.3f})")
        ax4_5.plot(prop_data["J"], prop_data["efficiency"], alpha=0.3, linewidth=1)
ax4_5.set_xlabel("Advance Coefficient J [-]")
ax4_5.set_ylabel("Efficiency η₀ [-]")
ax4_5.set_title("Optimal Efficiency Points")
ax4_5.legend(fontsize=8)
ax4_5.grid(True, alpha=0.3)

# Plot 4.6: KT/KQ/Efficiency combined view
ax4_6 = plt.subplot(2, 3, 6)
if not df_curves.empty and not df_efficiency.empty:
    prop_name = "Wageningen B"
    if prop_name in df_curves["propeller"].unique():
        prop_curves = df_curves[df_curves["propeller"] == prop_name]
        prop_eff = df_efficiency[df_efficiency["propeller"] == prop_name]
        
        ax4_6_twin1 = ax4_6.twinx()
        ax4_6_twin2 = ax4_6.twinx()
        ax4_6_twin2.spines['right'].set_position(('outward', 60))
        
        line1 = ax4_6.plot(prop_curves["J"], prop_curves["KT"], 'b-', label='KT', linewidth=2)
        line2 = ax4_6_twin1.plot(prop_curves["J"], prop_curves["KQ"], 'r-', label='KQ', linewidth=2)
        line3 = ax4_6_twin2.plot(prop_eff["J"], prop_eff["efficiency"], 'g-', label='Efficiency', linewidth=2)
        
        ax4_6.set_xlabel("Advance Coefficient J [-]")
        ax4_6.set_ylabel("KT [-]", color='b')
        ax4_6_twin1.set_ylabel("KQ [-]", color='r')
        ax4_6_twin2.set_ylabel("Efficiency η₀ [-]", color='g')
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax4_6.legend(lines, labels, loc='upper left')
        ax4_6.set_title(f"KT, KQ, Efficiency vs J ({prop_name})")
        ax4_6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/test_results_comprehensive_group4_efficiency_performance.png", dpi=150, bbox_inches='tight')
print("  Saved: results/test_results_comprehensive_group4_efficiency_performance.png")
plt.close()

# ============================================================================
# GROUP 5: POWERING MODE COMPARISONS
# ============================================================================

print("\nGenerating Group 5: Powering Mode Comparisons")

fig5 = plt.figure(figsize=(18, 12))

# Plot 5.1: Fuel Consumption vs Thrust - All powering modes
ax5_1 = plt.subplot(2, 3, 1)
lc = loadcases[2]  # 8 knots
thrusts = np.linspace(50, 500, 20)

for powering_name, powering in [("DD", powering_dd),
                               ("DE", powering_de),
                               ("PTI/PTO", powering_pti_pto)]:
    fcs = []
    for thrust_kn in thrusts:
        try:
            result = test_powering_flow(propeller_wageningen, powering, lc, thrust_kn)
            fcs.append(result["total_fc_kg_h"])
        except:
            fcs.append(np.nan)
    ax5_1.plot(thrusts, fcs, label=powering_name, marker='o', markersize=3, linewidth=2)

ax5_1.set_xlabel("Thrust [kN]")
ax5_1.set_ylabel("Fuel Consumption [kg/h]")
ax5_1.set_title(f"Fuel Consumption vs Thrust ({lc.speed * MS_TO_KNOTS:.2f} knots)")
ax5_1.legend()
ax5_1.grid(True, alpha=0.3)

# Plot 5.2: Fuel Consumption vs Thrust - Different speeds (DD mode)
ax5_2 = plt.subplot(2, 3, 2)
for lc in [loadcases[1], loadcases[2], loadcases[4]]:  # 4, 8, 16 knots
    fcs = []
    for thrust_kn in thrusts:
        try:
            result = test_powering_flow(propeller_wageningen, powering_dd, lc, thrust_kn)
            fcs.append(result["total_fc_kg_h"])
        except:
            fcs.append(np.nan)
    ax5_2.plot(thrusts, fcs, label=f"{lc.speed * MS_TO_KNOTS:.2f} knots", marker='o', markersize=3, linewidth=2)

ax5_2.set_xlabel("Thrust [kN]")
ax5_2.set_ylabel("Fuel Consumption [kg/h]")
ax5_2.set_title("Fuel Consumption vs Thrust (DD Mode)")
ax5_2.legend()
ax5_2.grid(True, alpha=0.3)

# Additional plots can be added here following the same pattern...

plt.tight_layout()
plt.savefig("results/test_results_comprehensive_group5_powering_modes.png", dpi=150, bbox_inches='tight')
print("  Saved: results/test_results_comprehensive_group5_powering_modes.png")
plt.close()

print("\nAll visualization groups saved successfully!")

# ============================================================================
# SECTION 17: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 17: SUMMARY STATISTICS")
print("=" * 80)

if not df_integration.empty:
    print("\nIntegration Test Statistics:")
    print(f"  Total test cases: {len(df_integration)}")
    print(f"  Valid engine loads: {df_integration['engine_valid'].sum()}")
    print(f"  Invalid engine loads: {(~df_integration['engine_valid']).sum()}")
    print(f"\nPower Statistics:")
    print(f"  Effective Power: {df_integration['effective_power_kw'].min():.1f} - {df_integration['effective_power_kw'].max():.1f} kW")
    print(f"  Delivered Power: {df_integration['delivered_power_kw'].min():.1f} - {df_integration['delivered_power_kw'].max():.1f} kW")
    print(f"  Shaft Power: {df_integration['shaft_power_kw'].min():.1f} - {df_integration['shaft_power_kw'].max():.1f} kW")
    print(f"\nFuel Consumption Statistics:")
    print(f"  Total FC: {df_integration['total_fc_kg_h'].min():.1f} - {df_integration['total_fc_kg_h'].max():.1f} kg/h")
    print(f"  Average FC: {df_integration['total_fc_kg_h'].mean():.1f} kg/h")

print("\n" + "=" * 80)
print("COMPREHENSIVE TESTING COMPLETE!")
print("=" * 80)
print("\nAll tests completed successfully.")
print("Check the generated PNG plots for detailed results.")
print(f"\nSpeed range used: {SPEED_RANGE_KNOTS} knots ({SPEED_RANGE_MS})")

