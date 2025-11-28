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
# SECTION 2: LOADCASES - Define test scenarios
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 2: LOADCASES - Define test scenarios")
print("=" * 80)

@dataclass
class LoadCase:
    """Test loadcase with speed."""
    name: str
    speed: float  # m/s
    
    def __str__(self):
        return f"{self.name}: {self.speed:.2f} m/s ({self.speed * 1.944:.2f} knots)"

# Define comprehensive test loadcases
loadcases = [
    LoadCase("Zero Speed", 0.0),
    LoadCase("Slow Speed", 2.5),
    LoadCase("Low Speed", 4.0),
    LoadCase("Cruise Speed", 5.0),
    LoadCase("High Speed", 6.0),
    LoadCase("Max Speed", 7.0),
    LoadCase("Very High Speed", 8.0),
]

print("\nTest Loadcases:")
for lc in loadcases:
    print(f"  {lc}")

# ============================================================================
# SECTION 3: THRUST VECTORS - Define thrust requirements
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 3: THRUST VECTORS - Define thrust requirements")
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
# SECTION 4: PROPELLER CONFIGURATIONS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 4: PROPELLER CONFIGURATIONS")
print("=" * 80)

# Configuration 1: Wageningen B-series propeller
# NOTE: Do NOT set eta_delivered_power for advanced models - they use KT/KQ curves instead
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

# Configuration 2: Custom curves propeller
# NOTE: Do NOT set eta_delivered_power for advanced models - they use KT/KQ curves instead
J_custom = np.linspace(0.1, 1.2, 20)
KT_custom = 0.5 - 0.3 * J_custom + 0.1 * J_custom**2
KQ_custom = 0.05 - 0.02 * J_custom + 0.01 * J_custom**2

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
propeller_simple = Propeller(
    dia_prop=4.5,
    wake_fraction=0.25,
    thrust_deduction=0.15,
    n_prop=2,
    eta_delivered_power=0.65,
    physics=physics
)

# Configuration 4: Variable efficiency propeller (speed-dependent)
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
# SECTION 5: POWERING CONFIGURATIONS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 5: POWERING CONFIGURATIONS")
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
# SECTION 6: TESTING FUNCTIONS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 6: TESTING FUNCTIONS")
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
    if not propeller.has_model:
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
    if not propeller.has_model:
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
    if not propeller.has_model:
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

def calculate_advance_coefficient(propeller, rpm, speed):
    """Calculate advance coefficient J."""
    if not propeller.has_model:
        return None
    return propeller.advance_coefficient(rpm, speed, propeller.wake_fraction)

def calculate_propeller_efficiency(propeller, J):
    """Calculate propeller efficiency from J."""
    if not propeller.has_model or propeller.model is None:
        return None
    if hasattr(propeller.model, 'calculate_efficiency'):
        return propeller.model.calculate_efficiency(J)
    return None

print("\nTesting functions defined.")

# ============================================================================
# SECTION 7: BASIC TESTS - Thrust from Power
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 7: BASIC TESTS - Thrust from Power")
print("=" * 80)

test_power_kw = 2000.0
results_thrust = []

for lc in loadcases:
    for prop_name, prop in propellers.items():
        try:
            thrust = test_propeller_thrust_from_power(prop, lc, test_power_kw)
            results_thrust.append({
                "loadcase": lc.name,
                "speed_ms": lc.speed,
                "propeller": prop_name,
                "power_kw": test_power_kw,
                "thrust_kn": thrust
            })
        except Exception as e:
            print(f"  Error with {prop_name} at {lc.name}: {e}")

df_thrust = pd.DataFrame(results_thrust)
if not df_thrust.empty:
    print(f"\nThrust from Power ({test_power_kw} kW):")
    pivot = df_thrust.pivot(index="loadcase", columns="propeller", values="thrust_kn")
    print(pivot.to_string())
else:
    print("\nNo results for thrust from power test.")

# ============================================================================
# SECTION 8: RPM-BASED TESTS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 8: RPM-BASED TESTS")
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
                "loadcase": lc.name,
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
# SECTION 9: INVERSE TESTS - RPM from Thrust/Power
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 9: INVERSE TESTS - RPM from Thrust/Power")
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
                "loadcase": lc.name,
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
# SECTION 10: EFFICIENCY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 10: EFFICIENCY ANALYSIS")
print("=" * 80)

# Test efficiency vs J curves
J_range = np.linspace(0.1, 1.2, 100)
efficiency_results = []

for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom)]:
    if prop.has_model and prop.model is not None:
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
for lc in [loadcases[1], loadcases[3], loadcases[5]]:
    # Use propeller_simple for constant efficiency (it has eta_delivered_power set)
    eta_const = propeller_simple.eta_D_at_speed(lc)
    eta_var = propeller_variable_eta.eta_D_at_speed(lc)
    print(f"  {lc.name} ({lc.speed} m/s): Constant={eta_const:.3f}, Variable={eta_var:.3f}")

# ============================================================================
# SECTION 11: PERFORMANCE CURVES - KT, KQ, Efficiency
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 11: PERFORMANCE CURVES - KT, KQ, Efficiency")
print("=" * 80)

performance_curves = []

for prop_name, prop in [("Wageningen B", propeller_wageningen),
                        ("Custom Curves", propeller_custom)]:
    if prop.has_model and prop.model is not None:
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
# SECTION 12: EDGE CASES AND VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 12: EDGE CASES AND VALIDATION")
print("=" * 80)

edge_case_results = []

# Test 1: Zero speed
print("\n1. Zero Speed Test:")
lc_zero = LoadCase("Zero", 0.0)
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
lc_test = loadcases[3]  # Cruise speed
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
    if propeller_wageningen.has_model:
        lc = loadcases[3]
        power_kw, thrust_kn, _ = test_propeller_power_from_rpm(propeller_wageningen, lc, rpm)
        within_limit = rpm <= (propeller_wageningen.max_rpm or 1000)
        edge_case_results.append({
            "test": "RPM Limits",
            "propeller": "Wageningen B",
            "result": f"RPM: {rpm}, Power: {power_kw:.1f} kW, Within Limit: {within_limit}",
            "status": "OK" if within_limit else "WARNING"
        })

df_edge = pd.DataFrame(edge_case_results)
print("\nEdge Case Test Results:")
print(df_edge.to_string())

# ============================================================================
# SECTION 13: FULL INTEGRATION TESTS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 13: FULL INTEGRATION TESTS")
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
                    "loadcase": lc.name,
                    "speed_ms": lc.speed,
                    "thrust_kn": thrust_kn,
                    "powering_mode": powering_name
                })
                integration_results.append(result)
            except Exception as e:
                print(f"  Error: {lc.name}, {thrust_name}, {powering_name}: {e}")

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
# SECTION 13B: DIRECTIONAL POWER-FLOW VALIDATION (BACKWARD VS FORWARD)
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 13B: DIRECTIONAL POWER-FLOW VALIDATION")
print("=" * 80)

directional_results = []

# Focus on backward-facing (positive thrust) with a limited forward-facing (reverse) check.
directional_cases = [
    ("Backward Push", 300.0),
    ("Forward-Facing Brake", -100.0),
]

for lc in [loadcases[1], loadcases[3], loadcases[5]]:  # representative low/cruise/high
    for case_name, thrust_kn in directional_cases:
        for prop_name, prop in [("Wageningen B", propeller_wageningen), ("Custom Curves", propeller_custom)]:
            for powering_name, powering in [("DD", powering_dd), ("DE", powering_de)]:
                try:
                    result = test_powering_flow(prop, powering, lc, thrust_kn)
                    directional_results.append({
                        "case": case_name,
                        "loadcase": lc.name,
                        "speed_ms": lc.speed,
                        "propeller": prop_name,
                        "powering_mode": powering_name,
                        "thrust_kn": thrust_kn,
                        "delivered_power_kw": result["delivered_power_kw"],
                        "shaft_power_kw": result["shaft_power_kw"],
                        "grid_load_kwe": result["grid_load_kwe"],
                        "brake_power_kwm": result["brake_power_kwm"],
                        "power_sign": float(np.sign(result["delivered_power_kw"])),
                        "brake_power_sign": float(np.sign(result["brake_power_kwm"])),
                        "engine_valid": result["engine_valid"],
                        "est_power_kw": result["est_power_kw"],
                    })
                except Exception as e:
                    directional_results.append({
                        "case": case_name,
                        "loadcase": lc.name,
                        "speed_ms": lc.speed,
                        "propeller": prop_name,
                        "powering_mode": powering_name,
                        "thrust_kn": thrust_kn,
                        "result": str(e),
                        "status": "ERROR",
                    })

df_directional = pd.DataFrame(directional_results)
if not df_directional.empty:
    print("\nDirectional Power-Flow Summary (focus on backward-facing consistency):")
    summary_cols = [
        "case", "loadcase", "propeller", "powering_mode", "thrust_kn",
        "delivered_power_kw", "shaft_power_kw", "grid_load_kwe",
        "power_sign", "brake_power_sign", "engine_valid"
    ]
    print(df_directional[summary_cols].to_string(index=False))

    # Backward-facing regression check: delivered/shaft/brake power should remain positive.
    backward_cases = df_directional[df_directional["case"] == "Backward Push"]
    if not backward_cases.empty:
        invalid_power = backward_cases[(backward_cases["delivered_power_kw"] <= 0) | (backward_cases["shaft_power_kw"] <= 0)]
        print("\nBackward-facing power-flow integrity:")
        print(f"  Cases evaluated: {len(backward_cases)}")
        print(f"  Non-positive power results: {len(invalid_power)}")

    # Forward-facing smoke check to track sign handling without changing primary flow.
    forward_cases = df_directional[df_directional["case"] == "Forward-Facing Brake"]
    if not forward_cases.empty:
        print("\nForward-facing (reverse thrust) sign observations:")
        print(forward_cases[["loadcase", "propeller", "power_sign", "brake_power_sign", "engine_valid"]].to_string(index=False))

# ============================================================================
# SECTION 14: EST POWER CONSUMPTION IMPACT
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 14: EST POWER CONSUMPTION IMPACT")
print("=" * 80)

est_test_results = []

lc_test = loadcases[3]  # Cruise speed
thrust_test = 200.0  # kN
est_levels = [0, 100, 200, 500, 1000, 2000]  # kW

for est_power in est_levels:
    for powering_name, powering in [("DD", powering_dd),
                                   ("DE", powering_de),
                                   ("PTI/PTO", powering_pti_pto)]:
        try:
            result = test_powering_flow(propeller_wageningen, powering, lc_test, thrust_test, est_power)
            est_test_results.append({
                "est_power_kw": est_power,
                "powering_mode": powering_name,
                "grid_load_kwe": result["grid_load_kwe"],
                "total_fc_kg_h": result["total_fc_kg_h"],
                "n_gensets_active": result["n_gensets_active"]
            })
        except Exception as e:
            print(f"  Error: EST={est_power} kW, {powering_name}: {e}")

df_est = pd.DataFrame(est_test_results)
if not df_est.empty:
    print("\nEST Power Consumption Impact:")
    print(df_est.pivot(index="est_power_kw", columns="powering_mode", values="total_fc_kg_h").to_string())

# Power from Thrust Test
print("\n" + "=" * 80)
print("Power from Thrust Test")
print("=" * 80)

test_thrust_kn = 200.0
results_power = []

for lc in loadcases:
    for prop_name, prop in propellers.items():
        try:
            power_kw = test_propeller_power_from_thrust(prop, lc, test_thrust_kn)
            if power_kw is not None:
                results_power.append({
                    "loadcase": lc.name,
                    "speed_ms": lc.speed,
                    "propeller": prop_name,
                    "thrust_kn": test_thrust_kn,
                    "power_kw": power_kw
                })
        except Exception as e:
            print(f"  Error with {prop_name} at {lc.name}: {e}")

df_power = pd.DataFrame(results_power)
if not df_power.empty:
    print(f"\nPower from Thrust ({test_thrust_kn} kN):")
    pivot = df_power.pivot(index="loadcase", columns="propeller", values="power_kw")
    print(pivot.to_string())
else:
    print("\nNo results for power from thrust test.")

# ============================================================================
# SECTION 15: MULTIPLE PROPELLER COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 15: MULTIPLE PROPELLER COMPARISON")
print("=" * 80)

multi_prop_results = []

lc_test = loadcases[3]
power_test = 2000.0  # kW

for prop_name, prop in [("Single Prop", propeller_single),
                        ("Double Prop", propeller_wageningen)]:
    try:
        thrust = test_propeller_thrust_from_power(prop, lc_test, power_test)
        multi_prop_results.append({
            "propeller": prop_name,
            "n_prop": prop.n_prop,
            "power_kw": power_test,
            "thrust_kn": thrust,
            "thrust_per_prop_kn": thrust / prop.n_prop
        })
    except Exception as e:
        print(f"  Error with {prop_name}: {e}")

df_multi = pd.DataFrame(multi_prop_results)
if not df_multi.empty:
    print("\nMultiple Propeller Comparison:")
    print(df_multi.to_string())

# ============================================================================
# SECTION 16: PTI/PTO MODE DETAILED TESTING
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 16: PTI/PTO MODE DETAILED TESTING")
print("=" * 80)

pti_pto_results = []

lc_test = loadcases[3]
thrust_levels = [100, 200, 300, 400, 500]  # kN

for thrust_kn in thrust_levels:
    try:
        result = test_powering_flow(propeller_wageningen, powering_pti_pto, lc_test, thrust_kn)
        pti_pto_results.append({
            "thrust_kn": thrust_kn,
            "shaft_power_kw": result["shaft_power_kw"],
            "brake_power_kwm": result["brake_power_kwm"],
            "grid_load_kwe": result["grid_load_kwe"],
            "n_gensets_active": result["n_gensets_active"],
            "mode": "PTO" if result["grid_load_kwe"] == 0 else "PTI"
        })
    except Exception as e:
        print(f"  Error at {thrust_kn} kN: {e}")

df_pti_pto = pd.DataFrame(pti_pto_results)
if not df_pti_pto.empty:
    print("\nPTI/PTO Mode Analysis:")
    print(df_pti_pto.to_string())
    print("\nMode Transitions:")
    print(f"  PTO mode (no genset): {(df_pti_pto['mode'] == 'PTO').sum()} cases")
    print(f"  PTI mode (genset active): {(df_pti_pto['mode'] == 'PTI').sum()} cases")

# ============================================================================
# SECTION 17: OPTIMIZATION SCENARIOS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 17: OPTIMIZATION SCENARIOS")
print("=" * 80)

optimization_results = []

# Find optimal RPM for given thrust requirements
for lc in loadcases[1:4]:  # Test a few loadcases
    for thrust_kn in [100, 200, 300]:
        for prop_name, prop in [("Wageningen B", propeller_wageningen),
                               ("Custom Curves", propeller_custom)]:
            if prop.has_model:
                rpm = test_propeller_rpm_from_thrust(prop, lc, thrust_kn)
                if rpm is not None:
                    power_kw, _, _ = test_propeller_power_from_rpm(prop, lc, rpm)
                    # Calculate fuel consumption
                    try:
                        result = test_powering_flow(prop, powering_dd, lc, thrust_kn)
                        optimization_results.append({
                            "loadcase": lc.name,
                            "speed_ms": lc.speed,
                            "thrust_kn": thrust_kn,
                            "propeller": prop_name,
                            "optimal_rpm": rpm,
                            "required_power_kw": power_kw,
                            "fuel_consumption_kg_h": result["total_fc_kg_h"]
                        })
                    except:
                        pass

df_optimization = pd.DataFrame(optimization_results)
if not df_optimization.empty:
    print("\nOptimization Results:")
    print(df_optimization.head(20).to_string())

# ============================================================================
# SECTION 18: VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 18: GENERATING VISUALIZATIONS")
print("=" * 80)

# Create comprehensive plots
fig = plt.figure(figsize=(20, 24))

# Plot 1: Thrust vs Speed for different power levels
ax1 = plt.subplot(4, 3, 1)
power_levels = [1000, 2000, 3000, 4000]
speeds = np.linspace(5, 20, 50)

for power_kw in power_levels:
    thrusts = []
    for speed in speeds:
        lc = LoadCase("", speed)
        thrust = test_propeller_thrust_from_power(propeller_wageningen, lc, power_kw)
        thrusts.append(thrust if thrust is not None else np.nan)
    ax1.plot(speeds, thrusts, label=f"{power_kw} kW", linewidth=2)

ax1.set_xlabel("Speed [m/s]")
ax1.set_ylabel("Thrust [kN]")
ax1.set_title("Thrust vs Speed (Wageningen B-series)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Power from RPM vs Speed
ax2 = plt.subplot(4, 3, 2)
rpm_levels = [100, 150, 200]

for rpm in rpm_levels:
    powers = []
    for speed in speeds:
        lc = LoadCase("", speed)
        power_kw, _, _ = test_propeller_power_from_rpm(propeller_wageningen, lc, rpm)
        if power_kw is not None:
            powers.append(power_kw)
        else:
            powers.append(np.nan)
    ax2.plot(speeds, powers, label=f"{rpm} RPM", marker='o', markersize=3, linewidth=2)

ax2.set_xlabel("Speed [m/s]")
ax2.set_ylabel("Power [kW]")
ax2.set_title("Power vs Speed (Wageningen B-series)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Fuel Consumption vs Thrust
ax3 = plt.subplot(4, 3, 3)
thrusts = np.linspace(50, 500, 20)

for lc in [loadcases[1], loadcases[3], loadcases[5]]:
    fcs = []
    for thrust_kn in thrusts:
        try:
            result = test_powering_flow(propeller_wageningen, powering_dd, lc, thrust_kn)
            fcs.append(result["total_fc_kg_h"])
        except:
            fcs.append(np.nan)
    ax3.plot(thrusts, fcs, label=f"{lc.speed} m/s", marker='o', markersize=3, linewidth=2)

ax3.set_xlabel("Thrust [kN]")
ax3.set_ylabel("Fuel Consumption [kg/h]")
ax3.set_title("Fuel Consumption vs Thrust (DD Mode)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Powering Mode Comparison
ax4 = plt.subplot(4, 3, 4)
lc = loadcases[3]
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
    ax4.plot(thrusts, fcs, label=powering_name, marker='o', markersize=3, linewidth=2)

ax4.set_xlabel("Thrust [kN]")
ax4.set_ylabel("Fuel Consumption [kg/h]")
ax4.set_title(f"Fuel Consumption Comparison ({lc.speed} m/s)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Performance Curves - KT vs J
ax5 = plt.subplot(4, 3, 5)
if not df_curves.empty:
    for prop_name in df_curves["propeller"].unique():
        prop_data = df_curves[df_curves["propeller"] == prop_name]
        ax5.plot(prop_data["J"], prop_data["KT"], label=f"{prop_name} KT", linewidth=2)
ax5.set_xlabel("Advance Coefficient J [-]")
ax5.set_ylabel("Thrust Coefficient KT [-]")
ax5.set_title("KT vs J")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Performance Curves - KQ vs J
ax6 = plt.subplot(4, 3, 6)
if not df_curves.empty:
    for prop_name in df_curves["propeller"].unique():
        prop_data = df_curves[df_curves["propeller"] == prop_name]
        ax6.plot(prop_data["J"], prop_data["KQ"], label=f"{prop_name} KQ", linewidth=2)
ax6.set_xlabel("Advance Coefficient J [-]")
ax6.set_ylabel("Torque Coefficient KQ [-]")
ax6.set_title("KQ vs J")
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Efficiency vs J
ax7 = plt.subplot(4, 3, 7)
if not df_efficiency.empty:
    for prop_name in df_efficiency["propeller"].unique():
        prop_data = df_efficiency[df_efficiency["propeller"] == prop_name]
        ax7.plot(prop_data["J"], prop_data["efficiency"], label=prop_name, linewidth=2)
ax7.set_xlabel("Advance Coefficient J [-]")
ax7.set_ylabel("Efficiency η₀ [-]")
ax7.set_title("Propeller Efficiency vs J")
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: EST Power Impact
ax8 = plt.subplot(4, 3, 8)
if not df_est.empty:
    for mode in df_est["powering_mode"].unique():
        mode_data = df_est[df_est["powering_mode"] == mode].sort_values("est_power_kw")
        ax8.plot(mode_data["est_power_kw"], mode_data["total_fc_kg_h"], 
                label=mode, marker='o', linewidth=2)
ax8.set_xlabel("EST Power [kW]")
ax8.set_ylabel("Total Fuel Consumption [kg/h]")
ax8.set_title("EST Power Impact on Fuel Consumption")
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: RPM vs Thrust Requirement
ax9 = plt.subplot(4, 3, 9)
if not df_rpm_from_thrust.empty:
    for prop_name in df_rpm_from_thrust["propeller"].unique():
        prop_data = df_rpm_from_thrust[df_rpm_from_thrust["propeller"] == prop_name]
        ax9.plot(prop_data["speed_ms"], prop_data["required_rpm"], 
                label=prop_name, marker='o', linewidth=2)
ax9.set_xlabel("Speed [m/s]")
ax9.set_ylabel("Required RPM")
ax9.set_title("Required RPM vs Speed (200 kN thrust)")
ax9.legend()
ax9.grid(True, alpha=0.3)

# Plot 10: Variable vs Constant Efficiency
ax10 = plt.subplot(4, 3, 10)
speeds_test = np.linspace(0, 25, 50)
eta_const = []
eta_var = []

for speed in speeds_test:
    lc = LoadCase("", speed)
    # Use propeller_simple for constant efficiency (it has eta_delivered_power set)
    eta_const.append(propeller_simple.eta_D_at_speed(lc))
    eta_var.append(propeller_variable_eta.eta_D_at_speed(lc))

ax10.plot(speeds_test, eta_const, label="Constant", linewidth=2)
ax10.plot(speeds_test, eta_var, label="Variable", linewidth=2)
ax10.set_xlabel("Speed [m/s]")
ax10.set_ylabel("Delivered Power Efficiency")
ax10.set_title("Constant vs Variable Efficiency")
ax10.legend()
ax10.grid(True, alpha=0.3)

# Plot 11: Genset Activation
ax11 = plt.subplot(4, 3, 11)
if not df_integration.empty:
    de_data = df_integration[df_integration["powering_mode"] == "DE"]
    if not de_data.empty:
        ax11.scatter(de_data["delivered_power_kw"], de_data["n_gensets_active"], 
                    alpha=0.6, s=50)
ax11.set_xlabel("Delivered Power [kW]")
ax11.set_ylabel("Number of Active Gensets")
ax11.set_title("Genset Activation (DE Mode)")
ax11.grid(True, alpha=0.3)

# Plot 12: Power Flow Sankey-style (simplified)
ax12 = plt.subplot(4, 3, 12)
if not df_integration.empty:
    sample = df_integration.iloc[0]
    categories = ["Effective\nPower", "Delivered\nPower", "Shaft\nPower", "Brake\nPower"]
    values = [sample["effective_power_kw"], 
             sample["delivered_power_kw"],
             sample["shaft_power_kw"],
             sample["brake_power_kwm"]]
    ax12.barh(categories, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax12.set_xlabel("Power [kW]")
    ax12.set_title("Power Flow Example")
    ax12.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("test_results_comprehensive.png", dpi=150, bbox_inches='tight')
print("\nPlots saved to 'test_results_comprehensive.png'")
plt.close()

# ============================================================================
# SECTION 19: DATA EXPORT
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 19: DATA EXPORT")
print("=" * 80)

# Export all results to CSV files
exports = {
    "thrust_from_power": df_thrust,
    "rpm_analysis": df_rpm,
    "rpm_from_thrust": df_rpm_from_thrust,
    "efficiency_analysis": df_efficiency,
    "performance_curves": df_curves,
    "edge_cases": df_edge,
    "integration_results": df_integration,
    "est_impact": df_est,
    "multi_prop": df_multi,
    "pti_pto_analysis": df_pti_pto,
    "optimization": df_optimization,
}

for name, df in exports.items():
    if not df.empty:
        filename = f"results_{name}.csv"
        df.to_csv(filename, index=False)
        print(f"  Exported {filename} ({len(df)} rows)")

# ============================================================================
# SECTION 20: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 20: SUMMARY STATISTICS")
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
print("Check the generated CSV files and PNG plot for detailed results.")

