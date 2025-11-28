<!-- 05927577-c839-4b7d-8bf9-d9136ef2fd16 282a8885-b6c9-4081-9bf4-c3cdf039ea0b -->
# Diesel-Electric Powering Architecture Implementation

## Overview

Extend Shipyard's powering model to support diesel-electric (DE) and PTI/PTO hybrid architectures, while establishing architectural foundations for future RPM-driven propulsion with Kt/Kq propeller models.

## Phase 1: Core Architecture & Abstraction Layer

### 1.1 Create Propulsion Mode Infrastructure

**File:** `Shipyard/src/core/models/propulsion_mode.py` (new)

- Create `PropulsionMode` enum: `SPEED_DRIVEN`, `POWER_DRIVEN`, `RPM_DRIVEN`
- Add mode detection logic and validation
- Document mode transition implications

### 1.2 Create Propeller Class

**File:** `Shipyard/src/core/forces/propeller.py` (new)

- Create base `Propeller` class with:
- Current implementation: Simple thrust model (thrust = Pe × η_D / speed)
- Interface design for future: `thrust_from_rpm(rpm, speed, wake_fraction)` using Kt(J)
- Interface design for future: `torque_from_rpm(rpm, speed, wake_fraction)` using Kq(J)
- Properties: diameter, number of propellers, wake fraction (w), thrust deduction (t)
- Add stub methods with clear TODOs for Kt/Kq implementation
- Design `PropellerCurve` class structure (not implemented yet) for Kt/Kq curves

### 1.3 Port PowerChain from Pelican1

**File:** `Shipyard/src/core/forces/power_chain.py` (new)

- Port `DieselEngine` class from Pelican1
- Port `PowerChain` class with full DE and PTI/PTO support
- Key methods:
- `fuel_consumption(wasp_power, shaft_power)` - current use
- `grid_load(grid_load, shaft_power)` - electrical grid balance
- `power_brake(other_consumers, shaft_power)` - ME brake power
- `n_gensets(aux_power)` - active genset count
- **NEW:** `max_brake_power()` - returns MCR/CSR for optimizer constraints (constraints imposed at brake power level)
- **NEW:** `brake_power_from_rpm(rpm)` - for future rpm mode, converts RPM to brake power
- **NEW:** `shaft_power_from_brake_power(pb)` - converts brake to shaft power through transmission chain
- Efficiency parameters: eta_C, eta_GB, eta_EM, eta_GEN, eta_PTIPTO
- Architecture flags: DE (diesel-electric), PTI_PTO

### 1.4 Update Hull Class

**File:** `Shipyard/src/core/forces/hull.py`

- Replace inline `Powering` class with import of `PowerChain`
- Keep `Powering` class as lightweight wrapper for backward compatibility
- Add `propeller` attribute using new `Propeller` class
- Update methods to use PowerChain:
- `propellor_thrust()` → delegate to `propeller.thrust()`
- `effective_power()`, `delivered_power()`, `brake_power()` → use propeller chain
- Add `shaft_power()` method (currently uses old efficiency chain)
- Add mode detection property: `propulsion_mode` (always SPEED_DRIVEN for now)

## Phase 2: API & Input Schema

### 2.1 Extend PoweringInput Schema

**File:** `Shipyard/src/api/inputs/designed_ship_input.py`

- Add fields to `PoweringInput`:
- `propulsion_mode: str | None` (default "speed_driven")
- `pti_pto: bool | None` (default False)
- `diesel_electric: bool | None` (default False)
- `eta_c: float | None` - Converter efficiency
- `eta_gb: float | None` - Gearbox efficiency
- `eta_s: float | None` - Shaft efficiency
- `eta_em: float | None` - Electric motor efficiency
- `eta_pti_pto: float | None` - PTI/PTO efficiency
- **Backward compatibility for eta_tr:**
- Keep `eta_tr` field for legacy ships (eta_tr = eta_GB × eta_S)
- If eta_tr provided but not eta_gb/eta_s: decompose using defaults (e.g., eta_GB=0.99, eta_S=eta_tr/0.99)
- If eta_gb and eta_s provided: compute eta_tr = eta_GB × eta_S for legacy compatibility
- Document that eta_tr is deprecated in favor of eta_gb and eta_s
- Add validation for mode-specific requirements
- Update `fill_default_values()` and `fill_from_reference_ship()`

### 2.2 Update Input Mapping

**File:** `Shipyard/src/api/routines/user_input_to_designed_ship_inputs.py` and related routines

- Follow Shipyard's existing pattern for input transformation (not Pelican1's legacy approach)
- Map new PoweringInput fields to PowerChain initialization dictionary
- Handle legacy powering format conversion (eta_tr decomposition)
- Set up propeller initialization with new fields
- Ensure proper camelCase ↔ snake_case conversion following existing conventions
- Integration with `store_digital_ship.py` workflow

## Phase 3: Output Variables & Postprocessing

### 3.1 Update Powering Output Variables

**File:** `Shipyard/src/postprocessing/output_variables/powering.py`

- Update `calculate_pb()` to use PowerChain methods
- Update `calculate_fc_me()` to handle DE mode (returns 0 when DE=True)
- Update `calculate_genset_fc()` to use PowerChain genset distribution
- Add new output variables:
- `grid_load_e` (kWe) - Electrical grid load
- `me_power_pct` (% MCR) - Main engine load percentage
- `gensets_active` (count) - Number of active gensets
- `propulsion_mode` (string) - Current mode indicator
- Ensure all calculations respect architecture flags (DE, PTI_PTO)

### 3.2 Add Architecture-Specific Outputs

**File:** `Shipyard/src/postprocessing/output_variables/powering.py`

- For PTI/PTO mode:
- `pti_pto_mode` - "PTO", "PTI", or "None"
- `pti_pto_power` - Power transferred through PTI/PTO
- For DE mode:
- `em_power` - Electric motor power
- `genset_loads` - Individual genset loads (array or dict)

## Phase 4: Testing & Validation

### 5.1 Unit Tests for PowerChain

**File:** `Shipyard/test/test_power_chain.py` (new)

- Test conventional mode (DE=False, PTI_PTO=False)
- Test diesel-electric mode (DE=True)
- Test PTI/PTO mode with PTO operation (below CSR)
- Test PTI/PTO mode with PTI operation (at CSR)
- Test genset distribution logic
- Test fuel consumption calculations
- Verify backward compatibility with old Powering class

### 5.2 Integration Tests

**File:** `Shipyard/test/test_hull_powering_integration.py` (new)

- Test Hull with PowerChain integration
- Test propeller class integration
- Test complete power flow: resistance → thrust → power → fuel consumption
- Compare results with Pelican1 reference cases

### 5.3 API Input/Output Tests

**File:** `Shipyard/test/test_powering_io.py` (new)

- Test PoweringInput validation
- Test GraphQL schema with new fields
- Test legacy format conversion
- Test output variable calculations for all modes

## Phase 6: Documentation & Migration

### 6.1 Technical Documentation

**File:** `Shipyard/documentation/powering_architecture.md` (new)

- Document diesel-electric architecture
- Document PTI/PTO operation modes
- Document propulsion mode framework
- Include power flow diagrams for each mode
- Document future RPM-driven mode design

### 6.2 Migration Guide

**File:** `Shipyard/documentation/powering_migration_guide.md` (new)

- Explain changes from old Powering to PowerChain
- Provide example configurations for each architecture
- List backward compatibility guarantees
- Document deprecated fields and their replacements

### 6.3 API Schema Updates

- Update GraphQL schema introspection
- Update API documentation with new powering fields
- Provide example queries for different architectures

## Key Design Principles

1. **Backward Compatibility:** All existing ships continue to work without modification
2. **Future-Ready:** Abstractions for RPM-driven mode in place, unused but documented
3. **Mode Separation:** Clear distinction between SPEED_DRIVEN (current), POWER_DRIVEN, and RPM_DRIVEN (future)
4. **Proven Implementation:** PowerChain logic ported directly from validated Pelican1 code
5. **Incremental Adoption:** Ships can opt-in to new features; no forced migration

## Future Work (Out of Scope)

- Kt/Kq propeller curve implementation
- RPM as state variable in optimizer
- Speed-solving mode with power/RPM constraints
- Battery/energy storage systems
- Multiple propulsion trains
- Dynamic PMS (power management system) logic

### To-dos

- [ ] Create PropulsionMode enum and mode detection infrastructure in core/models/propulsion_mode.py
- [ ] Create Propeller class with simple thrust model and Kt/Kq interface stubs in core/forces/propeller.py
- [ ] Port PowerChain and DieselEngine classes from Pelican1 to core/forces/power_chain.py with DE and PTI/PTO support
- [ ] Update Hull class to use PowerChain and Propeller, maintain backward compatibility with legacy Powering
- [ ] Extend PoweringInput schema with DE, PTI/PTO, and efficiency parameters in api/inputs/designed_ship_input.py
- [ ] Update input mapping to handle new PoweringInput fields and initialize PowerChain correctly
- [ ] Update powering output variables to use PowerChain methods and add architecture-specific outputs
- [ ] Add new powering fields to AlbatrosGUI JSON templates (designed_ship_template.json, configuration_inputs_template.json)
- [ ] Add UI controls for diesel-electric and PTI/PTO options in AlbatrosGUI Create Ship page
- [ ] Add new powering variables to AlbatrosGUI variable_names.py for proper display
- [ ] Create comprehensive unit tests for PowerChain covering conventional, DE, and PTI/PTO modes
- [ ] Create integration tests for Hull-PowerChain-Propeller interaction
- [ ] Create tests for PoweringInput validation and output variable calculations
- [ ] Write technical documentation and migration guide for new powering architecture