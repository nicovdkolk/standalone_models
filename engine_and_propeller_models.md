# From Engine Setting to Effective Power  
## Reference Model and Software Interfaces

## 1. Introduction

This document describes the **power flow** from a ship’s main engine to the **effective power** required to tow the hull through the water. Effective power is defined as

\[
P_E = R \, V,
\]

where:

- \(R\) is the total resistance acting on the ship (ship-fixed frame),
- \(V\) is the ship’s speed through the water.

- A physically consistent description of the power chain, and  
- A **software-oriented interface view** of engine and propeller models (forward and backward forms).

This document focuses on **interfaces, inputs/outputs, and model behavior**.

### 1.1 Scope and assumptions

We consider a conventional, mechanically driven, single-screw ship in:

- **Steady-state / quasi-static conditions**
  - No strong transients, no rapid manoeuvring.
  - Time-averaged behaviour over time scales where inertial terms can be neglected.

- **Self-propelled, deep-water conditions**
  - Propeller behind the hull, fully submerged.
  - No explicit modelling of shallow-water or bank effects (these can be treated via modified resistance).

- **Classical mechanical drivetrain**
  - Main engine → (optional gearbox) → shaft line → fixed-pitch propeller.
  - Current implementation assumes gear ratio \(i = 1\). Non-unity gear ratios are conceptually straightforward extensions.

Other configurations (diesel-electric, podded drives, CPP with active pitch control, hybrid systems) share the same physical ideas but are not treated in detail.

### 1.2 Key power levels and speeds

We use the following main power levels:

- **Brake power \(P_B\)**  
  Mechanical power at the engine crankshaft:
  \[
  P_B = 2\pi N Q,
  \]
  where \(N\) is engine rotational speed and \(Q\) crankshaft torque.

- **Shaft power \(P_S\)**  
  Power at the propeller shaft **after gearbox losses**:
  \[
  P_S = \eta_{	ext{gearbox}} \, P_B.
  \]

- **Delivered power \(P_D\)**  
  Power delivered to the propeller at the propeller plane, after additional shaft-line losses:
  \[
  P_D = \eta_{	ext{shaft}} \, P_S
      = \eta_{	ext{shaft}} \, \eta_{	ext{gearbox}} \, P_B.
  \]
  In many cases \(P_S pprox P_D\), but conceptually they are distinct.

- **Thrust power \(P_T\)**  
  Rate at which propeller thrust does work on the inflow:
  \[
  P_T = T \, V_A,
  \]
  where \(T\) is propeller thrust and \(V_A\) is inflow speed at the propeller plane.

- **Effective power \(P_E\)**  
  Minimum power required to tow the hull at speed \(V\) against total resistance \(R\):
  \[
  P_E = R V.
  \]

We distinguish:

- **Ship speed \(V\)** – speed of the ship through undisturbed water.  
- **Propeller inflow speed \(V_A\)** – average water speed into the propeller disc.

They are linked via the **wake fraction** \(w\):

\[
V_A = V (1 - w).
\]

The drivetrain mechanical efficiency is

\[
\eta_M = rac{P_D}{P_B}
       = \eta_{	ext{gearbox}} \, \eta_{	ext{shaft}},
\]

and in the implementation this is referred to as **transmission efficiency** \(\eta_{	ext{Tr}}\), with \(\eta_M = \eta_{	ext{Tr}}\).

---

## 2. Power Flow Overview

### 2.1 Power levels from engine to effective power

Conceptually, the chain is:

```text
Fuel energy → Engine → Brake power P_B
           → Gearbox → Shaft power P_S
           → Shaft line → Delivered power P_D
           → Propeller → Thrust power P_T
           → Hull–propeller interaction → Effective power P_E = R · V
           → Overcoming resistance → Ship motion at speed V
```

At each stage:

1. **Engine**
   - Converts chemical energy to mechanical power (brake power \(P_B\)).
   - Losses: combustion irreversibilities, heat, internal friction.

2. **Mechanical transmission**
   - **Gearbox stage**: \(P_B 	o P_S = \eta_{	ext{gearbox}} P_B\).  
   - **Shaft stage**: \(P_S 	o P_D = \eta_{	ext{shaft}} P_S\).  
   - Combined efficiency \(\eta_M = \eta_{	ext{gearbox}} \eta_{	ext{shaft}}\).

3. **Propeller hydrodynamics**
   - Converts delivered torque and rotation to thrust \(T\) and thrust power \(P_T = T V_A\).
   - Losses: blade drag, tip vortices, rotational energy left in the slipstream, etc.

4. **Hull–propeller–environment interaction**
   - Propeller thrust balances total resistance \(R\) (accounting for interaction).
   - Effective power \(P_E = R V\) is the useful hydrodynamic power.

From an energy viewpoint:

- \(P_B\) is the mechanical power output of the engine.
- \(P_E\) is the useful hydrodynamic power needed to sustain speed \(V\).

The difference between them is captured by mechanical and hydrodynamic efficiency factors.

### 2.2 Propulsive coefficients and overall efficiency

We use the standard propulsive coefficients:

1. **Open-water efficiency \(\eta_0\)**  
   Efficiency of the propeller in open water:
   \[
   \eta_0 = rac{P_T}{P_D} = rac{T V_A}{P_D}.
   \]

2. **Hull efficiency \(\eta_H\)**  
   Accounts for wake fraction \(w\) and thrust deduction \(t\). With
   \[
   V_A = V(1 - w), \quad R = (1 - t) T,
   \]
   we obtain
   \[
   \eta_H = rac{P_E}{P_T}
          = rac{R V}{T V_A}
          = rac{1 - t}{1 - w}.
   \]

3. **Relative rotative efficiency \(\eta_R\)**  
   Corrects for the difference between behind-hull and open-water performance. Typically close to unity and often treated as a constant.

4. **Mechanical / transmission efficiency \(\eta_M = \eta_{	ext{Tr}}\)**  
   As defined above:
   \[
   \eta_M = \eta_{	ext{Tr}} = rac{P_D}{P_B}
          = \eta_{	ext{gearbox}} \, \eta_{	ext{shaft}}.
   \]

The overall quasi-propulsive efficiency (or overall propulsive coefficient) is

\[
\eta_D = rac{P_E}{P_B}
       = \eta_M \, \eta_0 \, \eta_R \, \eta_H.
\]

---

## 3. Physical Sub-Models

### 3.1 Engine (physical view)

The **engine model** relates:

- Control inputs (e.g. fuel index, governor demand),
- Engine speed \(N\),

to:

- Brake torque \(Q\),
- Brake power \(P_B = 2\pi N Q\),
- Fuel consumption, often expressed via SFOC (kg/kWh).

At a high level:

- The engine has an **operating envelope** (limits on torque, power, speed, temperatures, etc.).
- Within this envelope, a control setting determines an admissible operating point and corresponding fuel flow.
- Engine efficiency \(\eta_{	ext{eng}}\) relates fuel energy rate \(\dot{E}_{	ext{fuel}}\) to brake power:
  \[
  P_B = \eta_{	ext{eng}} \, \dot{E}_{	ext{fuel}}.
  \]

### 3.2 Mechanical transmission (gearbox and shaft line)

The **mechanical transmission** converts brake power \(P_B\) to delivered power \(P_D\) with efficiency \(\eta_M\).

- **Gearbox**
  - Changes speed and torque according to gear ratio \(i\) (currently \(i = 1\)).
  - Efficiency \(\eta_{	ext{gearbox}} < 1\) accounts for friction and meshing losses.

- **Shaft line**
  - Transmits power from gearbox to propeller.
  - Efficiency \(\eta_{	ext{shaft}} \lesssim 1\) accounts for bearing, seal, and alignment losses.

Combined:

\[
P_S = \eta_{	ext{gearbox}} P_B, \quad
P_D = \eta_{	ext{shaft}} P_S = \eta_M P_B.
\]

The implementation treats \(\eta_M\) as a configurable **transmission efficiency** parameter.

### 3.3 Propeller hydrodynamics

The propeller is described through standard non-dimensional relations:

- **Advance ratio**
  \[
  J = rac{V_A}{n D},
  \]
  where \(n\) is propeller speed in revolutions per second and \(D\) is diameter.

- **Thrust and torque coefficients**
  \[
  T = ho n^2 D^4 K_T(J), \quad
  Q = ho n^2 D^5 K_Q(J),
  \]
  where \(K_T\) and \(K_Q\) are provided from series data or fitted curves, and \(ho\) is water density.

From these:

\[
P_D = 2\pi n Q, \quad
P_T = T V_A.
\]

The physical propeller model thus maps:

- Inputs: \(n, V_A\),
- Outputs: \(T, Q, P_D, P_T\).

### 3.4 Hull resistance and effective power

The **hull model** describes how total resistance \(R\) depends on:

- Speed \(V\),
- Draft, trim,
- Environmental conditions (waves, wind, shallow water, fouling, etc.).

Conceptually:

\[
R = Rigl(V, 	ext{draft}, 	ext{trim}, 	ext{environment}igr).
\]

We do not prescribe a specific resistance method here (Holtrop–Mennen, series, data-driven, etc.), only that the model provides a consistent mapping between \(V\) and \(R\). Effective power follows from:

\[
P_E = R V.
\]

### 3.5 Hull–propeller interaction

Hull and propeller interact via:

- **Wake fraction \(w\)**
  \[
  V_A = V (1 - w),
  \]
  representing the velocity deficit in the wake.

- **Thrust deduction factor \(t\)**
  \[
  R = (1 - t) T \quad \Rightarrow \quad T = rac{R}{1 - t},
  \]
  representing how much thrust is “lost” in modifying the pressure field around the hull.

Combined, they define hull efficiency:

\[
\eta_H = rac{P_E}{P_T}
       = rac{1 - t}{1 - w}.
\]

In practice, \(w\) and \(t\) may be modelled as functions of speed, loading, and geometry, or taken from empirical curves.

---

## 4. Implementation View: Forward and Backward Models

This section describes high-level **software interfaces** for the main sub-models:

- Engine (forward and backward),
- Propeller (forward and backward),
- System-level coupling.

### 4.1 Engine models

#### 4.1.1 Forward engine model

**Purpose**

Given an engine control input and speed, compute brake power, torque, fuel consumption, and diagnostics, subject to the engine envelope.

**Inputs**

- Engine control signal (e.g. relative load or fuel index).  
- Engine speed \(N\).  
- Engine capability and envelope data (maps for maximum torque/power vs speed).  
- SFOC map (kg/kWh) as function of speed and load.  
- Engine operating mode (e.g. diesel, dual-fuel).

**Outputs**

- Brake torque \(Q_{	ext{eng}}\).  
- Brake power \(P_B\).  
- SFOC at the operating point.  
- Fuel mass flow \(\dot{m}_f\).  
- Diagnostic flags (e.g. envelope limit violations, low-load operation).

**Assumptions**

- Quasi-static engine behaviour (no transient dynamics).  
- Engine maps represent steady-state performance.  
- Control signal is within [0, 1] or an equivalent normalized range.  

**Interface (high level)**

```python
def engine_forward(control, N, engine_map, sfoc_map, mode):
    """
    High-level forward engine model.

    Inputs:
        control: dimensionless engine control signal (e.g. relative load)
        N: engine speed
        engine_map: engine capability and operating envelope data
        sfoc_map: SFOC characteristics (kg/kWh)
        mode: engine operating mode (e.g. 'diesel', 'dual_fuel')

    Returns:
        P_B: brake power
        Q_eng: engine torque
        sfoc: specific fuel oil consumption (kg/kWh)
        m_dot_fuel: total fuel mass flow
        flags: diagnostic information (e.g. limit violations)
    """
    pass
```

---

#### 4.1.2 Backward engine model

**Purpose**

Given a required brake power or torque (e.g. from the propeller/shaft model) and constraints, determine an admissible engine operating point (speed, control) and associated fuel consumption.

**Inputs**

- Required brake power \(P_B^{	ext{req}}\) or torque.  
- Constraints (e.g. speed range, load limits, controller strategy).  
- Engine capability and envelope data.  
- SFOC map (kg/kWh).  
- Engine operating mode.

**Outputs**

- Selected engine speed \(N\).  
- Engine control setting (e.g. relative load, fuel index).  
- Achievable brake power \(P_B\).  
- SFOC and fuel mass flow \(\dot{m}_f\).  
- Feasibility/diagnostic flags (e.g. “power-limited”, “outside envelope”).

**Assumptions**

- Engine can operate anywhere inside the predefined envelope.  
- Control strategy (e.g. constant rpm vs variable rpm) is encoded via the `constraints` argument.  

**Interface (high level)**

```python
def engine_backward(P_B_req, constraints, engine_map, sfoc_map, mode):
    """
    High-level backward engine model.

    Inputs:
        P_B_req: required brake power (or equivalent torque requirement)
        constraints: engine and control strategy constraints
        engine_map: engine capability and operating envelope data
        sfoc_map: SFOC characteristics (kg/kWh)
        mode: engine operating mode (e.g. 'diesel', 'dual_fuel')

    Returns:
        N: selected engine speed
        control: engine control setting (e.g. relative load)
        P_B: achievable brake power
        sfoc: specific fuel oil consumption (kg/kWh)
        m_dot_fuel: total fuel mass flow
        flags: diagnostic information (e.g. infeasible, power-limited)
    """
    pass
```

---

### 4.2 Propeller models

#### 4.2.1 Forward propeller model

**Purpose**

Given propeller speed and inflow speed, compute thrust, torque, delivered power, and thrust power.

**Inputs**

- Propeller speed \(n\).  
- Inflow speed \(V_A\).  
- Propeller geometry (e.g. diameter \(D\), number of blades, pitch, expanded area ratio).  
- Series data or fitted curves for \(K_T(J)\) and \(K_Q(J)\).  
- Fluid properties (density \(ho\)).

**Outputs**

- Thrust \(T\).  
- Torque \(Q_{	ext{prop}}\).  
- Delivered power \(P_D\).  
- Thrust power \(P_T\).  
- Derived quantities (e.g. open-water efficiency \(\eta_0\)).

**Assumptions**

- Fixed-pitch propeller in non-cavitating regime.  
- Steady inflow characterised by a single effective advance speed \(V_A\).  
- Behind-hull corrections (via \(\eta_R\)) handled at system level or via modified curves.

**Interface (high level)**

```python
def propeller_forward(n, V_A, geometry, series_data, fluid_props):
    """
    High-level forward propeller model.

    Inputs:
        n: propeller rotational speed
        V_A: inflow speed at the propeller plane
        geometry: propeller geometry parameters (D, pitch, etc.)
        series_data: propeller series data or fitted KT/KQ curves
        fluid_props: fluid properties (e.g. density)

    Returns:
        T: propeller thrust
        Q_prop: propeller torque
        P_D: delivered power at the propeller
        P_T: thrust power
        extras: optional derived quantities (e.g. eta_0)
    """
    pass
```

---

#### 4.2.2 Backward propeller model

**Purpose**

Given a required thrust (from hull resistance and interaction) and inflow speed, determine an admissible propeller speed and corresponding load.

**Inputs**

- Required thrust \(T^{	ext{req}}\).  
- Inflow speed \(V_A\).  
- Propeller geometry and series data.  
- Fluid properties.  
- Operational constraints (e.g. allowable rpm range, cavitation/strength envelopes).

**Outputs**

- Selected propeller speed \(n\).  
- Resulting torque \(Q_{	ext{prop}}\).  
- Delivered power \(P_D\).  
- Thrust power \(P_T\).  
- Diagnostic flags (e.g. “thrust not attainable within rpm range”).

**Assumptions**

- Monotonic relationship between thrust and rpm over the operating region.  
- Fixed pitch; for CPP, pitch variation would be an additional degree of freedom.  

**Interface (high level)**

```python
def propeller_backward(T_req, V_A, geometry, series_data, fluid_props, constraints):
    """
    High-level backward propeller model.

    Inputs:
        T_req: required propeller thrust
        V_A: inflow speed at the propeller plane
        geometry: propeller geometry parameters (D, pitch, etc.)
        series_data: propeller series data or fitted KT/KQ curves
        fluid_props: fluid properties (e.g. density)
        constraints: propeller operating constraints (rpm range, etc.)

    Returns:
        n: selected propeller speed
        Q_prop: propeller torque
        P_D: delivered power at the propeller
        P_T: thrust power
        flags: diagnostic information (e.g. thrust infeasible)
    """
    pass
```

---

### 4.3 System-level coupling and solving strategies

At system level we compose:

- Engine model (forward or backward).  
- Transmission model (mechanical efficiency and gear ratio).  
- Propeller model (forward or backward).  
- Hull resistance and interaction models (R–V relation, \(w\), \(t\)).  

**Forward (engine-driven) mode**

Inputs: engine control, environment and hull condition.  
Outputs: operating point (speed, thrust, powers, fuel flow).

```python
def solve_forward_operating_point(engine_ctrl, env, hull_model,
                                  propeller_model, interaction_model,
                                  transmission, constraints):
    """
    Solve forward operating point: from engine control to ship speed and fuel.

    Inputs:
        engine_ctrl: engine control inputs (e.g. N setpoint, relative load)
        env: environment and loading state (draft, trim, metocean, hull condition)
        hull_model: interface to R(V, env)
        propeller_model: interface to propeller_forward
        interaction_model: interface providing w(V, ...) and t(V, ...)
        transmission: mechanical transmission parameters (eta_M, gear ratio)
        constraints: system-level constraints (e.g. bounds on speed, rpm)

    Returns:
        state: operating point (V, n, T, Q, P_B, P_D, P_E, fuel flow, flags)

    Notes:
        Uses forward engine and propeller models and enforces force and torque
        balance internally.
    """
    pass
```

**Backward (speed-driven) mode**

Inputs: target speed and environment.  
Outputs: engine operating point and fuel flow.

```python
def solve_backward_operating_point(target_speed, env, hull_model,
                                   propeller_model, engine_model,
                                   interaction_model, transmission,
                                   constraints):
    """
    Solve backward operating point: from target speed to engine setting.

    Inputs:
        target_speed: required ship speed through water
        env: environment and loading state (draft, trim, metocean, hull condition)
        hull_model: interface to R(V, env)
        propeller_model: interfaces to propeller_backward/forward as needed
        engine_model: interfaces to engine_backward/forward as needed
        interaction_model: interface providing w(V, ...) and t(V, ...)
        transmission: mechanical transmission parameters (eta_M, gear ratio)
        constraints: system-level constraints (e.g. engine load limits)

    Returns:
        state: operating point (N, control, n, T, Q, P_B, P_D, P_E, fuel flow, flags)

    Notes:
        Uses backward models to meet the required speed while respecting
        engine and propeller envelopes.
    """
    pass
```

---

## 5. Quasi-Static Equilibrium and Operating Point Determination

### 5.1 Force balance and torque balance

A steady operating point satisfies:

- **Force equilibrium (longitudinal)**
  \[
  R(V, 	ext{condition}) = (1 - t) T,
  \]
  where \(t\) is the thrust deduction factor.

- **Torque equilibrium (rotational)**
  \[
  Q_{	ext{eng}} = rac{Q_{	ext{prop}}}{\eta_M \, i},
  \]
  with gear ratio \(i\) (currently \(i = 1\)) and \(\eta_M = \eta_{	ext{Tr}}\).

- **Kinematic relations**
  \[
  V_A = V (1 - w), \quad
  J = rac{V_A}{n D}.
  \]

These equations are closed by sub-models:

- Engine model: \(Q_{	ext{eng}} = f_{	ext{eng}}(N, 	ext{control})\).  
- Propeller model: \(T, Q_{	ext{prop}} = f_{	ext{prop}}(n, V_A)\).  
- Hull model: \(R = f_{	ext{hull}}(V, 	ext{condition})\).  
- Interaction model: \(w, t = f_{	ext{int}}(V, T, 	ext{geometry})\).

### 5.2 Engine-driven (forward) problem

**Question**  
Given an engine setting (e.g. rpm and relative load), what speed and thrust result?

**Conceptual sequence**

1. Forward engine model → brake power \(P_B\), torque \(Q_{	ext{eng}}\).  
2. Transmission → delivered power \(P_D\), propeller shaft torque.  
3. Forward propeller model → thrust \(T\), torque \(Q_{	ext{prop}}\), with an estimate of \(V\) and \(V_A\).  
4. Hull resistance and interaction → iterate on \(V\) until
   \[
   R(V) = (1 - t(V)) T,
   \]
   and torque balance is satisfied.

Outputs: ship speed \(V\), thrust \(T\), \(P_E\), fuel flow, plus diagnostics.

### 5.3 Speed-driven (backward) problem

**Question**  
Given a target speed \(V\), what engine operating point is required?

**Conceptual sequence**

1. Hull model → resistance \(R(V)\) and effective power \(P_E = R V\).  
2. Interaction model → required thrust and inflow:
   \[
   T = rac{R}{1 - t(V)}, \quad
   V_A = V (1 - w(V)).
   \]
3. Backward propeller model → propeller speed \(n\), torque \(Q_{	ext{prop}}\), delivered power \(P_D\).  
4. Transmission → required brake power \(P_B = P_D / \eta_M\).  
5. Backward engine model → engine speed \(N\), control setting, achievable \(P_B\), SFOC, and fuel flow.

Outputs: engine rpm and load, fuel consumption, feasibility flags.

### 5.4 Mixed / implicit formulations

More advanced use cases treat multiple variables as unknowns simultaneously:

- \(V, n, N, T, Q_{	ext{eng}}, Q_{	ext{prop}}\), engine control, etc.

The model becomes a coupled system enforcing:

- Force balance,  
- Torque balance,  
- Engine envelope constraints,  
- Propeller characteristics,  
- Control policies (e.g. constant-rpm strategies).

These are solved implicitly (e.g. through multi-dimensional root finding) and are encapsulated in the `solve_forward_operating_point` and `solve_backward_operating_point` interfaces.

---

## 6. Fuel, LHV, SFOC, and Dual-Fuel Operation

This section connects the fuel-side description to brake power \(P_B\) and thus to the rest of the power-flow chain.

### 6.1 Fuel chemistry and lower heating value

For any fuel, the chemical energy rate is:

\[
\dot{E}_{	ext{fuel}} = \dot{m}_f \, H_L,
\]

where:

- \(\dot{m}_f\) is fuel mass flow [kg/s],
- \(H_L\) is lower heating value (LHV) [MJ/kg].

The engine converts this to brake power:

\[
P_B = \eta_{	ext{eng}} \, \dot{m}_f \, H_L,
\]

where \(\eta_{	ext{eng}}\) is overall engine efficiency (thermodynamic + internal mechanical).

Typical LHVs (order of magnitude):

- Marine Gas/Diesel Oil: \(H_L pprox 42	ext{–}43.5\ 	ext{MJ/kg}\).  
- Heavy Fuel Oil: \(H_L pprox 39	ext{–}41\ 	ext{MJ/kg}\).  
- LNG (mostly methane): \(H_L pprox 48	ext{–}50\ 	ext{MJ/kg}\).  
- Methanol: \(H_L pprox 20	ext{–}21\ 	ext{MJ/kg}\).  
- Ammonia: \(H_L pprox 18	ext{–}23\ 	ext{MJ/kg}\).

Density \(ho_f\) [kg/m\(^3\)] allows conversion to volumetric flows \(\dot{V}_f = \dot{m}_f / ho_f\) and volumetric energy density \(H_L ho_f\).

### 6.2 SFOC in kg/kWh and engine efficiency

SFOC (kg/kWh) is defined as fuel mass consumed per unit brake energy:

\[
	ext{SFOC} = rac{\dot{m}_f \cdot 3600}{P_B}
\quad [	ext{kg/kWh}].
\]

Rearranging:

\[
\dot{m}_f = rac{	ext{SFOC} \cdot P_B}{3600}.
\]

Combining with the definition of engine efficiency:

\[
P_B = \eta_{	ext{eng}} \, \dot{m}_f H_L
\quad \Rightarrow \quad
\eta_{	ext{eng}}
= rac{3600}{	ext{SFOC} \cdot H_L}.
\]

Key points:

- For a given fuel, lower SFOC implies higher \(\eta_{	ext{eng}}\).  
- For different fuels, SFOC must be interpreted together with \(H_L\).

In practice, each engine–fuel combination has an SFOC map over \((N, P_B)\) or similar, with a “sweet spot” region of minimum SFOC.

### 6.3 Integrating different fuels in the power-flow model

From the system point of view, changing fuel modifies:

1. Heating value \(H_L\).  
2. Density \(ho_f\).  
3. The SFOC map for the chosen engine–fuel mode.

The rest of the power chain (transmission, propeller, hull) is unchanged because it only sees \(P_B\).

For a given operating point:

1. Hydrodynamics and propulsive efficiency define required \(P_B\).  
2. Engine operating point \((N, P_B)\) is chosen within the envelope.  
3. Fuel model (SFOC map for the active fuel) gives \(\dot{m}_f\) and \(\dot{V}_f\).

Fuel switching between, e.g., HFO, LNG, methanol, or ammonia is represented by selecting the appropriate SFOC map and fuel properties.

### 6.4 Pilot fuels in dual-fuel operation

Dual-fuel engines use:

- A **main alternative fuel** (e.g. LNG, methanol, ammonia), and  
- A small amount of **pilot fuel** (typically MGO/MDO) for ignition.

Let:

- \(H_{L,	ext{main}}, H_{L,	ext{pilot}}\) be LHVs,  
- \(\dot{m}_{f,	ext{main}}, \dot{m}_{f,	ext{pilot}}\) mass flows.

Total chemical energy rate:

\[
\dot{E}_{	ext{fuel,total}}
= \dot{m}_{f,	ext{main}} H_{L,	ext{main}}
+ \dot{m}_{f,	ext{pilot}} H_{L,	ext{pilot}}.
\]

Brake power:

\[
P_B = \eta_{	ext{eng}} \, \dot{E}_{	ext{fuel,total}},
\]

with \(\eta_{	ext{eng}}\) now for dual-fuel mode.

We can define:

\[
	ext{SFOC}_{	ext{total}} =
rac{\dot{m}_{f,	ext{main}} + \dot{m}_{f,	ext{pilot}}}{P_B} \cdot 3600,
\]

and, optionally, separate:

\[
	ext{SFOC}_{	ext{main}} =
rac{\dot{m}_{f,	ext{main}}}{P_B} \cdot 3600, \quad
	ext{SFOC}_{	ext{pilot}} =
rac{\dot{m}_{f,	ext{pilot}}}{P_B} \cdot 3600.
\]

Pilot fractions in modern dual-fuel engines are typically a **few percent** of total fuel energy (order of 1–10%, technology-dependent).

### 6.5 Fuel interfaces (conceptual)

In the engine model interfaces, fuel modelling appears via:

```python
def fuel_properties(mode):
    """
    Return fuel properties for a given engine mode.

    Inputs:
        mode: engine operating mode (e.g. 'diesel', 'lng', 'methanol_dual_fuel')

    Returns:
        H_L: lower heating value for each active fuel stream
        rho_f: density for each active fuel stream
        emission_factors: optional emission factors per unit mass or energy

    Notes:
        Used by engine models and reporting layers.
    """
    pass
```

The forward/backward engine models then use SFOC maps and \(H_L\) to compute fuel mass flows for each active stream (main and pilot).

---

## 7. Assumptions, Limitations, and Possible Extensions

### 7.1 Key assumptions

- **Steady-state / quasi-static behaviour**
  - No explicit modelling of transients in ship motion or engine/propeller dynamics.

- **Single operating point**
  - The model describes equilibria, not transitions between them.

- **Simplified environment**
  - Wind, waves, currents, shallow water, and fouling are represented via effective resistance and possibly modified \(w\) and \(t\).

- **Single-screw, mechanically driven**
  - Multi-propeller or hybrid systems are not explicitly covered.

### 7.2 When assumptions break down

The quasi-static framework is less accurate when:

- Ship is manoeuvring aggressively (turns, crash stops, rapid acceleration).  
- Seaway is highly irregular and short-term transients dominate.  
- Control systems apply rapid changes to engine or propeller settings.

In those cases, dynamic models with explicit mass and inertia terms are required.

### 7.3 Possible extensions

Conceptual extensions include:

1. **Dynamic surge equation**
   \[
   m rac{dV}{dt} = (1 - t) T - R(V, t),
   \]
   with time-varying resistance and thrust.

2. **Rotational dynamics**
   \[
   I rac{d\omega}{dt} = Q_{	ext{eng}} - Q_{	ext{prop}},
   \]
   for engine/shaft/propeller inertia.

3. **Richer environmental models**
   - Explicit sea state, wind fields, shallow-water effects.

4. **Multiple propulsors / hybrid systems**
   - Multiple engines, propellers, electric drives, batteries, and power management logic.

5. **Advanced interaction models**
   - Non-uniform wake, rudder–propeller interaction, detailed CFD-based corrections.

All of these still obey the same underlying principles: power flows from fuel to effective power, and operating points are defined by force and torque equilibria.

---

## 8. Summary

This document provides a **reference model** for power flow from engine setting to effective power, together with high-level software interfaces:

- It defines the main power levels \(P_B, P_S, P_D, P_T, P_E\) and how they are connected through mechanical efficiency \(\eta_M\), open-water efficiency \(\eta_0\), relative rotative efficiency \(\eta_R\), and hull efficiency \(\eta_H\).

- It describes physical sub-models for engine, transmission, propeller, hull resistance, and hull–propeller interaction, including wake fraction \(w\) and thrust deduction \(t\).

- It introduces **forward and backward engine and propeller models** as software interfaces, with clearly defined inputs, outputs, assumptions, and roles in the overall solver.

- It outlines system-level coupling via `solve_forward_operating_point` and `solve_backward_operating_point`, which enforce force and torque balance to determine quasi-static operating points.

- It integrates **fuel modelling** (LHV, density, SFOC) and **dual-fuel/pilot fuel** behaviour at the engine boundary, without changing the mechanical and hydrodynamic structure of the model.

The detailed numerical algorithms and data structures are implemented elsewhere; this document is intended as a stable, shared reference for both modelling and software design discussions.
