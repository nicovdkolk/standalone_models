# From Engine Setting to Effective Power: Physical Principles of Power Flow

## 1. Introduction

This document describes the **physical principles of power flow** from a ship’s main engine to the **effective power** required to tow the hull through the water, defined as

\[
P_E = R \cdot V,
\]

where:

- \(R\) = total resistance acting on the ship (in the ship-fixed frame),  
- \(V\) = ship speed through the water.

The focus is on **conceptual physics and model structure**, not on software or numerical implementation.

### 1.1 Scope and assumptions

We consider a conventional, mechanically driven, single-screw ship in:

- **Steady-state / quasi-static** operating conditions:
  - No strong transients, no rapid maneuvering.
  - Time-averaged behavior over timescales where acceleration terms are negligible.
- **Self-propelled, deep-water** condition:
  - The propeller is behind the hull, fully submerged.
  - No strong shallow-water or bank effects (though these can be treated as modifications to resistance).
- **Classical mechanical drivetrain**:
  - Main engine → (possibly gearbox) → shaft → propeller.

Other configurations (diesel-electric, podded drives, controllable-pitch propellers with active control, hybrid systems) exist, but the **physical ideas of power flow and balances** are similar. We focus on the classical picture for clarity.

### 1.2 Key quantities

We will refer to the following power levels:

- **Brake power \(P_B\)**  
  Mechanical power at the **engine crankshaft**, after fuel has been converted to torque and rotational motion.

- **Shaft power \(P_S\)**  
  Power transmitted through the **propeller shaft** after mechanical losses in the gearbox. Shaft power is related to brake power via gearbox efficiency:
  
  \[
  P_S = \eta_{	ext{gearbox}} \, P_B.
  \]

- **Delivered power \(P_D\)**  
  Power **delivered to the propeller** at the propeller plane. In many practical cases \(P_D pprox P_S\), but conceptually we distinguish them:

  \[
  P_D = \eta_{	ext{shaft}} \, P_S = \eta_{	ext{shaft}} \cdot \eta_{	ext{gearbox}} \, P_B.
  \]
  
  While \(P_S\) and \(P_D\) are often close in value, they are conceptually distinct: \(P_S\) is the power at the shaft, while \(P_D\) is the power actually delivered to the propeller after all shaft losses.

- **Thrust power \(P_T\)**  
  Rate at which the **propeller thrust does work** on the flow:

  \[
  P_T = T \cdot V_A,
  \]

  where \(T\) is propeller thrust and \(V_A\) is the **inflow speed** to the propeller.

- **Effective power \(P_E\)**  
  Minimum power required to tow the hull at speed \(V\) against total resistance \(R\):

  \[
  P_E = R \cdot V.
  \]

We also distinguish:

- **Ship speed \(V\)**: speed of the ship through the undisturbed water.  
- **Propeller inflow speed \(V_A\)**: average speed of the water entering the propeller disc (reduced by the hull wake).

These quantities are linked by **forces, power balances, and efficiency coefficients** that we will introduce step by step.

---

## 2. Overview of the Power Flow Chain

Conceptually, the power flow chain can be pictured as:

```text
Fuel energy → Engine → Brake power P_B
           → Mechanical transmission (gearbox) → Shaft power P_S
           → Shaft transmission → Delivered power P_D
           → Propeller (open water) → Thrust power P_T
           → Hull–propeller interaction → Effective power P_E = R · V
           → Overcoming resistance → Ship motion at speed V
```

At each stage:

1. **Engine:**  
   Chemical energy in fuel is converted to **mechanical power** at the crankshaft (brake power \(P_B\)). Losses include:
   - Combustion irreversibilities,
   - Heat losses,
   - Friction within the engine.

2. **Mechanical transmission (engine → propeller):**  
   The transmission occurs in two stages:
   
   - **Gearbox stage:** Converts \(P_B\) to **shaft power** \(P_S\) with efficiency \(\eta_{\text{gearbox}}\), accounting for losses in gear meshing, gearbox bearings, etc.
   
   - **Shaft stage:** Converts \(P_S\) to **delivered power** \(P_D\) with efficiency \(\eta_{\text{shaft}}\), accounting for losses in shaft bearings, couplings, seals, misalignment, etc.
   
   The overall transmission efficiency \(\eta_M = \eta_{\text{gearbox}} \cdot \eta_{\text{shaft}}\) reduces \(P_B\) to \(P_D\).

3. **Propeller hydrodynamics:**  
   The propeller converts delivered torque and rotation into a **pressure jump** across the propeller disc, generating thrust \(T\) on the water and reaction torque on the shaft. Losses include:
   - Blade drag,
   - Tip vortices,
   - Rotational kinetic energy left in the slipstream.

   The useful result is **thrust power** \(P_T = T V_A\).

4. **Hull–propeller–environment interaction:**  
   Propeller thrust accelerates the water in the ship’s wake and induces pressure fields around the hull, which experiences:
   - Total resistance \(R\) (viscous, wave-making, added resistance in waves, wind, etc.),
   - Additional interaction effects (e.g., suction near the stern).

   The resulting **effective power** \(P_E = R V\) is the power required to tow the hull at speed \(V\).

From the viewpoint of energy:

- \(P_B\) is the **mechanical output** of the engine.
- \(P_E\) is the **useful hydrodynamic power** needed to sustain speed \(V\) against resistance.

The **difference** between these levels is a combination of **mechanical** and **hydrodynamic** losses, expressible via propulsive efficiency coefficients.

---

## 3. Definitions and Power Levels

### 3.1 Brake power \(P_B\)

**Brake power** is the mechanical power at the **engine crankshaft**. For a rotating shaft:

\[
P_B = 2\pi N Q,
\]

where

- \(N\) = rotational speed (revolutions per second, or per minute with appropriate unit conversions),
- \(Q\) = torque at the crankshaft.

The engine model bridges between:

- **Control setting** (e.g. fuel index, governor setpoint, rpm)  
  and
- **Output torque/power** and fuel consumption.

Conceptually, **engine efficiency** converts fuel energy rate \(\dot{E}_	ext{fuel}\) to brake power:

\[
P_B = \eta_{	ext{eng}} \, \dot{E}_	ext{fuel},
\]

where \(\eta_{	ext{eng}}\) encapsulates all combustion and internal mechanical losses. We do **not** delve into combustion details here.

### 3.2 Shaft power \(P_S\) and delivered power \(P_D\)

**Shaft power** is the power transmitted through the shaft towards the propeller, after mechanical losses in:

- Gearbox,
- Couplings,
- Bearings,
- Shaft seals, etc.

We can define **mechanical efficiencies** for the drivetrain, split into two components:

\[
\eta_M = rac{P_D}{P_B},
\]

**Shaft power \(P_S\)** is the power transmitted through the propeller shaft after losses in the gearbox:

\[
P_S = \eta_{\text{gearbox}} \, P_B,
\]

where \(\eta_{\text{gearbox}}\) accounts for losses in gear meshing, gearbox bearings, and related mechanical components.

**Delivered power \(P_D\)** is the power actually delivered to the propeller at the propeller plane, after additional losses in the shaft transmission:

\[
P_D = \eta_{\text{shaft}} \, P_S = \eta_{\text{shaft}} \cdot \eta_{\text{gearbox}} \, P_B,
\]

where \(\eta_{\text{shaft}}\) accounts for losses in:
- Shaft bearings,
- Couplings,
- Shaft seals,
- Alignment losses, etc.

The overall **mechanical efficiency** for the drivetrain is:

\[
\eta_M = \frac{P_D}{P_B} = \eta_{\text{gearbox}} \cdot \eta_{\text{shaft}}.
\]

> **Note on terminology:** In the implementation, this overall mechanical efficiency is referred to as **transmission efficiency** and denoted as \(\eta_{\text{Tr}}\). The terms are equivalent: \(\eta_{\text{Tr}} = \eta_M = \eta_{\text{gearbox}} \cdot \eta_{\text{shaft}}\).

Typical values:
- \(\eta_{\text{gearbox}} \approx 0.95\text{–}0.98\) (depending on gear type and load),
- \(\eta_{\text{shaft}} \approx 0.98\text{–}0.99\) (typically very high),
- \(\eta_M \approx 0.93\text{–}0.97\) overall.

While \(P_S\) and \(P_D\) are often numerically close (since \(\eta_{\text{shaft}}\) is typically near unity), they represent distinct stages in the power flow chain:

- \(P_B\): at engine crankshaft,
- \(P_S\): at propeller shaft (after gearbox),
- \(P_D\): at propeller plane (after all shaft losses).
which relates delivered power to brake power:
\[
P_D = \eta_M \cdot P_B.
\]

Typical values of \(\eta_{\text{shaft}}\) and \(\eta_{\text{gearbox}}\) are less than 1 due to mechanical losses, with \(\eta_{\text{shaft}} \approx 0.99\) and \(\eta_{\text{gearbox}} \approx 0.97\) being representative values.

In many simplified models, distinctions between \(P_S\) and \(P_D\) are collapsed into a single transmission efficiency factor, but conceptually it is useful to keep:

- \(P_B\): at engine,
- \(P_S\): at shaft (after gearbox),
- \(P_D\): at propeller plane.

### 3.3 Thrust power \(P_T\)

The propeller exerts thrust \(T\) on the water. The **thrust power** is defined as:

\[
P_T = T \cdot V_A,
\]

where:

- \(T\) = propeller thrust (force),
- \(V_A\) = **inflow speed** of water at the propeller plane (average over the disc).

#### Ship speed vs inflow speed

Due to the **wake** created by the hull, the water arriving at the propeller moves **slower** than the undisturbed free-stream speed \(V\) of the ship. We define the **wake fraction** \(w\) via:

\[
V_A = V (1 - w).
\]

- \(V\): ship speed through water,
- \(w\): wake fraction; typically \(0 < w < 1\).

Physically, the hull drags water along, creating a velocity deficit in the stern region. The propeller then “sees” a lower inflow speed, which influences thrust and torque characteristics.

### 3.4 Effective power \(P_E\)

The **effective power** is the power required to pull the hull at speed \(V\) in the ship-fixed frame:

\[
P_E = R \cdot V,
\]

where \(R\) is the **total resistance**, including:

- Viscous (frictional + form),
- Wave-making,
- Appendage resistance,
- Added resistance due to wind, waves, shallow water, etc.

In a towing tank, \(P_E\) can be thought of as the power delivered by an ideal towing dynamometer that:

- Applies a steady tow force equal to \(R\),
- Moves at speed \(V\),
- Without any additional losses.

Thus, \(P_E\) is the **minimum hydrodynamic power** required for the given resistance and speed.

### 3.5 Propulsive coefficients

To connect power levels, we introduce **propulsive coefficients**, which express the efficiency of each stage.

Common conceptual coefficients:

1. **Open-water efficiency \(\eta_0\)**  
   Efficiency of the **propeller alone** in uniform inflow:

   \[
   \eta_0 = rac{P_T}{P_D} = rac{T V_A}{P_D}.
   \]

   In classical propeller theory, \(\eta_0\) is expressed in terms of thrust and torque coefficients and advance ratio, but here we keep it conceptual: it measures how effectively delivered power is converted into thrust power.

2. **Hull efficiency \(\eta_H\)**  
   Accounts for **hull–propeller interaction** through wake and thrust deduction:

   - **Wake fraction \(w\)**: relates \(V_A\) and \(V\),
   - **Thrust deduction factor \(t\)**: relates thrust \(T\) and resistance \(R\):

     \[
     R = (1 - t) T \quad \Rightarrow \quad T = rac{R}{1 - t}.
     \]

   Combining:

   \[
   \eta_H = rac{P_E}{P_T} = rac{R V}{T V_A}
          = rac{(1 - t) T \, V}{T (1 - w) V}
          = rac{1 - t}{1 - w}.
   \]

   This expresses how well thrust power is translated into effective power, considering the flow disturbances induced by the hull and propeller.

3. **Relative rotative efficiency \(\eta_R\)**  
   Accounts for the difference between propeller performance in **behind-hull conditions** and in open water, at the same inflow and loading. It captures three-dimensional wake and interaction effects. Conceptually:

   \[
   \eta_R pprox rac{P_{D,	ext{open water equivalent}}}{P_D}.
   \]

   For conceptual power-flow modeling, \(\eta_R\) is often taken as a constant near unity.

4. **Mechanical efficiency \(\eta_M\)** (also called **transmission efficiency \(\eta_{\text{Tr}}\)** in the implementation)  
   As introduced, this is the product of gearbox and shaft efficiencies:

   \[
   \eta_M = \eta_{\text{Tr}} = \frac{P_D}{P_B} = \eta_{\text{gearbox}} \cdot \eta_{\text{shaft}}.
   \]
   
   > **Note:** The implementation uses the term "transmission efficiency" (\(\eta_{\text{Tr}}\)) for this quantity, which is equivalent to the mechanical efficiency (\(\eta_M\)) used in this document.

Putting it all together, one can define an overall **quasi-propulsive coefficient** (or overall propulsive efficiency) as:

\[
\eta_D = rac{P_E}{P_B}
       = \eta_M \, \eta_0 \, \eta_R \, \eta_H.
   \]

This expresses **how much of the engine’s brake power is converted into effective power**.

---

## 4. Sub-Models and Physical Principles

We now describe the main **sub-models**: engine, propeller, hull/resistance, and hull–propeller interaction. Each is a physical relationship that links certain inputs and outputs.

### 4.1 Engine model

**Physical role:**  
The engine model maps an **engine control setting** (e.g. rpm setpoint, fuel index, load command) to:

- Brake torque \(Q\),
- Brake power \(P_B\),
- Fuel consumption and emissions (conceptually).

#### Forward-facing engine model

- **Inputs:**
  - Engine speed \(N\),
  - Control setting (e.g. fuel index, governor demand).
- **Outputs:**
  - Brake torque \(Q(N,	ext{setting})\),
  - Brake power \(P_B = 2\pi N Q\),
  - Fuel consumption \(\dot{m}_f(N,	ext{setting})\).

This **forward model** answers: *“Given an engine setting, what torque and power does the engine produce?”*

It must respect **engine operating limits**, e.g.:

- Maximum continuous rating (MCR),
- Torque and cylinder pressure limits,
- Maximum/minimum rpm,
- Smoke and emission constraints.

#### Backward-facing engine model

In many performance calculations, we instead know the **required shaft power** or torque and want to know:

- What engine setting is needed?
- Is this operating point allowed?

A **backward model** thus maps:

- **Inputs:**
  - Required shaft torque/power (after accounting for transmission),
- **Outputs:**
  - Engine speed and control setting,
  - feasibility (within operating envelope or not).

This is conceptually the inverse relationship of the forward model, respecting constraints like:

- Overload regions (too much torque at given rpm),
- Underload or low-load operation (risk of inefficiency or engine health issues).

### 4.2 Propeller model

The propeller is a rotating lifting surface in water. Standard propeller theory describes thrust and torque in terms of:

- **Advance ratio \(J\):**

  \[
  J = rac{V_A}{nD},
  \]

  where \(n\) = revolutions per second, \(D\) = propeller diameter.

- **Non-dimensional thrust and torque coefficients \(K_T(J)\), \(K_Q(J)\):**

  \[
  T = 
ho n^2 D^4 K_T(J), \quad
  Q = 
ho n^2 D^5 K_Q(J),
  \]

  where \(
ho\) is water density.

From these, delivered power is:

\[
P_D = 2\pi n Q = 2\pi 
ho n^3 D^5 K_Q(J),
\]

and thrust power is:

\[
P_T = T V_A = 
ho n^2 D^4 K_T(J) V_A.
\]

The **propeller model** therefore connects:

- **Inputs:**
  - Shaft speed \(n\),
  - Inflow speed \(V_A\),
- **Outputs:**
  - Thrust \(T\),
  - Torque \(Q\),
  - Delivered power \(P_D\),
  - Thrust power \(P_T\).

#### Levels of fidelity

Conceptually, propeller models can range from:

- **Series-based / empirical models:**
  - Use tabulated \(K_T(J)\), \(K_Q(J)\) data for a family of propellers.
  - Good for quick performance estimations.
- **Semi-empirical / lifting-line models:**
  - Represent blade loading and circulation more explicitly.
- **High-fidelity hydrodynamic models:**
  - Resolve flow around the propeller in detail.

For power-flow modeling, the key requirement is a **consistent relation** between \(n\), \(V_A\), \(T\), and \(Q\).

### 4.3 Hull and resistance model

The **hull model** describes how total resistance \(R\) depends on operating and environmental conditions.

#### Total resistance

Total resistance \(R\) is typically decomposed conceptually into:

- **Calm-water resistance:**
  - Frictional resistance (skin friction),
  - Form resistance (pressure drag),
  - Wave-making resistance,
  - Appendage resistance.
- **Added resistance:**
  - Due to wind (aerodynamic),
  - Due to waves (heave/pitch and wave scattering),
  - Due to shallow water and bank effects,
  - Due to fouling and roughness.

For power-flow modeling, we can treat resistance as a function:

\[
R = Rigl(V, 	ext{draft}, 	ext{trim}, 	ext{environment}igr),
\]

where “environment” is a compact representation of waves, wind, water depth, current (through relative speed), and hull condition.

#### Modeling approaches

Conceptually:

- **Empirical/series-based models:**
  - Use regression or classical series (e.g. Holtrop–Mennen type) to predict calm-water resistance.
- **Physics-based decompositions:**
  - Frictional resistance via boundary layer theory or ITTC-type formulations,
  - Residual (wave + viscous pressure) resistance via regression or simplified theory,
  - Explicit added resistance models for wind and waves.
- **Data-driven or hybrid models:**
  - Combine measured data, physics-informed features, and statistical or machine-learning components.

For the purposes of this document, the key point is that for any operating condition, we can **evaluate** or **model** a relationship:

\[
R \leftrightarrow V
\]

and thus compute effective power \(P_E = R V\).

### 4.4 Hull–propeller interaction

The propeller operates in the **wake field** of the hull, and the thrust it produces alters the flow around the stern. This interaction is captured through:

- **Wake fraction \(w\)**: relates inflow speed \(V_A\) to ship speed \(V\).
- **Thrust deduction factor \(t\)**: relates propeller thrust \(T\) to hull resistance \(R\).

#### Wake fraction \(w\)

As introduced:

\[
V_A = V (1 - w).
\]

Physical intuition:

- The hull drags and accelerates water in its boundary layer and wake.
- The propeller, located in this non-uniform flow, effectively experiences a lower inflow speed.
- Larger stern fullness or high block coefficients generally increase wake fraction.

The wake fraction influences:

- The **effective loading** of the propeller at a given ship speed,
- The relationship between shaft speed \(n\), ship speed \(V\), and thrust \(T\).

#### Thrust deduction factor \(t\)

The propeller induces pressure changes near the stern, which affect the force on the hull. The **thrust deduction factor** is defined by:

\[
R = (1 - t) T \quad \Rightarrow \quad T = rac{R}{1 - t}.
\]

Physical interpretation:

- Not all thrust goes into overcoming resistance; some is lost in modifying the pressure and velocity field around the hull.
- The hull experiences an additional suction or interaction force that means more thrust is required for the same net tow force.

#### Connection between thrust power and effective power

Using the definitions:

- \(P_T = T V_A\),
- \(P_E = R V\),
- \(V_A = V (1 - w)\),
- \(R = (1 - t) T\),

we obtain the **hull efficiency**:

\[
\eta_H = rac{P_E}{P_T} = rac{1 - t}{1 - w}.
\]

This expresses how hull–propeller interaction modifies the mapping between **thrust power** and **effective power**. Hull shape, propeller loading, and wake distribution all influence \(w\) and \(t\), and therefore \(\eta_H\).

---

## 5. Quasi-Static Equilibrium and Coupling

A steady operating point of the engine–propeller–hull system satisfies a set of **equilibrium conditions**. Conceptually:

1. **Force equilibrium (longitudinal):**

   - In the ship-fixed frame, the net force must be zero in steady state:

     \[
     	ext{Net force} = 0 \quad \Rightarrow \quad R = (1 - t) T.
     \]

   - Equivalently, we may say *“thrust balances resistance after interaction effects.”*

2. **Torque equilibrium (rotational):**

   - The **torque demanded by the propeller** at the shaft must equal the **torque delivered by the engine** (scaled by gear ratio and transmission efficiency):

     \[
     Q_{	ext{engine}} = rac{Q_{	ext{prop}}}{\eta_M \, i},
     \]

     where \(i\) is the gear ratio (if present), and \(\eta_M = \eta_{\text{shaft}} \cdot \eta_{\text{gearbox}}\) is the transmission efficiency.

     > **Note on gear ratio:** In the current implementation, the gear ratio \(i\) is assumed to be 1 (direct coupling between engine and propeller shaft). For systems with gear reduction or increase, \(i \neq 1\) and the relationship becomes \(Q_{\text{engine}} = Q_{\text{prop}} / (\eta_M \cdot i)\). The implementation can be extended to handle non-unity gear ratios if needed.

3. **Kinematic compatibility:**

   - Ship speed \(V\), propeller speed \(n\), and inflow speed \(V_A\) must satisfy:

     \[
     V_A = V(1 - w), \quad
     J = rac{V_A}{nD}.
     \]

4. **Model consistency (sub-model closures):**

   - Engine model: \(Q_{	ext{engine}} = Q_{	ext{eng}}(N, 	ext{setting})\),
   - Propeller model: \(T, Q = f_{	ext{prop}}(n, V_A)\),
   - Hull model: \(R = f_{	ext{hull}}(V, 	ext{condition})\),
   - Interaction model: \(w, t = f_{	ext{int}}(V, T, 	ext{geometry})\) (often treated as functions of speed and loading).

Conceptually, the **operating point** \((V, n, T, Q, 	ext{engine setting})\) is the solution of these coupled relationships where:

- **Longitudinal force balance** is satisfied,
- **Torque balance** is satisfied,
- Engine operating limits are respected.

An “iterative solver” is typically needed in practice, but the underlying principle is simply: **find a state where all physical sub-models agree simultaneously**.

---

## 6. Solving Strategies and Modeling Perspectives

Different modeling tasks emphasize different **inputs** and **outputs**. We outline three common conceptual perspectives: forward, backward, and mixed/implicit formulations.

### 6.1 Forward problem: from engine setting to speed and thrust

**Question:**  
*Given an engine setting (e.g. rpm and fuel index), what ship speed and thrust result?*

**Given:**

- Engine control setting (e.g. target rpm \(N\), load command).

**Conceptual steps:**

1. **Engine model → brake power/torque**

   - Using the forward engine model, compute:
     \[
     Q_{	ext{engine}}, \quad P_B = 2\pi N Q_{	ext{engine}}.
     \]

2. **Transmission → shaft power → delivered power**

   - Apply gearbox efficiency to obtain shaft power:
     \[
     P_S = \eta_{	ext{gearbox}} P_B.
     \]
   - Apply shaft efficiency to obtain delivered power:
     \[
     P_D = \eta_{	ext{shaft}} P_S = \eta_M P_B,
     \]
     where \(\eta_M = \eta_{	ext{gearbox}} \cdot \eta_{	ext{shaft}}\).
   - Given rpm and torque, we also know the propeller shaft torque \(Q_{	ext{prop}}\).

3. **Propeller model → thrust and thrust power**

   - With \(n = N / i\) and an initial estimate of \(V\) (or \(V_A\)), use the propeller model to compute:
     \[
     T, Q_{	ext{prop}}, P_T = T V_A.
     \]

4. **Force balance → ship speed**

   - Use the hull resistance model \(R(V)\) and interaction relations to find \(V\) such that:
     \[
     R(V) = (1 - t) T,
     \]
     with \(t\) and \(w\) evaluated at that \(V\) (and possibly load).

   - The operating speed \(V\) is the one where **available thrust matches required resistance** (after interaction).

**Outputs:**

- Ship speed \(V\),
- Thrust \(T\),
- Effective power \(P_E = R(V) V\),
- Induced engine load and fuel consumption.

This perspective is natural when simulating **what happens** when the crew or controller chooses certain rpm or load settings.

### 6.2 Backward problem: from required speed (or effective power) to engine setting

**Question:**  
*Given a target ship speed \(V\) (or effective power), what engine operating point is required?*

**Given:**

- Target speed \(V\) (e.g. schedule requirement),
- Ship condition and environment (draft, trim, wind/wave).

**Conceptual steps:**

1. **From speed to resistance and effective power**

   - Evaluate total resistance:
     \[
     R = R(V, 	ext{condition}),
     \]
   - Compute effective power:
     \[
     P_E = R \cdot V.
     \]

2. **From resistance to thrust and thrust power**

   - Use thrust deduction relation:
     \[
     T = rac{R}{1 - t},
     \]
   - Use wake fraction to obtain inflow velocity:
     \[
     V_A = V (1 - w),
     \]
   - Compute thrust power:
     \[
     P_T = T V_A.
     \]

3. **Through propeller model → shaft power and torque**

   - For a given propeller speed \(n\) (unknown), the propeller model must produce thrust \(T\) at inflow speed \(V_A\). This determines:
     \[
     Q_{	ext{prop}}, \quad P_D = 2\pi n Q_{	ext{prop}}.
     \]
   - Conceptually, we adjust \(n\) such that propeller thrust matches the required \(T\) for that inflow speed.

4. **Through engine model → engine setting**

   - With required shaft speed and torque, apply transmission efficiency and gear ratio to find brake torque and power:
     \[
     P_B = rac{P_D}{\eta_M};
     \]
     
     > **Note:** The gear ratio \(i\) is assumed to be 1 (direct coupling) in this formulation. For systems with gear reduction, the relationship would include the gear ratio: \(P_B = P_D / (\eta_M \cdot i)\).
   - Use the **backward engine model** to determine:
     - The control setting (fuel index, governor position, etc.),
     - Check whether the point lies within the engine envelope (e.g. not exceeding MCR).

**Outputs:**

- Required engine rpm and load,
- Brake power and fuel rate,
- Check of feasibility (e.g. whether target speed is attainable under given conditions).

This perspective is natural for **voyage planning, control design**, and energy efficiency assessments where the target is specified in terms of **speed or power**, and the engine must adjust to meet it.

### 6.3 Mixed / implicit formulations

In more complex scenarios, one may treat several key quantities as **simultaneous unknowns**:

- Engine rpm and load,
- Ship speed \(V\),
- Propeller thrust \(T\),
- Fuel consumption.

The model then consists of:

- Engine equilibrium: torque equality and engine map,
- Propeller characteristics: \(T, Q = f_{	ext{prop}}(n, V_A)\),
- Resistance and interaction: \(R = f_{	ext{hull}}(V), w(V), t(V)\),
- Force balance: \(R = (1 - t) T\).

These can be written as a **coupled system of equations** where the solution yields an operating point consistent with:

- Physical balances,
- Engine limits,
- Possibly controller policies (e.g. constant power or constant rpm strategies).

This approach is natural when modeling:

- Systems with multiple actuators (e.g. controllable pitch propeller plus engine control),
- Automatic controllers maintaining speed or power,
- Multi-objective constraints (e.g. speed, emissions, fuel limits).

In all cases, the underlying principle remains: **find a set of variables that simultaneously satisfies all physical sub-models and constraints.**

---

## 7. Assumptions, Limitations, and Extensions

### 7.1 Key assumptions

The conceptual framework presented relies on several simplifying assumptions:

1. **Steady-state / quasi-static behavior**

   - The ship and engine are modeled at a single operating point.
   - Acceleration/deceleration, maneuvering, and short-term unsteadiness are neglected.

2. **Single operating point**

   - We do not model transitions between states; only the equilibrium behavior at given conditions.

3. **Simplified environment**

   - Wind, waves, currents, shallow water, and hull fouling are included implicitly via **effective resistance** and possibly modified wake and thrust deduction.
   - Spatial and temporal variability of the environment is not resolved explicitly.

4. **Single-screw, mechanically driven system**

   - Generalization to multiple propellers, azimuthing thrusters, or complex hybrid systems is conceptually straightforward but not covered in detail here.

### 7.2 Physical implications

These assumptions work well when:

- The ship sails in **approximately steady conditions**, e.g. long straight legs of a voyage at near-constant speed and loading.
- Time scales of interest are long compared to engine and propeller dynamics, such that transients have decayed.

They break down when:

- The ship is **maneuvering aggressively** (turning, acceleration, crash stop),
- The seaway is highly irregular and **transients dominate**, e.g. in severe storm conditions,
- The control system changes settings rapidly (e.g. step changes in rpm or pitch).

In such cases, **dynamic models** including inertia and time-dependent responses of the engine and propeller are needed.

### 7.3 Conceptual extensions

Possible extensions of this framework, all still based on physical principles, include:

1. **Dynamic modeling:**

   - Include surge equation of motion:
     \[
     m rac{dV}{dt} = (1 - t) T - R(V, t),
     \]
   - Include engine and propeller rotational inertia:
     \[
     I rac{d\omega}{dt} = Q_{	ext{engine}} - Q_{	ext{prop}}.
     \]

2. **More detailed environmental models:**

   - Explicitly model waves and wind fields,
   - Treat added resistance and speed loss as functions of sea state and heading.

3. **Multiple propulsors / hybrid systems:**

   - Extend the power flow chain to multiple engines and propellers,
   - Include electric drives, batteries, and power management logic.

4. **Advanced hull–propeller interaction models:**

   - Spatially varying wake and non-uniform inflow,
   - Detailed modeling of rudder–propeller interaction.

All these extensions preserve the **core idea**: power flows from the engine through mechanical and hydrodynamic transformations, and steady operating points are defined by **equilibrium of forces and torques**.

---



## 8. Summary

This document has outlined the **power flow** from a ship’s main engine to **effective power**:

- **Brake power \(P_B\)** is produced by the engine at the crankshaft, as a function of engine control settings and operating conditions.
- Through the **mechanical transmission**, brake power \(P_B\) is first reduced to **shaft power \(P_S\)** via gearbox efficiency \(\eta_{\text{gearbox}}\), then further reduced to **delivered power \(P_D\)** via shaft efficiency \(\eta_{\text{shaft}}\). The overall mechanical efficiency (called transmission efficiency \(\eta_{\text{Tr}}\) in the implementation) is \(\eta_M = \eta_{\text{Tr}} = \eta_{\text{gearbox}} \cdot \eta_{\text{shaft}}\).
- The **propeller** converts delivered torque and rotation into **thrust \(T\)** and **thrust power \(P_T = T V_A\)**, with performance characterized by open-water efficiency \(\eta_0\) and relative rotative efficiency \(\eta_R\).
- **Hull–propeller interaction** modifies inflow speed \(V_A\) (via wake fraction \(w\)) and relates thrust \(T\) to hull resistance \(R\) (via thrust deduction \(t\)), yielding hull efficiency \(\eta_H = (1 - t)/(1 - w)\).
- The ship experiences **total resistance** \(R\) as a function of speed and condition, and the **effective power** is:

  \[
  P_E = R \cdot V,
  \]

  which is the minimum power needed to tow the ship at speed \(V\).

These stages are tied together by **quasi-static equilibrium conditions**:

- Longitudinal force balance: \(R = (1 - t) T\),
- Torque balance: engine torque equals propeller torque (through gear ratio and mechanical/transmission efficiency, with gear ratio \(i = 1\) assumed in the current implementation),
- Kinematic relations connecting ship speed, wake, and propeller inflow.

From a modeling perspective:

- The **forward problem** starts from an engine setting and predicts speed, thrust, and effective power.
- The **backward problem** starts from a desired speed or effective power and determines the required engine operating point.
- **Mixed or implicit formulations** solve for engine, propeller, and ship variables simultaneously as a coupled system.

Throughout, the focus is on **physical balances** and **sub-model coupling**. Implementation details, numerical methods, and software choices are deliberately left aside so that the core physics and modeling concepts remain clear and accessible to engineers and applied scientists working on ship performance.

## Fuel Chemistry, Calorific Value, SFOC (kg/kWh), and Pilot Fuels

This section connects the **fuel side** of the problem (chemistry, calorific value, SFOC, pilot fuels) to the previously described **power-flow chain** from brake power \(P_B\) to effective power \(P_E\).

---

### 1. Fuel Chemistry and Lower Heating Value

#### 1.1 Chemical energy rate

For any fuel, the rate at which **chemical energy** is supplied to the engine is

\[
\dot{E}_{\text{fuel}} = \dot{m}_f \, H_L,
\]

where:

- \(\dot{m}_f\) [kg/s] = fuel mass flow,
- \(H_L\) [MJ/kg] = lower heating value (LHV) of the fuel.

The engine converts this to brake power:

\[
P_B = \eta_{\text{eng}} \, \dot{E}_{\text{fuel}} = \eta_{\text{eng}} \, \dot{m}_f \, H_L,
\]

with \(\eta_{\text{eng}}\) the **overall engine efficiency** (thermodynamic + internal mechanical).

#### 1.2 Typical LHVs for marine fuels

Order-of-magnitude LHVs for relevant marine fuels are:

- **Conventional distillate and residual fuels**  
  - Marine Gas Oil (MGO) / Marine Diesel Oil: \(H_L \approx 42\text{–}43.5\ \text{MJ/kg}\).  
  - Heavy Fuel Oil (HFO): \(H_L \approx 39\text{–}41\ \text{MJ/kg}\).  

- **LNG (liquefied natural gas)**  
  - Dominated by methane; representative LHV \(H_L \approx 48\text{–}50\ \text{MJ/kg}\).  

- **Methanol**  
  - Typical LHV \(H_L \approx 20\text{–}21\ \text{MJ/kg}\).  

- **Ammonia**  
  - Reported LHVs of liquid ammonia are typically in the range \(18\text{–}23\ \text{MJ/kg}\); a commonly cited value for marine-fuel studies is about \(18\text{–}19\ \text{MJ/kg}\).  

These values are approximate and depend on composition, but they are sufficient for conceptual modeling.

#### 1.3 Fuel chemistry and carbon/hydrogen/oxygen/nitrogen content

Fuel chemistry affects both **energy per kilogram** and **emissions per unit energy**:

- Hydrocarbon oils (HFO, MGO) are mostly C and H:
  - Relatively high LHV per kg,
  - Significant CO\(_2\) emissions, roughly proportional to carbon content per unit energy.
- LNG (mostly methane, CH\(_4\)):
  - Very high hydrogen-to-carbon ratio,
  - High LHV per kg and **lower CO\(_2\) per kWh** than HFO/MGO for comparable engine efficiency.
- Methanol (CH\(_3\)OH) and similar oxygenated fuels:
  - Contain oxygen in the molecule,
  - Lower LHV per kg (roughly half MGO), but can enable clean combustion and lower local pollutants.
- Ammonia (NH\(_3\)):
  - Contains nitrogen and no carbon,
  - No direct CO\(_2\) from combustion, but lower LHV per kg and significant combustion and NO\(_x\) challenges.

For performance modeling, the main **input** from chemistry is \(H_L\) (and an emission factor if emissions are modeled). Fine-grained reaction mechanisms are not needed at this level.

#### 1.4 Densities and volumetric energy density

Fuel **density** \(\rho_f\) [kg/m\(^3\)] links mass and volume flows, and together with LHV gives **volumetric energy density** \(H_L \rho_f\) [MJ/m\(^3\)]:

- **Conventional fuels** (approximate densities at 15 °C):  
  - MGO: \(\rho_f \approx 820\text{–}890\ \text{kg/m}^3\).  
  - HFO: \(\rho_f \approx 975\text{–}1010\ \text{kg/m}^3\).  

- **LNG** (around \(-160^\circ\)C):  
  - \(\rho_f \approx 410\text{–}500\ \text{kg/m}^3\).  

- **Methanol** (near ambient):  
  - \(\rho_f \approx 790\ \text{kg/m}^3\) (order-of-magnitude).  

- **Liquid ammonia** (cryogenic or pressurized):  
  - \(\rho_f \approx 600\text{–}700\ \text{kg/m}^3\), with representative values around 680 kg/m\(^3\).  

For tank sizing and fuel logistics, the volumetric energy density \(H_L \rho_f\) is critical. For the power-flow model, it is mainly used to convert **mass fuel consumption** into **bunkered volume**.

---

### 2. Specific Fuel Oil Consumption (SFOC) in kg/kWh and Engine Efficiency

#### 2.1 Definition in kg/kWh

In this section we use **SFOC in kg/kWh**, defined as the fuel mass required to produce one kilowatt-hour of brake energy:

\[
\text{SFOC} = \frac{\text{fuel mass consumed in 1 hour}}{\text{brake energy produced in 1 hour}}
\quad [\text{kg/kWh}].
\]

For steady operation at brake power \(P_B\) [kW] and mass flow \(\dot{m}_f\) [kg/s]:

- Fuel mass consumed over 1 hour: \(m_{f,1h} = \dot{m}_f \cdot 3600\ [\text{kg}]\),
- Brake energy over 1 hour: \(E_{B,1h} = P_B \cdot 1\ \text{h}\ [\text{kWh}]\).

So:

\[
\text{SFOC} = \frac{m_{f,1h}}{E_{B,1h}}
= \frac{\dot{m}_f \cdot 3600}{P_B}
\quad [\text{kg/kWh}].
\]

Rearranged:

\[
\dot{m}_f = \frac{\text{SFOC} \cdot P_B}{3600}.
\]

This is the key link between **brake power** and **fuel mass flow** for a given operating point and fuel.

#### 2.2 Connection to LHV and engine efficiency

Recall:

\[
P_B = \eta_{\text{eng}} \,\dot{m}_f H_L.
\]

Substituting \(\dot{m}_f = \text{SFOC} \cdot P_B / 3600\) gives:

\[
P_B 
= \eta_{\text{eng}} \left(\frac{\text{SFOC} \cdot P_B}{3600}\right) H_L
\quad \Rightarrow \quad
\eta_{\text{eng}}
= \frac{3600}{\text{SFOC} \cdot H_L}.
\]

Conceptually:

- For a **fixed fuel** (\(H_L\) fixed), lower SFOC means higher engine efficiency.
- For **different fuels** (different \(H_L\)), comparing SFOC alone is not enough; it must be interpreted together with heating value.

#### 2.3 SFOC maps over operating space

For each **engine–fuel combination**, SFOC is typically given as a map over:

- Engine speed \(N\) and load \(P_B\), or  
- Engine speed and brake mean effective pressure (BMEP).

Key conceptual features:

- A **“sweet spot”** region with minimum SFOC (highest \(\eta_{\text{eng}}\)).
- Increased SFOC at:
  - Very low load (fixed losses dominate, incomplete combustion issues),
  - Very high load (approaching limits of air utilization, temperatures, etc.),
  - Operation far from design speed/load.

For the **power-flow model**:

1. The hydrodynamic/propulsive part (hull + propeller + interaction) yields a required brake power \(P_B\) for a given operating point.
2. The engine model selects an admissible \((N, P_B)\) operating point.
3. The SFOC map (for the chosen fuel and mode) gives the corresponding **fuel mass flow**.

---

### 3. Conceptual Integration of Different Fuels (LNG, Methanol, Ammonia)

From the viewpoint of the **system-level power-flow model**, introducing a new fuel modifies essentially **three inputs** at the engine boundary:

1. **Heating value** \(H_L\),
2. **Density** \(\rho_f\),
3. **SFOC map in kg/kWh** for that fuel and engine mode.

The rest of the chain (mechanical transmission, propeller, hull) is unchanged; it still works with brake power \(P_B\) as its input.

#### 3.1 Single-fuel operation

For a given operating point:

1. Hydrodynamic model → effective power \(P_E = R V\).
2. Propulsive chain and efficiencies → required brake power \(P_B\).
3. **Fuel model**:
   - Use the SFOC map \(\text{SFOC}_f(N, P_B)\) (kg/kWh) for the chosen fuel \(f\),
   - Compute mass flow:
     \[
     \dot{m}_f = \frac{\text{SFOC}_f(N, P_B)\, P_B}{3600},
     \]
   - Convert to volumetric flow:
     \[
     \dot{V}_f = \frac{\dot{m}_f}{\rho_f}.
     \]

Changing from HFO/MGO to LNG, methanol, or ammonia then consists, conceptually, of:

- Changing \(H_L\) (different energy per kg),
- Changing \(\rho_f\) (different energy per m\(^3\)),
- Using a different SFOC map (engine tuned and optimized differently).

#### 3.2 Fuel switching and blends (conceptual)

Many vessels can operate with **multiple fuels**:

- E.g. MGO and LNG, or MGO and methanol, with one being the main energy carrier and the other used as backup or in specific areas.

At a given operating point, the model can:

- Select a **single active fuel** \(f\) (pure switching), or
- Represent a **blend in energy terms**, with energy fractions:
  - \(\alpha_f\) = fraction of brake energy supplied by fuel \(f\),
  - \(\sum_f \alpha_f = 1\).

In the blended case, the total chemical energy rate is:

\[
\dot{E}_{\text{fuel,total}} = \sum_f \dot{m}_f H_{L,f},
\]

and must satisfy:

\[
P_B = \eta_{\text{eng}} \,\dot{E}_{\text{fuel,total}}.
\]

For most practical conceptual models, we treat each **mode** (e.g. “LNG mode”, “methanol mode”, “conventional mode”) as having its own SFOC map and use a **single active mode** at a time. Mixed-fuel operation then appears as a **time-sequence** of modes along the voyage.

---

### 4. Pilot Fuels in Dual-Fuel Operation

Dual-fuel marine engines often use:

- A **main alternative fuel** (LNG, methanol, ammonia), and
- A small amount of **pilot fuel** (typically MGO or MDO) to ensure reliable ignition.

The mechanical and hydrodynamic power-flow chain is unchanged; what changes is the **composition of the fuel input** at the engine.

#### 4.1 Role of pilot fuel

In many dual-fuel concepts:

- The main fuel is introduced as gas or liquid (e.g. LNG, methanol, ammonia),
- A small injection of **pilot diesel fuel** (MGO/MDO) initiates and stabilizes combustion.

The pilot fuel provides:

- High reactivity and autoignition reliability,
- Stable ignition across varying loads and ambient conditions.

From a modeling perspective, the engine now has **two fuel streams**:

- \(\dot{m}_{f,\text{main}}\) for the alternative fuel,  
- \(\dot{m}_{f,\text{pilot}}\) for the pilot oil.

#### 4.2 Energy balance with main and pilot fuels

Define:

- \(H_{L,\text{main}}\) [MJ/kg]: LHV of the main fuel (e.g. LNG, methanol, ammonia),
- \(H_{L,\text{pilot}}\) [MJ/kg]: LHV of the pilot fuel (e.g. MGO),
- \(\dot{m}_{f,\text{main}}, \dot{m}_{f,\text{pilot}}\) [kg/s]: mass flows.

Total chemical energy rate is:

\[
\dot{E}_{\text{fuel,total}}
= \dot{m}_{f,\text{main}} H_{L,\text{main}}
+ \dot{m}_{f,\text{pilot}} H_{L,\text{pilot}}.
\]

The engine converts this to brake power:

\[
P_B = \eta_{\text{eng}} \, \dot{E}_{\text{fuel,total}},
\]

where \(\eta_{\text{eng}}\) here represents the **overall efficiency in dual-fuel mode** (generally slightly different from pure diesel mode).

#### 4.3 Effective SFOC in dual-fuel mode

We can define a **total SFOC** in kg/kWh including both fuels:

\[
\text{SFOC}_\text{total}
= \frac{\dot{m}_{f,\text{main}} + \dot{m}_{f,\text{pilot}}}{P_B} \cdot 3600
\quad [\text{kg/kWh}].
\]

For accounting purposes, it is sometimes useful to separate:

\[
\text{SFOC}_\text{main} =
\frac{\dot{m}_{f,\text{main}}}{P_B} \cdot 3600, \qquad
\text{SFOC}_\text{pilot} =
\frac{\dot{m}_{f,\text{pilot}}}{P_B} \cdot 3600,
\]

such that:

\[
\text{SFOC}_\text{total}
= \text{SFOC}_\text{main} + \text{SFOC}_\text{pilot}.
\]

This decomposition is particularly relevant for:

- **Cost modeling**, when main and pilot fuels have different prices,
- **Emissions modeling**, when their emission factors differ.

#### 4.4 Typical pilot fractions (order-of-magnitude)

Pilot fuel fractions in modern marine dual-fuel engines are typically small:

- For some LNG/methanol dual-fuel concepts, pilot MGO requirements are on the order of **a few percent of the total energy input**, with indicative values from roughly 1–10% depending on technology and load.
- Advanced LNG dual-fuel engines with optimized pilot injection systems report pilot-oil consumption as low as **around 1–2% of total fuel energy**.

These numbers are **technology- and load-dependent**, but they illustrate that:

- The **main fuel** carries the vast majority of the energy,
- The **pilot fuel** is a small but non-negligible additional stream that must be included in mass and energy balances.

In a conceptual model, we can represent this with an approximate **pilot fraction** \(\beta\) in energy terms, for example:

\[
\beta
= \frac{\dot{m}_{f,\text{pilot}} H_{L,\text{pilot}}}{\dot{E}_{\text{fuel,total}}},
\]

with \(\beta\) in the range:

- \(\beta \sim 0.01\text{–}0.05\) (1–5%) for many modern dual-fuel designs,
- Possibly higher for older or conservative configurations.

Given \(\beta\), the energy split is:

\[
\dot{E}_{\text{fuel,total}} =
(1 - \beta)\,\dot{E}_{\text{fuel,total}}^{\text{main}}
+ \beta\,\dot{E}_{\text{fuel,total}}^{\text{pilot}},
\]

from which \(\dot{m}_{f,\text{main}}\) and \(\dot{m}_{f,\text{pilot}}\) follow for given LHVs.

#### 4.5 Volumetric flows and tankage for main and pilot fuels

With densities \(\rho_{\text{main}}\) and \(\rho_{\text{pilot}}\), the volumetric fuel flows are:

\[
\dot{V}_{\text{main}} = \frac{\dot{m}_{f,\text{main}}}{\rho_{\text{main}}}, \qquad
\dot{V}_{\text{pilot}} = \frac{\dot{m}_{f,\text{pilot}}}{\rho_{\text{pilot}}}.
\]

Because pilot fuel is typically a **liquid distillate** (MGO/MDO) and the main fuel may be cryogenic or low-LHV (LNG, methanol, ammonia), the dual-fuel model naturally leads to:

- **One tank system** sized for a relatively small but “high-energy-per-m\(^3\)” conventional fuel, and  
- **Another tank system** sized for a larger volume of main fuel (often with lower volumetric energy density and more complex storage systems).

---

### 5. Conceptual Takeaways

- Fuel **chemistry** enters the power-flow model primarily through:
  - Heating value \(H_L\),
  - Density \(\rho_f\),
  - Fuel-specific SFOC maps (kg/kWh).
- The **hydrodynamic and mechanical chain** (engine → shaft → propeller → hull → \(P_E\)) is unchanged by fuel choice.
- Different fuels (HFO/MGO, LNG, methanol, ammonia) change:
  - How much mass (and volume) of fuel is needed per kWh of brake power,
  - The cost and emissions per unit of effective power.
- **Pilot fuels** in dual-fuel engines add a second fuel stream:
  - Small fraction of the total energy, but important for ignition, cost, and emissions,
  - Naturally modeled by adding a second mass/energy balance at the engine boundary and combining both into a total SFOC and total chemical energy rate.

This extended fuel-side modeling integrates seamlessly into the existing **power-flow framework**, allowing the same quasi-static equilibrium (thrust = resistance, torque balance) to be evaluated for different fuel strategies, including dual-fuel and pilot-fuel configurations.
