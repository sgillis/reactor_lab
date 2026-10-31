---
title: "A Toy Model for Thermal-Plasma Powder Processing"
subtitle: "Differentiable Reactor Lab — physics, interpretation, and numerical method"
lang: en
---

> **Purpose.** This guide summarizes the reduced-order, single-particle model used in the project. It is a synthetic scientific-software test case, not a quantitatively validated model of a real plasma reactor or material.

# 1. Model overview

Thermal-plasma powder processing can be reduced to the following causal chain:

**hot gas → particle heating → temperature-dependent oxide reduction → processed particle**

The model follows one spherical particle through a prescribed reactor temperature history. Its state contains only:

- particle temperature, $T_p(t)$;
- remaining oxide fraction, $x(t)$.

The two states are coupled: gas temperature changes particle temperature, and particle temperature controls the reduction rate. This is the essential thermal–kinetic interaction represented by the toy model.

The model belongs to the family of **single-particle thermal models**. It combines lumped-capacitance heat transfer, convection and radiation, first-order Arrhenius-style kinetics, and a prescribed gas-temperature profile. The combined model is custom; it is not a named, published reactor model.

# 2. Meaning of remaining oxide fraction

The variable $x$ is a **dimensionless oxide inventory**, not a count of oxygen atoms. It can loosely be interpreted as the fraction of the particle’s reference material that remains in oxide form:

- $x=0.9$: the reference material is 90% oxide;
- $x=0.45$: half as much oxide remains;
- $x=0$: no oxide remains.

For example, a trajectory that starts at $x_0=0.9$ and ends at $x(	au)=0.874$ has lost 0.026, or **2.6 percentage points**, of oxide fraction. Relative to the initial oxide inventory, its reduction is

$$
\frac{x_0-x(\tau)}{x_0}
=\frac{0.9-0.874}{0.9}
\approx 2.9\%.
$$

If the oxide has a fixed chemical formula, such as $\mathrm{MO}$ or $\mathrm{M_2O_3}$, the oxygen bound in that oxide is proportional to the oxide inventory. The model nevertheless does not track atoms, individual chemical species, stoichiometry, or an explicit oxygen mass balance.

# 3. Prescribed gas-temperature profile

The reactor environment is imposed through a smooth sinusoidal gas-temperature profile:

$$
\boxed{
T_g(t)=T_s+(T_{\mathrm{peak}}-T_s)
\sin\!\left(\frac{\pi t}{\tau}\right)
}
\qquad 0\leq t\leq\tau .
$$

Here $t$ is elapsed time, $\tau$ is residence time, $T_s$ is surroundings temperature, and $T_{\mathrm{peak}}$ is peak gas temperature. The particle enters at $T_s$, experiences $T_{\mathrm{peak}}$ halfway through its residence time, and exits at $T_s$.

For $T_s=300\ \mathrm{K}$, $T_{\mathrm{peak}}=2500\ \mathrm{K}$, and $\tau=0.5\ \mathrm{s}$:

| Elapsed time | Gas temperature |
|---:|---:|
| 0 s | 300 K |
| 0.125 s | approximately 1856 K |
| 0.25 s | 2500 K |
| 0.375 s | approximately 1856 K |
| 0.5 s | 300 K |

This profile is a **boundary condition**, not a prediction. It represents entry into and exit from a hot zone without solving plasma flow. Its symmetric sinusoidal shape is an illustrative choice, not a measured reactor profile.

At every solver step, $T_g(t)$ enters convective heat transfer. The resulting particle temperature then changes the reaction rate:

**sinusoidal profile → gas temperature → particle heating → oxide reduction**.

# 4. Particle heat transfer

The particle is assumed to have a uniform temperature. Conservation of energy gives

$$
\boxed{
\underbrace{\rho\frac{\pi d^3}{6}c_p\frac{\mathrm{d}T_p}{\mathrm{d}t}}_{\text{rate of change of particle energy}}
=
\underbrace{\pi d^2 h(T_g-T_p)}_{\text{convection}}
+
\underbrace{\pi d^2\varepsilon\sigma(T_s^4-T_p^4)}_{\text{net radiation}}
}
$$

The factors $\pi d^3/6$ and $\pi d^2$ are the volume and surface area of a sphere. Density $\rho$ and specific heat capacity $c_p$ determine thermal inertia. The heat-transfer coefficient $h$ controls convective exchange, while emissivity $\varepsilon$ and the Stefan–Boltzmann constant $\sigma$ control radiation.

Dividing by particle heat capacity gives the implemented temperature equation:

$$
\frac{\mathrm{d}T_p}{\mathrm{d}t}
=\frac{6}{\rho c_p d}
\left[h(T_g-T_p)+\varepsilon\sigma(T_s^4-T_p^4)\right].
$$

A positive gas–particle temperature difference heats the particle. Radiation transfers energy toward the surroundings. Latent heat and reaction heat are omitted, so crossing the melting temperature does not alter this energy equation.

# 5. Oxide-reduction kinetics

The remaining oxide fraction obeys first-order kinetics:

$$
\boxed{
\frac{\mathrm{d}x}{\mathrm{d}t}=-k(T_p)x
}
$$

with the Arrhenius-style rate coefficient

$$
\boxed{
k(T_p)=k_0c_{H_2}
\exp\!\left(-\frac{E_a}{RT_p}\right)
}
$$

The pre-exponential factor $k_0$ sets the overall rate scale, hydrogen concentration $c_{H_2}$ scales the prescribed driving force, and activation energy $E_a$ controls temperature sensitivity. The universal gas constant is $R$.

This equation produces important limiting behavior:

- if $k_0=0$ or $c_{H_2}=0$, oxide fraction remains unchanged;
- a larger rate coefficient produces more reduction under otherwise identical conditions;
- for a finite non-negative rate, oxide fraction remains non-increasing and bounded below by zero.

The reaction is a simplified scalar law, not a resolved hydrogen–oxide chemical mechanism.

# 6. Where residence time enters

Residence time affects both the gas history and the total exposure time. The simulator integrates from $t=0$ to $t=\tau$, and the sinusoidal profile depends explicitly on $t/\tau$.

The effect on reaction can be seen by formally integrating the kinetic equation:

$$
\boxed{
x(\tau)=x_0\exp\!\left[-\int_0^\tau k\!\left(T_p(t)\right)\,\mathrm{d}t\right]
}
$$

Thus, residence time sets the interval over which reduction accumulates. It also changes heating because $T_g(t)$ and particle temperature depend on $\tau$. Consequently, residence time is part of the model even though it does not appear as a standalone factor in the instantaneous expression $\mathrm{d}x/\mathrm{d}t=-kx$.

# 7. Fixed-step numerical method

`SolverConfig.num_steps` divides residence time into $N$ equal steps:

$$
\Delta t=\frac{\tau}{N}, \qquad t_n=n\Delta t .
$$

Temperature uses an explicit Euler update:

$$
T_p^{n+1}=T_p^n+\Delta t
\left.\frac{\mathrm{d}T_p}{\mathrm{d}t}\right|_{t_n}.
$$

Oxide fraction uses an exponential update with the rate evaluated at the current temperature:

$$
\boxed{
x^{n+1}=x^n\exp\!\left[-k(T_p^n)\Delta t\right]
}
$$

The exponential update preserves the oxide lower bound and makes the zero-rate case exact. The combined scheme is first-order and has no adaptive error control. Sufficiently small steps are required for accurate and stable heat integration. The output contains $N+1$ samples, including both endpoints.

# 8. Symbols and units

| Symbol | Meaning | Unit |
|---|---|---|
| $t$ | Elapsed time | s |
| $\tau$ | Total residence time | s |
| $T_p$ | Particle temperature | K |
| $T_g$ | Prescribed gas temperature | K |
| $T_s$ | Surroundings temperature | K |
| $T_{\mathrm{peak}}$ | Peak gas temperature | K |
| $x$ | Remaining oxide fraction | dimensionless |
| $x_0$ | Initial oxide fraction | dimensionless |
| $\rho$ | Particle density | kg m$^{-3}$ |
| $c_p$ | Specific heat capacity | J kg$^{-1}$ K$^{-1}$ |
| $d$ | Particle diameter | m |
| $h$ | Convective heat-transfer coefficient | W m$^{-2}$ K$^{-1}$ |
| $\varepsilon$ | Particle emissivity | dimensionless, 0–1 |
| $\sigma$ | Stefan–Boltzmann constant | W m$^{-2}$ K$^{-4}$ |
| $k_0$ | Reaction pre-exponential factor | m$^3$ mol$^{-1}$ s$^{-1}$ |
| $c_{H_2}$ | Prescribed hydrogen concentration | mol m$^{-3}$ |
| $E_a$ | Activation energy | J mol$^{-1}$ |
| $R$ | Universal gas constant | J mol$^{-1}$ K$^{-1}$ |
| $N$ | Number of fixed solver steps | integer |
| $\Delta t$ | Solver step size | s |

The material and kinetic parameters in the current configuration are illustrative. They are not measurements for a particular powder.

# 9. Outputs and physical interpretation

| Output | Interpretation |
|---|---|
| Temperature trajectory | How the particle heats and cools along the prescribed history |
| Oxide-fraction trajectory | When and how quickly reduction occurs |
| Final oxide fraction | Whether a simplified reduction target is met |
| Maximum particle temperature | Peak thermal exposure |
| Melting-threshold flag | Whether particle temperature ever reaches the configured threshold |

The melting flag does **not** simulate phase change. It is only a threshold indicator that can later serve as a constraint in operating-condition optimization.

# 10. Why this is a useful toy model

The model retains several features that make thermal-plasma powder processing interesting:

1. **Thermal–kinetic coupling:** gas temperature affects reaction indirectly through particle temperature.
2. **Competing timescales:** heating, reaction, and residence time jointly determine the result.
3. **Particle-size effects:** volume and surface area scale differently with diameter.
4. **Interpretable objectives:** final oxide, peak temperature, and melting thresholds resemble processing goals.
5. **Checkable limits:** zero reaction, thermal equilibrium, monotonic reduction, and bounded oxide provide meaningful tests.
6. **Cheap computation:** the model is deterministic, differentiable, and suitable for JAX `jit` and `vmap`, synthetic datasets, surrogate training, and optimization.

It is therefore complex enough to demonstrate a scientific simulation-and-learning workflow, but small enough to inspect and verify.

# 11. Limitations and appropriate use

The model excludes plasma chemistry, electromagnetic fields, coupled reactor flow, particle drag, turbulence, internal temperature gradients, latent heat, reaction heat, particle-size evolution, and detailed reaction mechanisms. Material properties and hydrogen concentration are prescribed and constant. There is no coupling between particle processing and the gas.

The model is appropriate for:

- testing scientific software and numerical workflows;
- generating clearly labelled synthetic data;
- demonstrating surrogate modelling and constrained optimization;
- exploring qualitative behavior of a reduced thermal–kinetic system.

It is not appropriate for reactor design, materials prediction, process safety, or quantitative operating recommendations.

**Implementation reference:** `src/reactor_lab/simulator.py`  
**Detailed validation domains:** `docs/physics.md`
