# Physics and numerical model

This is a deliberately simplified, synthetic single-particle model. Its parameter values are illustrative, not fitted material properties. It must not be used for quantitative plasma-reactor or materials predictions.

## State and gas profile

The particle state is temperature $T_p$ and remaining oxide fraction $x$. Time $t\in[0,\tau]$ runs through the residence time $\tau$. A prescribed sinusoidal gas profile starts and ends at the surroundings temperature $T_s$ and reaches the configured peak $T_{peak}$ at $\tau/2$:

$$
T_g(t) = T_s + (T_{peak}-T_s)\sin\left(\frac{\pi t}{\tau}\right).
$$

The particle is spherical and has uniform temperature. Its diameter and material properties remain constant.

## Heat transfer

Convection to the gas and radiation to the surroundings change particle temperature:

$$
\rho\frac{\pi d^3}{6}c_p\frac{dT_p}{dt}
=\pi d^2\left[h(T_g-T_p)+\varepsilon\sigma(T_s^4-T_p^4)\right].
$$

Here $\sigma=5.670374419\times10^{-8}$ W m$^{-2}$ K$^{-4}$. The model neglects latent heat, including the heat effect of oxide reduction and melting. `reached_melting` is only a threshold crossing, not a phase-change model.

## Oxide reduction

The simplified first-order Arrhenius-style reaction is

$$
\frac{dx}{dt}=-k_0 c_{H_2}\exp\left(-\frac{E_a}{RT_p}\right)x,
$$

where $R=8.314462618153239$ J mol$^{-1}$ K$^{-1}$. Hydrogen concentration is prescribed rather than coupled to gas chemistry or particle consumption.

## Fixed-step integration

`SolverConfig.num_steps` divides the residence time into equal steps $\Delta t=\tau/N$. Temperature uses an explicit Euler update. Oxide fraction then uses an exponential update with the rate evaluated at $T_p^n$:

$$
T_p^{n+1}=T_p^n+\Delta t\left.\frac{dT_p}{dt}\right|_n,
\qquad
x^{n+1}=x^n\exp\left[-k(T_p^n)\Delta t\right].
$$

The exponential update keeps oxide fraction in $[0,1]$ for finite, non-negative rates and makes the zero-reaction limiting case exact. The scheme is first-order and has no adaptive error control. Explicit heat integration can become unstable for large steps; users must check step-size convergence for their configurations. Outputs include $N+1$ samples, including both endpoints.

## Units and validation domains

| Configuration value | Unit | Accepted domain |
|---|---:|---:|
| `peak_temperature` | K | Finite, $>0$, and $\geq T_s$ |
| `residence_time` | s | Finite, $>0$ |
| `hydrogen_concentration` | mol m$^{-3}$ | Finite, $\geq0$ |
| `diameter` | m | Finite, $>0$ |
| `initial_temperature` | K | Finite, $>0$ |
| `initial_oxide_fraction` | - | Finite, $0\leq x_0\leq1$ |
| `density` | kg m$^{-3}$ | Finite, $>0$ |
| `heat_capacity` | J kg$^{-1}$ K$^{-1}$ | Finite, $>0$ |
| `heat_transfer_coefficient` | W m$^{-2}$ K$^{-1}$ | Finite, $>0$ |
| `emissivity` | - | Finite, $0\leq\varepsilon\leq1$ |
| `surroundings_temperature` | K | Finite, $>0$ |
| `melting_temperature` | K | Finite, $>0$ |
| `reaction_pre_exponential` | m$^3$ mol$^{-1}$ s$^{-1}$ | Finite, $\geq0$ |
| `activation_energy` | J mol$^{-1}$ | Finite, $\geq0$ |
| `num_steps` | - | Integer, $\geq1$ |

The defaults are illustrative: $\rho=5000$ kg m$^{-3}$, $c_p=500$ J kg$^{-1}$ K$^{-1}$, $h=100$ W m$^{-2}$ K$^{-1}$, $\varepsilon=0.8$, $T_s=300$ K, $T_m=1800$ K, $k_0=0.05$ m$^3$ mol$^{-1}$ s$^{-1}$, and $E_a=30000$ J mol$^{-1}$.

## API and batching

```python
from reactor_lab import simulate, simulate_batch

trajectory = simulate(config)
batch = simulate_batch([config, other_config])
```

`simulate` is jitted with the typed configuration as a static argument. `simulate_batch` stacks reactor and particle inputs, then applies `vmap` to the same solver. Batch entries must share material and solver settings. Batched arrays have a leading batch dimension; scalar summaries have one value per batch entry. All simulation data and plots are synthetic.

## Limitations

No plasma chemistry, flow coupling, drag, atomization, particle-size evolution, internal temperature gradients, reaction heat, latent heat, or coupled hydrogen balance is represented. The gas profile and heat-transfer coefficient are prescribed. These omissions are intentional MVP simplifications.
