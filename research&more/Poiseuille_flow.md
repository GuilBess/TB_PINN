# Poiseuille flow sim definition
We have to set Boundary Conditions (BCs) for the Poiseuille flow. In our case, we have a 2D Poiseuille flow between 2 infinitely wide planes. So here is what we have to look at:
- Wall boundaries
- Inlet boundary
- Outlet boundary
source: https://www.fifty2.eu/innovation/planar-poiseuille-flow-2-d-in-preonlab/

## Wall boundaries
In the case of a viscous flow (like ours), we want to have no flow velocity at the walls. This in turn gives us: u(y = ±h) = 0, v(y = ±h) = 0

## Inlet boundary
We want to tell how the flow acts when "entering the pipe". We will just fix the velocity of the flow at the inlet to be constant, with no y componant: u(x = 0) = const., v(x = 0) = 0

## Outlet boundary
At the outlet, we want the flow to be fully developed, meaning the velocity gradient should be 0 => ∂u/∂x​=0, ∂v/∂x​=0

# Simulation hyperparameters
We need laminar flow to have a stable simulation. According to https://www.simscale.com/docs/simwiki/numerics-background/what-is-the-reynolds-number/, the flow should be laminar with a Reynolds number under 2300.

We can find the Reynolds number with the formula: (ρ * u * 2​h​)/μ. For a first simulation, I decided to go with water, an average speed of 0.01m/s and a height of the half channel of 1cm (0.02 m) which gives us:

- ρ = 1000 [kg/m^3]
- u = 0.01 [m/s]
- h = 0.01 [m]
- μ = 1 * 10^-3 [Pa * s]
- Re = (1000 * 0.01 * 0.02) / (0.001) = 200

The Reynolds number is way under 2300, we will have to see if it's a problem during tests

In order to guarantee we are in a steady state, we have to guarantee a certain channel lenght, typically 10 * h, so 10cm in our case

so we have:
- ρ = 1000 [kg/m^3]
- u = 0.01 [m/s]
- h = 0.01 [m]
- μ = 1 * 10^-3 [Pa * s]
- L = 0.1 [m]

![image](img\schema.png)