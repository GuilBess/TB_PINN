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