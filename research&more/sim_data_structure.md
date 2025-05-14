# Simulation zone data structure
We have a couple options to choose from:
- Graph networks
- Grid
- Coordinates

## Graph network
This method consist of discretising simulation zone into a lot of points with edges connecting them to adjacent points.
After some research, it is a method used also for neural networks physics simulation.

## Grid representation
This method discretises the space in a grid, where each square has a certain velocity, pressure, etc... This is an easy way to represent the state of the simulation, but it has a few problem for us. First problem is that we would like to be able to have greater accuracy around specific parts of the simulation. While we can augment resolution by dividing the squares. Having a grid like this poses a problem for the size of the input. We have to give our model the "full picture" of the grid in order to train it, which means a lot of input and output nodes.

## Coordinates
For this method, we take as input the coordinates of the system and the time (if we want to simulate through time) and return the velocity and the pressure. this method is the most straightforward. We know that for this point there is a way to know it's state, so we train the model to do just that, with multiple points. Howerver, we have to choose those points in a meaningful way.