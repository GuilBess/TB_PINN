# Neural Network Architecture

- Inputs: will depend on the number of dimensions of the simulation and if we want time or no (2 to 4)
- Output: will depend on the number of dimensions of the simulation (3 or 4)
- Hidden: MLP 

Hard to know how many hidden layers and how many neurons/layer to use. ChatGPT was used w/ prompt : "how many hidden layers and how many neurons do I need for a PINN designed to simulate fluids?". The answer was based on a few papers, results were : 

- Around 6-8   hidden layers and 40-60  neuron/layer for simple 2D flow, 
- Around 8-10  hidden layers and 60-100 neuron/layer for unsteady flow, 
- Around 10-12 hidden layers and 100 +  neuron/layer for high res, turbulent flow

This gives me an idea where to start, but chatGPT also stated in it's answer that there is no right answer, and I will probably have to try a few different architectrues by hand

- Activation function: When choosing activation functions, we have to be careful that it can be differentiable at least a couple of time. ReLU for example becomes 0 after 2 derivative.
This paper (https://arc.aiaa.org/doi/abs/10.2514/6.2023-1803) shows that Swish would yield the best accuracy, but also states that tanh is also sufficiently good and needs less computational cost. We will probably be using the tanh activation function.