# Physics Informed Neural Network

This is a project made at the HES-SO in Sion for my Bachelor's degree. The goal is to explore Physics Informed Neural Networks (PINNs) and apply them to fluid simulations.

At first, we will have to simulate a Poiseuille and a Von Karman instability in 2D and in a steady state.

We have to compare results to traditional methods. We want to know how accurate and efficient PINNs are.

We then want to simulate a Von Karman instability in an unsteady state, meaning over time, in 2 dimensions.
With that, we want to evaluate PINNs for fluid simulation in this domain.

Finally, we want to try to simulate more complex situations, either with 3D simulations or by adding obstacles to previous simulations.

## test_script.py
This is a test script that helped me to familiarize with both PINNs and running scripts on chacha. 
The aim was to be able to simulate the physics of a spring using a PINN. We can see the same thing in the example.ipynb note book, but this script was made in order to familiarize with the process of runing scripts using slurm.

## example.ipynb
This is a notebook used to try different things during the project. If I wanted to see what kind of points I was generating, visualize zones in the domain, test functions,... I would do that in this notebook. 

## poiseuille_script.py
This is the training script for the poiseuille flow PINN, to be run on the computing server

## display_results.ipynb
This notebook is used to visualize results from the PINN, for the poiseuille flow.

## run_slurm.sh
bash script for slurm, used to launch the job on the computing server, using the GPU.

## container.def
The sysadmins asked of us to run ous tasks from a special type of conainer, known as apptainer. This file describes what script we want in the appainer and installs the dependencies.

# Dependencies
for this version of the code, please install the following libraries:

```bash
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install matplotlib
pip install random-fourier-features-pytorch
pip install pandas
```

# Running the code
To run the script, just move the script file, run_slurm file and container.def file on the server and use the folowwing command to build the apptainer:

```bash
apptainer build mycontainer.sif container.def
```

After the apptainer has been built, you can launch it with slurm:
```bash
sbatch run_slurm.sh
```

When the script is done running, you can either see the losses graphs or use the .pt files generated to see the results in the display_result.ipynb notebook

# DISCLAIMER
At this time and with this code, the results are not correct. The simulation is not conform the the theorical physical result.