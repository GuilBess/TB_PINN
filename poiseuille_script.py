import rff.layers
from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import random
import rff

# Paths and device setup
path = os.curdir
print(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Neural network model definition
class Model(nn.Module):
    def __init__(self, inputs, outputs, hidden, n_layers):
        super().__init__()
        layers = []
        
        # Random Fourier Features Pytorch is an implementation of "Fourier Features Let
        # Networks Learn High Frequency Functions in Low Dimensional Domains" by Tancik et al.
        # (from https://github.com/jmclong/random-fourier-features-pytorch)
        layers.append(rff.layers.GaussianEncoding(10.0, inputs, 64)) 
        # Input layer       
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden, outputs))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

# Point generation
def generate_points(N):
    zones = [
        (0, 0.4, -0.02, 0.02, 0.01),
        (0, 0.4, 0.01, 0.02, 0.00),
        (0, 0.4, -0.02, -0.01, 0.00),
        (0.19, 0.21, -0.01, 0.01, 0.0)
    ]
    weighted_areas = []
    total_weighted_area = 0
    for x_min, x_max, y_min, y_max, weight in zones:
        area = (x_max - x_min) * (y_max - y_min)
        w_area = area * weight
        weighted_areas.append(w_area)
        total_weighted_area += w_area

    points_per_zone = [int((w_area / total_weighted_area) * N) for w_area in weighted_areas]

    all_points = []
    for i, (x_min, x_max, y_min, y_max, _) in enumerate(zones):
        count = points_per_zone[i]
        xs = np.random.uniform(x_min, x_max, count)
        ys = np.random.uniform(y_min, y_max, count)
        all_points.append(np.column_stack((xs, ys)))
    return np.vstack(all_points)

def compute_PDE_full(xy, y_pred):
    u = y_pred[:, 0:1]
    v = y_pred[:, 1:2]
    p = y_pred[:, 2:3]
    
    # Gradient function
    grads = lambda out, inp: torch.autograd.grad(
        out, inp, 
        grad_outputs=torch.ones_like(out),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]
    
    # First derivatives for ALL points
    grads_u = grads(u, xy)
    grads_v = grads(v, xy)
    grads_p = grads(p, xy)
    
    dudx, dudy = grads_u[:, 0:1], grads_u[:, 1:2]
    dvdx, dvdy = grads_v[:, 0:1], grads_v[:, 1:2]
    dpdx, dpdy = grads_p[:, 0:1], grads_p[:, 1:2]
    
    # Second derivatives for ALL points
    d2udx2 = grads(dudx, xy)[:, 0:1]
    d2udy2 = grads(dudy, xy)[:, 1:2]
    d2vdx2 = grads(dvdx, xy)[:, 0:1]
    d2vdy2 = grads(dvdy, xy)[:, 1:2]
    
    return dudx, dudy, dpdx, dpdy, d2udx2, d2udy2, dvdx, dvdy, d2vdx2, d2vdy2

def compute_loss(model, inputs, masks, i):
    no_slip_mask, inlet_mask, outlet_mask, interior_mask = masks

    yhp_all = model(inputs)
    
    # Step 2: Separate forward pass for interior points (for physics)
    interior_inputs = inputs[interior_mask].clone().detach().requires_grad_(True)
    
    # Compute ALL gradients once
    dudx, dudy, dpdx, dpdy, d2udx2, d2udy2, dvdx, dvdy, d2vdx2, d2vdy2 = compute_PDE_full(inputs, yhp_all)
    
    w_no_slip = 2.0
    w_inlet = 1.0
    w_phys = 4.0
    w_cont = 4.0
    
    # BC losses
    no_slip_loss = (torch.mean(yhp_all[no_slip_mask, 0]**2) + 
                   torch.mean(yhp_all[no_slip_mask, 1]**2)) * w_no_slip
    
    inlet_loss = (torch.mean((yhp_all[inlet_mask, 0] - u_avg)**2) +
                  torch.mean(yhp_all[inlet_mask, 1]**2) +
                  torch.mean(yhp_all[inlet_mask, 2]**2)) * w_inlet
    
    # Physics computation - use precomputed gradients, subset to interior
    u_int = yhp_all[0:1]
    v_int = yhp_all[1:2]
    
    # Navier-Stokes equations (dimensionless form)
    Re = (rho * u_avg * 2*h) / mu
    navier_x = (u_int * dudx + v_int * dudy + 
                dpdx/(rho*(u_avg**2)) - (d2udx2 + d2udy2)/Re)
    navier_y = (u_int * dvdx + v_int * dvdy + 
                dpdy/(rho*(u_avg**2)) - (d2vdx2 + d2vdy2)/Re)
    continuity = dudx + dvdy

    # Gradually reduce initialization loss
    if i < 1:
        w_init = 1
    elif i < 250000:
        w_init = 1 - ((i-1) / 250000)
    else:
        w_init = 0

    init_loss = (torch.mean((yhp_all[interior_mask, 0] - u_avg)**2) +
                torch.mean((yhp_all[interior_mask, 1])**2) + 
                torch.mean((yhp_all[interior_mask, 2])**2)) * w_init
    
    # Physics loss with weighting
    loss_phys = (torch.mean(navier_x**2) + torch.mean(navier_y**2)) * w_phys + torch.mean(continuity**2) * w_cont

    outlet_loss = torch.Tensor([0])

    total_loss = inlet_loss + loss_phys + init_loss + no_slip_loss
    
    return total_loss, loss_phys, no_slip_loss, inlet_loss, outlet_loss, init_loss, w_init

# Simulation parameters
h = 0.01   # demi-height of sim
L = 0.2    # demi-length of sim
u_avg = 0.01 # average x velocity
rho = 1000 # density of the fluid
mu = 0.001 # viscosity of the fluid
Re = (rho * u_avg * 2*h) / mu
print(f"Reynolds number: {Re}")

rng = random.randint(0, 200000)

torch.manual_seed(rng)
np.random.seed(rng)

poiseuille_model = Model(2, 3, 150, 12).to(device) 

optimizer_a = torch.optim.Adam(poiseuille_model.parameters(), lr=5e-4)

def lr_schedule(epoch):
    if epoch < 10000:
        return 1.0
    elif epoch < 50000:
        return 0.5
    elif epoch < 100000:
        return 0.2
    elif epoch < 250000:
        return 0.1
    else:
        return 0.05

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_a, lr_schedule)

# Logging
losses = {key: [] for key in ["weight","tot", "phys", "no_slip", "inlet", "outlet", "init"]}

total_points = 25000 

# points generation
full_tensor_np = generate_points(total_points)
full_tensor = torch.from_numpy(full_tensor_np).float().to(device).requires_grad_()

# Training loop
for i in trange(300000):  # Reduced training steps

    optimizer_a.zero_grad()

    # Generate new points 
    full_tensor_np = generate_points(total_points)
    full_tensor = torch.from_numpy(full_tensor_np).float().to(device).requires_grad_()

    x_vals_tensor = full_tensor[:, 0]
    y_vals_tensor = full_tensor[:, 1]

    # binary masks describing the the zones of the BC in the simulation. Outlet is not used right now
    no_slip_mask = (y_vals_tensor < -h) | (y_vals_tensor > h)
    inlet_mask = (torch.abs(x_vals_tensor) < 1e-2) & (~no_slip_mask)
    outlet_mask = ((L * 2 - x_vals_tensor) < 1e-2) & (~no_slip_mask)
    interior_mask = ~(no_slip_mask | inlet_mask)
    masks = (no_slip_mask, inlet_mask, outlet_mask, interior_mask)
    
    loss, loss_phys, no_slip_loss, inlet_loss, outlet_loss, init_loss, w_init = compute_loss(poiseuille_model, full_tensor, masks, i)

    loss.backward()
    # Gradient clipping for stability
    # torch.nn.utils.clip_grad_norm_(poiseuille_model.parameters(), max_norm=1.0)
    optimizer_a.step()
    scheduler.step()
    
    # saving the model over a few epochs
    if i == 50000:
        torch.save(poiseuille_model.state_dict(), "50k.pt")

    if i == 75000:
        torch.save(poiseuille_model.state_dict(), "75k.pt")

    if i == 100000:
        torch.save(poiseuille_model.state_dict(), "100k.pt")
    
    if i == 125000:
        torch.save(poiseuille_model.state_dict(), "125k.pt")

    # Logging
    if i % 100 == 0:  # More frequent logging for monitoring
        losses["weight"].append(w_init)
        losses["tot"].append(loss.item())
        losses["phys"].append(loss_phys.item())
        losses["no_slip"].append(no_slip_loss.item())
        losses["inlet"].append(inlet_loss.item())
        losses["outlet"].append(outlet_loss.item())
        losses["init"].append(init_loss.item())

    if i % 5000 == 0 or i == 74999:        
        tqdm.write(f"Step {i+1}, Loss: {loss.item():.6f}\n Physics: {loss_phys.item():.6f}   No slip: {no_slip_loss.item():.6f}   Inlet: {inlet_loss.item():.6f}   Outlet: {outlet_loss.item():.6f}   Init: {init_loss.item():.6f}")

# Plotting losses with moving average for smoother visualization
window_size = 50
# Create individual plots for each loss
for key in losses:
    try:
        values = np.array(losses[key])
        if len(values) > window_size:
            # Calculate moving average
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            epochs_smooth = range(len(smoothed))
            plt.plot(epochs_smooth, smoothed, label=f'{key} (smoothed)')
            plt.plot(range(len(values)), values, alpha=0.3, label=f'{key} (raw)')
            plt.title(f"Loss {key}")
            plt.xlabel("Epoch (x100)")
            plt.ylabel("Loss")
            plt.legend()
            if key != "weight":
                plt.yscale("log")
            plt.grid(True)
            plt.savefig(f"{key}_improved.png")
            plt.clf()
    except:
        print(f"could not plot {key}")

# Create combined plot with all smoothed values
plt.figure(figsize=(12, 8))
losses.pop("weight", None)
for key in losses:
    try:
        values = np.array(losses[key])
        if len(values) > window_size:
            # Calculate moving average
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            epochs_smooth = range(len(smoothed))
            plt.plot(epochs_smooth, smoothed, label=f'{key} (smoothed)', linewidth=2)
    except:
        print(f"could not plot {key}")

plt.title("All Loss Components (Smoothed)")
plt.xlabel("Epoch (x100)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.yscale("log")
plt.savefig("all_losses_combined.png")
plt.clf()

torch.save(poiseuille_model.state_dict(), "test.pt")