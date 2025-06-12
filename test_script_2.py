from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import random

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
        act_f = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(inputs, hidden),
                        act_f()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(hidden, hidden),
                            act_f()]) for _ in range(n_layers-1)])
        self.fce = nn.Linear(hidden, outputs)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# Point generation
def generate_points(N):
    zones = [
        (0, 0.4, -0.02, 0.02, 0.01),
        (0, 0.4, 0.01, 0.02, 0.01),
        (0, 0.4, -0.02, -0.01, 0.01),
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

# PDE computation
def compute_PDE(xy, y_in):
    u = y_in[:, 0:1]
    v = y_in[:, 1:2]
    p = y_in[:, 2:3]
    grads = lambda out, inp: torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out), create_graph=True)[0]
    grads_u = grads(u, xy)
    grads_v = grads(v, xy)
    grads_p = grads(p, xy)
    dudx, dudy = grads_u[:, 0], grads_u[:, 1]
    dvdx, dvdy = grads_v[:, 0], grads_v[:, 1]
    dpdx, dpdy = grads_p[:, 0], grads_p[:, 1]
    d2udx2 = grads(dudx, xy)[:, 0]
    d2udy2 = grads(dudy, xy)[:, 1]
    d2vdx2 = grads(dvdx, xy)[:, 0]
    d2vdy2 = grads(dvdy, xy)[:, 1]
    return dudx, dudy, dpdx, dpdy, d2udx2, d2udy2, dvdx, dvdy, d2vdx2, d2vdy2

# Loss function
def compute_loss(model, inputs, masks, i):
    yhp = model(inputs)
    no_slip_mask, inlet_mask, outlet_mask, interior_mask = masks

    p_outlet = yhp[outlet_mask, 2]
    outlet_loss = torch.mean(p_outlet**2) * 250
    no_slip_loss = (torch.mean(yhp[no_slip_mask, 0]**2) + torch.mean(yhp[no_slip_mask, 1]**2)) * 200
    inlet_loss = (torch.mean((yhp[inlet_mask, 0] - u_avg)**2) +
                  torch.mean(yhp[inlet_mask, 1]**2) +
                  torch.mean((yhp[inlet_mask, 2] - 1)**2)) * 200

    x_interior = inputs[interior_mask]
    y_interior = model(x_interior)
    dudx, dudy, dpdx, dpdy, d2udx2, d2udy2, dvdx, dvdy, d2vdx2, d2vdy2 = compute_PDE(x_interior, y_interior)
    navier_x = rho * (y_interior[:, 0] * dudx + y_interior[:, 1] * dudy) + dpdx - mu * (d2udx2 + d2udy2)
    navier_y = rho * (y_interior[:, 0] * dvdx + y_interior[:, 1] * dvdy) + dpdy - mu * (d2vdx2 + d2vdy2)
    continuity = dudx + dvdy
    loss_phys = (torch.mean(navier_x**2) + torch.mean(navier_y**2) + torch.mean(continuity**2) * 1000)
    loss = no_slip_loss + inlet_loss + outlet_loss + loss_phys
    return loss, loss_phys, no_slip_loss, inlet_loss, outlet_loss

# Simulation parameters
h = 0.01
L = 0.2
u_avg = 0.01
rho = 1000
mu = 0.001
Re = (rho * u_avg * 2*h) / mu
print(f"Reynolds number: {Re}")

rng = random.randint(0, 200000)
torch.manual_seed(rng)

# Model initialization
poiseuille_model = Model(2, 3, 100, 10).to(device)
optimizer_a = torch.optim.Adam(poiseuille_model.parameters(), lr=1e-4)

# Logging
losses = {key: [] for key in ["tot", "phys", "no_slip", "inlet", "outlet"]}

batch_size = 5000
total_points = 50000
num_batches = total_points // batch_size

# Training loop
for i in trange(75000):
    full_tensor_np = generate_points(total_points)
    full_tensor = torch.from_numpy(full_tensor_np).float().to(device)
    
    # Optional: Shuffle
    perm = torch.randperm(total_points)
    full_tensor = full_tensor[perm-1]

    optimizer_a.zero_grad()

    for b in range(num_batches):
        batch = full_tensor[b*batch_size:(b+1)*batch_size].clone().detach().requires_grad_(True)
        x_vals_tensor = batch[:, 0]
        y_vals_tensor = batch[:, 1]

        no_slip_mask = (y_vals_tensor < -h) | (y_vals_tensor > h)
        inlet_mask = (torch.abs(x_vals_tensor - 0) < 1e-2) & (~no_slip_mask)
        outlet_mask = (torch.abs(x_vals_tensor - L * 2) < 1e-2) & (~no_slip_mask)
        interior_mask = ~(no_slip_mask | inlet_mask | outlet_mask)
        masks = (no_slip_mask, inlet_mask, outlet_mask, interior_mask)
        
        loss, loss_phys, no_slip_loss, inlet_loss, outlet_loss = compute_loss(poiseuille_model, batch, masks, i)
        loss.backward()
    
    optimizer_a.step()


    if i > 30000:
        losses["tot"].append(loss.item())
        losses["phys"].append(loss_phys.item())
        losses["no_slip"].append(no_slip_loss.item())
        losses["inlet"].append(inlet_loss.item())
        losses["outlet"].append(outlet_loss.item())

    if i % 5000 == 0 or i == 99999:
        tqdm.write(f"Step {i+1}, Loss: {loss.item():.6f}\n Physics: {loss_phys.item():.6f}   No slip: {no_slip_loss.item():.6f}   Inlet: {inlet_loss.item():.6f}   Outlet: {outlet_loss.item():.6f}")

# Plotting losses
epochs = range(30000, len(losses["tot"]) + 30000)
for key in losses:
    values = np.array(losses[key])
    upper = np.percentile(values, 95)
    plt.plot(epochs, values, label=key)
    plt.ylim(0, upper)
    plt.title(f"Loss {key}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{key}_clipped.png")
    plt.clf()

torch.save(poiseuille_model.state_dict(), "test.pt")
