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
        layers = []
        
        # Input layer
        layers.append(nn.Linear(inputs, hidden))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.1))
        
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



def compute_DPDY(xy, y_in, mask):
    p = y_in[mask, 2:3]
    grads = lambda out, inp: torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out), 
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 only_inputs=True)[0]
    grads_p = grads(p, xy)
    dpdx, dpdy = grads_p[:, 0], grads_p[:, 1]
    return dpdy

# Loss function
def compute_PDE_full(xy, y_pred):
    """
    Compute PDE terms for ALL points, then subset later
    """
    u = y_pred[:, 0:1]
    v = y_pred[:, 1:2]
    p = y_pred[:, 2:3]
    
    # Gradient function
    grads = lambda out, inp: torch.autograd.grad(
        out, inp, 
        grad_outputs=torch.ones_like(out), 
        create_graph=True,
        retain_graph=True,
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
    # Single model inference for ALL points
    yhp = model(inputs)
    no_slip_mask, inlet_mask, outlet_mask, interior_mask = masks
    
    # Compute ALL gradients once - this is the key insight
    dudx, dudy, dpdx, dpdy, d2udx2, d2udy2, dvdx, dvdy, d2vdx2, d2vdy2 = compute_PDE_full(inputs, yhp)
    
    # Boundary condition losses (no spatial derivatives needed)
    no_slip_loss = (torch.mean(yhp[no_slip_mask, 0]**2) + 
                   torch.mean(yhp[no_slip_mask, 1]**2)) * 500
    
    inlet_loss = (torch.mean((yhp[inlet_mask, 0] - u_avg)**2) +
                  torch.mean(yhp[inlet_mask, 1]**2)) * 200
    
    # Outlet pressure gradient - use precomputed gradients
    if outlet_mask.sum() > 0:
        dpdy_outlet = dpdy[outlet_mask, 0]  # dpdy was computed for all points
        outlet_loss = torch.mean(dpdy_outlet**2) * 250
    else:
        outlet_loss = torch.tensor(0.0, device=inputs.device)
    
    # Physics computation - use precomputed gradients, subset to interior
    if interior_mask.sum() > 0:
        # Subset everything to interior points
        u_int = yhp[interior_mask, 0:1]
        v_int = yhp[interior_mask, 1:2]
        dudx_int = dudx[interior_mask]
        dudy_int = dudy[interior_mask]
        dvdx_int = dvdx[interior_mask]
        dvdy_int = dvdy[interior_mask]
        dpdx_int = dpdx[interior_mask]
        dpdy_int = dpdy[interior_mask]
        d2udx2_int = d2udx2[interior_mask]
        d2udy2_int = d2udy2[interior_mask]
        d2vdx2_int = d2vdx2[interior_mask]
        d2vdy2_int = d2vdy2[interior_mask]
        
        # Navier-Stokes equations
        navier_x = (rho * (u_int * dudx_int + v_int * dudy_int) + 
                   dpdx_int - mu * (d2udx2_int + d2udy2_int))
        navier_y = (rho * (u_int * dvdx_int + v_int * dvdy_int) + 
                   dpdy_int - mu * (d2vdx2_int + d2vdy2_int))
        continuity = dudx_int + dvdy_int
        
        loss_phys = (torch.mean(navier_x**2) + torch.mean(navier_y**2) + 
                    torch.mean(continuity**2) * 1000)
    else:
        loss_phys = torch.tensor(0.0, device=inputs.device)
    
    total_loss = no_slip_loss + inlet_loss + outlet_loss + loss_phys
    
    return total_loss, loss_phys, no_slip_loss, inlet_loss, outlet_loss

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
poiseuille_model = Model(2, 3, 100, 12).to(device)
optimizer_a = torch.optim.Adam(poiseuille_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_a, "min", patience=1000)

# Logging
losses = {key: [] for key in ["tot", "phys", "no_slip", "inlet", "outlet"]}

total_points = 20000

full_tensor_np = generate_points(total_points)
full_tensor = torch.from_numpy(full_tensor_np).float().to(device).requires_grad_()

# Training loop
for i in trange(50000):

    optimizer_a.zero_grad()

    x_vals_tensor = full_tensor[:, 0]
    y_vals_tensor = full_tensor[:, 1]

    no_slip_mask = (y_vals_tensor < -h) | (y_vals_tensor > h)
    inlet_mask = (torch.abs(x_vals_tensor - 0) < 1e-2) & (~no_slip_mask)
    outlet_mask = ((L * 2 - x_vals_tensor) < 1e-2) & (~no_slip_mask)
    interior_mask = ~(no_slip_mask | inlet_mask | outlet_mask)
    masks = (no_slip_mask, inlet_mask, outlet_mask, interior_mask)
    
    loss, loss_phys, no_slip_loss, inlet_loss, outlet_loss = compute_loss(poiseuille_model, full_tensor, masks, i)

    loss.backward()
    optimizer_a.step()
    scheduler.step(loss)


    if i > 10000:
        losses["tot"].append(loss.item())
        losses["phys"].append(loss_phys.item())
        losses["no_slip"].append(no_slip_loss.item())
        losses["inlet"].append(inlet_loss.item())
        losses["outlet"].append(outlet_loss.item())

    if i % 5000 == 0 or i == 49999:        
        tqdm.write(f"Step {i+1}, Loss: {loss.item():.6f}\n Physics: {loss_phys.item():.6f}   No slip: {no_slip_loss.item():.6f}   Inlet: {inlet_loss.item():.6f}   Outlet: {outlet_loss.item():.6f}")

# Plotting losses
epochs = range(10000, len(losses["tot"]) + 10000)
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
