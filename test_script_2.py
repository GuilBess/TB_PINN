from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = os.curdir
print(path)

print(device)

print(torch.version.cuda)         # Shows CUDA version PyTorch was built with
print(torch.cuda.is_available())  # Should be True if GPU is usable

class Model(nn.Module):
    # defines a fully connected neural network, with a tanh activation function
    def __init__(self, inputs, outputs, hidden, n_layers):
        super().__init__()
        act_f = nn.Tanh

        # First layer, "*" unpacks the list into arguments of nn.Sequential
        self.fcs = nn.Sequential(*[
                        nn.Linear(inputs, hidden),
                        act_f()])
        
        # Hidden layers with activation function
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(hidden, hidden),
                            act_f()]) for _ in range(n_layers-1)])
        
        # Final layer that maps from last hidden layer to output size
        self.fce = nn.Linear(hidden, outputs)
    
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


def generate_points(N):
    # Define zones as (x_min, x_max, y_min, y_max, weight)
    zones = [
        (0, 0.4, -0.02, 0.02, 0.01),           # Base zone
        (0, 0.4, 0.01, 0.015, 0.03),          # Upper horizontal band
        (0, 0.4, -0.015, -0.01, 0.03),        # Lower horizontal band
        (0.19, 0.21, -0.01, 0.01, 0.05)           # Center square
    ]
    
    # Compute area and weighted area of each zone
    weighted_areas = []
    total_weighted_area = 0
    for x_min, x_max, y_min, y_max, weight in zones:
        area = (x_max - x_min) * (y_max - y_min)
        w_area = area * weight
        weighted_areas.append(w_area)
        total_weighted_area += w_area

    # Determine number of points per zone
    points_per_zone = [int((w_area / total_weighted_area) * N) for w_area in weighted_areas]

    # Sample points in each zone
    all_points = []
    for i, (x_min, x_max, y_min, y_max, _) in enumerate(zones):
        count = points_per_zone[i]
        xs = np.random.uniform(x_min, x_max, count)
        ys = np.random.uniform(y_min, y_max, count)
        all_points.append(np.column_stack((xs, ys)))

    # Combine all points into a single array
    points = np.vstack(all_points)

    return points

def compute_PDE(xy, y_in):
    u = y_in[:, 0:1]
    v = y_in[:, 1:2]
    p = y_in[:, 2:3]

    # Gradient helper
    grads = lambda out, inp: torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out), create_graph=True)[0]

    # First-order derivatives
    grads_u = grads(u, xy)
    grads_v = grads(v, xy)
    grads_p = grads(p, xy)

    dudx, dudy = grads_u[:, 0], grads_u[:, 1]
    dvdx, dvdy = grads_v[:, 0], grads_v[:, 1]
    dpdx, dpdy = grads_p[:, 0], grads_p[:, 1]

    # Second-order derivatives
    d2udx2 = grads(dudx, xy)[:, 0]
    d2udy2 = grads(dudy, xy)[:, 1]
    d2vdx2 = grads(dvdx, xy)[:, 0]
    d2vdy2 = grads(dvdy, xy)[:, 1]

    return dudx, dudy, dpdx, dpdy, d2udx2, d2udy2, dvdx, dvdy, d2vdx2, d2vdy2


# Simulation parameters
h = 0.01
L = 0.2
u_avg = 0.01
rho = 1000
mu = 0.001
Re = (rho * u_avg * 2*h) / mu
print(f"Reynolds number: {Re}")

h_factor = 2 # y dim simulation factor (y_max, y_min = +-h * h_factor)
L_factor = 2 # x dim simulation factor (x_max, x_min = L * L_factor, 0)

torch.manual_seed(12)
#model with 2 input (x,y position), 3 output layers (x, y velocity & pressure) and 8 hidden layers of 50 neurons
poiseuille_model = Model(2, 3, 100, 5).to(device)
optimizer = torch.optim.Adam(poiseuille_model.parameters(),lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

tensor_np = generate_points(50000)

tensor = torch.from_numpy(tensor_np).float().to(device).requires_grad_(True)

x_vals_tensor = tensor[:, 0]
y_vals_tensor = tensor[:, 1]

no_slip_mask = (y_vals_tensor < -h) | (y_vals_tensor > h)
inlet_mask = (torch.abs(x_vals_tensor - 0) < 1e-2) & (~no_slip_mask)
outlet_mask = (torch.abs(x_vals_tensor - L * L_factor) < 1e-2) & (~no_slip_mask)
interior_mask = ~(no_slip_mask | inlet_mask | outlet_mask)

vals_phys = []
vals_BC = []

for i in trange(10000):
    optimizer.zero_grad()
    
    # Boundray Conditions part
    yhp = poiseuille_model(tensor)

    p_outlet = yhp[outlet_mask, 2]

    outlet_loss = torch.mean(p_outlet**2) * 25
    no_slip_loss = torch.mean(yhp[no_slip_mask, 0])**2 + torch.mean(yhp[no_slip_mask, 1]**2) * 20
    inlet_loss = torch.mean((yhp[inlet_mask, 0] - u_avg)**2) + torch.mean((yhp[inlet_mask, 1])**2) * 20

    #Physics loss part
    x_interior = tensor[interior_mask]
    y_interior = poiseuille_model(x_interior)

    dudx, dudy, dpdx, dpdy, d2udx2, d2udy2, dvdx, dvdy, d2vdx2, d2vdy2 = compute_PDE(x_interior, y_interior)

    navier_x = rho*(y_interior[:, 0] * dudx + y_interior[:, 1] * dudy) + dpdx - mu * (d2udx2 + d2udy2)
    navier_y = rho*(y_interior[:, 0] * dvdx + y_interior[:, 1] * dvdy) + dpdy - mu * (d2vdx2 + d2vdy2)
    continuity = dudx + dvdy

    loss_phys = (torch.mean(navier_x**2) + torch.mean(navier_y**2) + torch.mean(continuity**2) * 1000)

    loss = loss_phys + 10 * (no_slip_loss + inlet_loss + outlet_loss)

    tensor_np = generate_points(50000)

    tensor = torch.from_numpy(tensor_np).float().to(device).requires_grad_(True)

    x_vals_tensor = tensor[:, 0]
    y_vals_tensor = tensor[:, 1]

    if i%1000 == 0 or i == 9999:
        tqdm.write(f"Step {i+1}, Loss: {loss.item():.6f}\n Physics: {loss_phys.item():.6f}   No slip: {no_slip_loss.item():.6f}   Inlet: {inlet_loss.item():.6f}   Outlet: {outlet_loss.item():.6f}")

    loss.backward()
    optimizer.step()
    scheduler.step()


torch.save(poiseuille_model.state_dict(), path)