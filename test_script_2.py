# Simulation parameters
h = 0.01
L = 0.2
u_avg = 0.01
rho = 1000
mu = 0.001

torch.manual_seed(12)
#model with 2 input (x,y position), 3 output layers (x, y velocity & pressure) and 8 hidden layers of 50 neurons
poiseuille_model = Model(2, 3, 50, 8).to(device)
optimizer = torch.optim.Adam(poiseuille_model.parameters(),lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

x_min, x_max = 0, L  # Horizontal span
y_min, y_max = -h, h  # Vertical span

# Getting x values, with higer density to the left
n_x = 100
skew_x = np.linspace(0, 1, n_x)  # uniform base
x_vals = x_min + (x_max - x_min) * (0.5 - 0.5 * np.cos(np.pi * skew_x)) # skew to the left

# Getting y values, with higher density on the top/bottom
n_y = 100
skew_y = np.linspace(0, 1, n_y)
y_vals = y_min + (y_max - y_min) * (0.5 - 0.5 * np.cos(np.pi * skew_y))

X, Y = np.meshgrid(x_vals, y_vals)

tensor_np = np.column_stack((X.ravel(), Y.ravel()))

tensor = torch.from_numpy(tensor_np).float().to(device).requires_grad_(True)

plt.scatter(tensor_np[:, 0], tensor_np[:, 1], s=1, color='blue')


def compute_PDE(t_in, y_in):
    # Extract and mask
    x = t_in[:, 0:1] # shape [M, 1]
    y = t_in[:, 1:2]
    xy = torch.cat([x, y], dim=1).requires_grad_(True)  # Rebuild masked input for gradient tracking

    u = y_in[:, 0:1]
    v = y_in[:, 1:2]
    p = y_in[:, 2:3]

    print(u.requires_grad, xy.requires_grad)

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

x_vals_tensor = tensor[:, 0]
y_vals_tensor = tensor[:, 1]

no_slip_mask = (torch.abs(y_vals_tensor - h) < 5e-4) | (torch.abs(y_vals_tensor + h) < 5e-4)
inlet_mask = (torch.abs(x_vals_tensor - x_min) < 1e-6) & (~no_slip_mask)
outlet_mask = (torch.isclose(tensor[:, 0], torch.tensor(x_max).to(tensor), atol=1e-6)) & (~no_slip_mask)
interior_mask = ~(no_slip_mask | inlet_mask | outlet_mask)

for i in range(20000):
    optimizer.zero_grad()
    
    # compute the "physics loss"
    yhp = poiseuille_model(tensor)
    int_x = tensor[interior_mask]
    int_y = yhp[interior_mask]

    print(int_x.requires_grad, int_y.requires_grad)
    
    dudx, dudy, dpdx, dpdy, d2udx2, d2udy2, dvdx, dvdy, d2vdx2, d2vdy2 = compute_PDE(int_x, int_y)

    navier_x = rho*(int_y[:, 0] * dudx + int_y[:, 1] * dudy) + dpdx - mu * (d2udx2 + d2udy2)
    navier_y = rho*(int_y[:, 0] * dvdx + int_y[:, 1] * dvdy) + dpdy - mu * (d2vdx2 + d2vdy2)
    continuity = dudx + dvdy

    # Extract pressure predictions at those points
    p_outlet = yhp[outlet_mask, 2]

    # Penalize pressure deviation from zero
    outlet_loss = torch.mean(p_outlet**2) * 5  # the weight 5 is tunable

    no_slip_loss = torch.mean(torch.abs(yhp[no_slip_mask, 0]))**2 + torch.mean(torch.abs(yhp[no_slip_mask, 1])**2)

    inlet_loss = torch.mean(torch.abs(yhp[inlet_mask, 0] - u_avg))**2 + torch.mean(torch.abs(yhp[inlet_mask, 1])**2)

    loss = (torch.mean(navier_x**2) + torch.mean(navier_y**2) + torch.mean(continuity**2))

    loss = loss + no_slip_loss + inlet_loss + outlet_loss

    if i%5000 == 4999:
        print(loss)

    loss.backward()
    optimizer.step()
    scheduler.step()