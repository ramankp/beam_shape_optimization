
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from model import TopOptGNN  

# === Parameters ===
niternp = 20
niter = 80
pmax = 4
thetamin = 0.001
thetamoy = 0.4
E = Constant(1)
nu = Constant(0.3)
lamda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / (2 * (1 + nu))
f = Constant((0, -1))

# === Mesh and Function Spaces ===
mesh = RectangleMesh(Point(-2, 0), Point(2, 1), 50, 30, "crossed")
V0 = FunctionSpace(mesh, "DG", 0)
V2 = VectorFunctionSpace(mesh, "CG", 2)

def left(x, on_boundary): return near(x[0], -2) and on_boundary
def load(x, on_boundary): return near(x[0], 2) and near(x[1], 0.5, 0.05)

facets = MeshFunction("size_t", mesh, 1)
AutoSubDomain(load).mark(facets, 1)
ds = Measure("ds", subdomain_data=facets)

bc = DirichletBC(V2, Constant((0, 0)), left)

# === Initial Fields ===
p = Constant(1)
lagrange = Constant(1)
thetaold = Function(V0, name="Density")
thetaold.interpolate(Constant(thetamoy))
theta = Function(V0)
volume = assemble(Constant(1.0) * dx(domain=mesh))
avg_density_0 = assemble(thetaold * dx) / volume
u = Function(V2, name="Displacement")
u_ = TestFunction(V2)
du = TrialFunction(V2)

# === GNN Load ===
def build_grid_edges(h, w):
    edges = []
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    nidx = ni * w + nj
                    edges.append((idx, nidx))
    return torch.tensor(edges, dtype=torch.long).t()

edge_index = build_grid_edges(30, 50)

gnn_model = TopOptGNN()
gnn_model.load_state_dict(torch.load("topopt_gnn_model_new.pth", map_location=torch.device("cpu")))
gnn_model.eval()

# === Elasticity and Projection ===
def eps(v): return sym(grad(v))
def sigma(v): return (thetaold ** p) * (lamda * div(v) * Identity(2) + 2 * mu * eps(v))
def energy_density(u, v): return inner(sigma(u), eps(v))

a = inner(sigma(u_), eps(du)) * dx
L = dot(f, u_) * ds(1)

def local_project(v, V):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_) * dx
    b_proj = inner(v, v_) * dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    return u

# === Optimization Loop ===
compliance_history = []
old_compliance = 1e30
exponent_counter = 0

print("\n🔁 Starting GNN-based Topology Optimization...\n")

for i in range(niter):
    print(f"\n--- Iteration {i + 1} ---")

    solve(a == L, u, bc)

    compliance = assemble(action(L, u))
    compliance_history.append(compliance)

    # Energy and Density Maps
    energy_local = local_project(energy_density(u, u), V0).vector().get_local()
    energy1500 = energy_local.reshape((1500, 4)).mean(axis=1)
    energy_map = energy1500.reshape((30, 50))

    density_local = thetaold.vector().get_local().reshape((1500, 4)).mean(axis=1)
    density_map = density_local.reshape((30, 50))

    # === GNN Prediction ===
    print("🔮 Predicting next density using GNN...")
    x = np.stack([density_map.flatten(), energy_map.flatten()], axis=1)
    graph = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index)
    
    with torch.no_grad():
        pred = gnn_model(graph.x, graph.edge_index).numpy()
    
    pred_density = pred.reshape((30, 50))
    pred_density = np.clip(pred_density, thetamin, 1.0)

    # === SIMP Penalization and Volume Correction ===
    if i >= niternp:
        pred_density = pred_density ** float(p)
        pred_density *= avg_density_0 / np.mean(pred_density)
        pred_density = np.clip(pred_density, thetamin, 1.0)

    # === Apply to θ ===
    pred_flat = pred_density.flatten()
    pred_expanded = np.repeat(pred_flat, 4)

    relaxation_weight = 0.5
    theta_old_np = thetaold.vector().get_local()
    blended_theta = (1 - relaxation_weight) * theta_old_np + relaxation_weight * pred_expanded
    blended_theta = np.clip(blended_theta, thetamin, 1.0)

    theta.vector().set_local(blended_theta)
    theta.vector().apply("insert")

    avg_density = assemble(theta * dx) / volume
    print(f"📏 Avg Density: {avg_density:.4f} | Compliance: {compliance:.4f}")

    # === Update SIMP exponent ===
    def update_exponent(counter, i, compliance, old_compliance, history):
        counter += 1
        if i < niternp:
            p.assign(Constant(1))
        else:
            if abs(compliance - old_compliance) < 0.01 * history[0] and counter > 4:
                gray = assemble((theta - thetamin)*(1 - theta)*dx) * 4 / volume
                p.assign(Constant(min(float(p) * (1 + 0.3 ** (1 + gray / 2)), pmax)))
                counter = 0
                print(f"⚡️ Updated SIMP exponent p = {float(p):.3f}")
        return counter

    exponent_counter = update_exponent(exponent_counter, i, compliance, old_compliance, compliance_history)
    p.assign(Constant(min(float(p), pmax)))

    thetaold.assign(theta)
    old_compliance = compliance

    # === Plot every 10 iterations ===
    if (i + 1) % 10 == 0 or i == niter - 1:
        plot(theta, cmap="bone_r")
        plt.title(f"Density Iteration {i + 1}")
        plt.show()

# === Final Results ===
plot(theta, cmap="bone_r")
plt.title("Final Density Distribution")
plt.show()

plt.figure()
plt.plot(np.arange(1, niter + 1), compliance_history)
plt.xlabel("Iteration")
plt.ylabel("Compliance")
plt.title("Convergence History")
plt.grid(True)
plt.show()
