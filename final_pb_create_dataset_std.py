
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import os
from random import random

# === Output File for Dataset ===
output_file = "topopt_dataset.npz"

# Algorithmic parameters
niternp = 20
niter = 80
pmax = 4
exponent_update_frequency = 4
tol_mass = 1e-4
thetamin = 0.001

# Problem parameters
thetamoy = 0.4
E = Constant(1)
nu = Constant(0.3)
lamda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
f = Constant((0, -1))

# Mesh
mesh = RectangleMesh(Point(-2, 0), Point(2, 1), 50, 30, "crossed")

# Boundaries
def left(x, on_boundary):
    return near(x[0], -2) and on_boundary
def load(x, on_boundary):
    return near(x[0], 2) and near(x[1], 0.5, 0.05)
facets = MeshFunction("size_t", mesh, 1)
AutoSubDomain(load).mark(facets, 1)
ds = Measure("ds", subdomain_data=facets)

# Function spaces
V0 = FunctionSpace(mesh, "DG", 0)
V2 = VectorFunctionSpace(mesh, "CG", 2)
bc = DirichletBC(V2, Constant((0, 0)), left)

# Variables
p = Constant(1)
exponent_counter = 0
lagrange = Constant(1)

thetaold = Function(V0, name="Density")
random_density = np.clip(
    np.random.normal(loc=thetamoy, scale=0.05, size=thetaold.vector().size()),
    thetamin, 1.0
)
thetaold.vector()[:] = random_density

def coeff():
    return thetaold ** p
theta = Function(V0)

volume = assemble(Constant(1.)*dx(domain=mesh))
avg_density_0 = assemble(thetaold*dx)/volume
num_elements = mesh.num_cells()

def eps(v):
    return sym(grad(v))
def sigma(v):
    return coeff()*(lamda*div(v)*Identity(2)+2*mu*eps(v))
def energy_density(u, v):
    return inner(sigma(u), eps(v))

# Variational problem
u_ = TestFunction(V2)
du = TrialFunction(V2)
a = inner(sigma(u_), eps(du))*dx
L = dot(f, u_)*ds(1)

# Output file
ffile = XDMFFile("topology_optimization.xdmf")
ffile.parameters["flush_output"] = True
ffile.parameters["functions_share_mesh"] = True

# Storage for dataset
dataset = {
    'density': [],     # t
    'energy': [],      # t
    'next_density': [],  # t+1
    'compliance': []   # optional
}

def local_project(v, V):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    return u

def update_theta():
    theta.assign(local_project((p*coeff()*energy_density(u, u)/lagrange)**(1/(p+1)), V0))
    thetav = theta.vector().get_local()
    theta.vector().set_local(np.maximum(np.minimum(1, thetav), thetamin))
    theta.vector().apply("insert")
    avg_density = assemble(theta*dx)/volume
    return avg_density

def update_lagrange_multiplier(avg_density):
    avg_density1 = avg_density
    if avg_density1 < avg_density_0:
        lagmin = float(lagrange)
        while avg_density < avg_density_0:
            lagrange.assign(Constant(lagrange/2))
            avg_density = update_theta()
        lagmax = float(lagrange)
    elif avg_density1 > avg_density_0:
        lagmax = float(lagrange)
        while avg_density > avg_density_0:
            lagrange.assign(Constant(lagrange*2))
            avg_density = update_theta()
        lagmin = float(lagrange)
    else:
        lagmin = float(lagrange)
        lagmax = float(lagrange)

    inddico = 0
    while abs(1. - avg_density/avg_density_0) > tol_mass:
        lagrange.assign(Constant((lagmax + lagmin)/2))
        avg_density = update_theta()
        inddico += 1
        if avg_density < avg_density_0:
            lagmin = float(lagrange)
        else:
            lagmax = float(lagrange)
    print("   Dichotomy iterations:", inddico)

def update_exponent(exponent_counter):
    exponent_counter += 1
    if i < niternp:
        p.assign(Constant(1))
    elif i >= niternp:
        if i == niternp:
            print("\n Starting penalized iterations\n")
        if abs(compliance-old_compliance) < 0.01*compliance_history[0] and exponent_counter > exponent_update_frequency:
            gray_level = assemble((theta-thetamin)*(1.-theta)*dx)*4/volume
            p.assign(Constant(min(float(p)*(1+0.3**(1.+gray_level/2)), pmax)))
            exponent_counter = 0
            print("   Updated SIMP exponent p =", float(p))
    return exponent_counter

# Optimization loop
u = Function(V2, name="Displacement")
old_compliance = 1e30
compliance_history = []

print("\nStarting Optimization Process...\n")

for i in range(niter):
    print(f"\n--- Iteration {i + 1} ---")
    solve(a == L, u, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
    ffile.write(thetaold, i)
    ffile.write(u, i)

    compliance = assemble(action(L, u))
    compliance_history.append(compliance)
    print(f"Compliance: {compliance:.6f}")

    avg_density = update_theta()
    print(f"Average Density: {avg_density:.6f}")
    print(f"Lagrange Multiplier: {float(lagrange):.6f}")

    update_lagrange_multiplier(avg_density)

    exponent_counter = update_exponent(exponent_counter)
    print(f"SIMP Exponent (p): {float(p):.6f}")

    # Extract non-penalized current density (before applying SIMP)
    current_density = thetaold.vector().get_local().reshape((1500, 4)).mean(axis=1).reshape((30, 50))

    # Extract energy map
    local_energy = local_project(energy_density(u, u), V0).vector().get_local()
    energy_map = local_energy.reshape((1500, 4)).mean(axis=1).reshape((30, 50))

    # Save next density (after update_theta())
    next_density = theta.vector().get_local().reshape((1500, 4)).mean(axis=1).reshape((30, 50))

    # Append to dataset
    dataset['density'].append(current_density)
    dataset['energy'].append(energy_map)
    dataset['next_density'].append(next_density)
    dataset['compliance'].append(compliance)

    # Optional: Save periodically
    if (i + 1) % 10 == 0 or i == niter - 1:
        print(f"Appending dataset at iteration {i + 1}...")

        # Load old data if exists
        if os.path.exists(output_file):
            old_data = np.load(output_file)
            old_Xd = old_data['density']
            old_Xe = old_data['energy']
            old_yd = old_data['next_density']
            old_comp = old_data['compliance']
        else:
            old_Xd = old_Xe = old_yd = np.empty((0, 30, 50), dtype=np.float32)
            old_comp = np.empty((0,), dtype=np.float32)

        # Remove last sample to ensure valid pairs
        X_density = np.array(dataset['density'])[:-1]
        X_energy = np.array(dataset['energy'])[:-1]
        y_density = np.array(dataset['next_density'])[1:]
        comp_values = np.array(dataset['compliance'])[1:]

        # Normalize energy maps
        if len(X_energy) > 0:
            X_energy = (X_energy - X_energy.min()) / (X_energy.max() - X_energy.min() + 1e-8)

        # Concatenate old and new
        new_Xd = np.concatenate([old_Xd, X_density], axis=0)
        new_Xe = np.concatenate([old_Xe, X_energy], axis=0)
        new_yd = np.concatenate([old_yd, y_density], axis=0)
        new_comp = np.concatenate([old_comp, comp_values], axis=0)

        # Save back to .npz file
        np.savez_compressed(
            output_file,
            density=new_Xd,
            energy=new_Xe,
            next_density=new_yd,
            compliance=new_comp
        )
        print(f"Dataset appended and saved to '{output_file}'")

        # Clear current dataset buffer
        dataset = {
            'density': [],
            'energy': [],
            'next_density': [],
            'compliance': []
        }

    # Update theta field
    thetaold.assign(theta)
    old_compliance = compliance


# Final visualization
plot(thetaold, cmap="bone_r")
plt.title("Final Density Distribution")
plt.show()