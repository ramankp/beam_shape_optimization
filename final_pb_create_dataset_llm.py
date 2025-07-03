
import time
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from openai import AzureOpenAI
import keyring
import os
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import uniform_filter

# === Redirecting Output to a File ===
#output_file = "optimization_results_standard.txt"  # Specify the output file name
#original_stdout = sys.stdout  # Save the original stdout
#with open(output_file, "w") as f:
    #sys.stdout = f 

# Algorithmic parameters
#niternp = 10  # number of non-penalized iterations
niter = 40   # total number of iterations
exponent_update_frequency = 4  # minimum steps between exponent update
tol_mass = 1e-4  # tolerance on mass when finding Lagrange multiplier
thetamin = 0.001  # minimum density modeling void

# Algorithmic parameters
niternp_std = 20  # number of non-penalized iterations
niter_std = 80    # total number of iterations
pmax_std = 4      # maximum SIMP exponent
exponent_update_frequency_std = 4  # minimum steps between exponent update
tol_mass_std = 1e-4  # tolerance on mass when finding Lagrange multiplier
thetamin_std = 0.001  # minimum density modeling void


# Problem parameters
thetamoy = 0.4  # target average material density
E = Constant(1)
nu = Constant(0.3)
lamda = E*nu/(1+nu)/(1-2*nu)
mu = E/(2*(1+nu))
f = Constant((0, -1))  # vertical downwards force

thetamoy_std = 0.4  # target average material density
E_std = Constant(1)
nu_std = Constant(0.3)
lamda_std = E_std*nu_std/(1+nu_std)/(1-2*nu_std)
mu_std = E_std/(2*(1+nu_std))
f_std = Constant((0, -1))  # vertical downwards force


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


# Mesh
mesh_std = RectangleMesh(Point(-2, 0), Point(2, 1), 50, 30, "crossed")
# Boundaries
def left_std(x, on_boundary):
    return near(x[0], -2) and on_boundary
def load_std(x, on_boundary):
    return near(x[0], 2) and near(x[1], 0.5, 0.05)
facets_std = MeshFunction("size_t", mesh_std, 1)
AutoSubDomain(load_std).mark(facets_std, 1)
ds_std = Measure("ds", subdomain_data=facets_std)


# Function space for density field
V0 = FunctionSpace(mesh, "DG", 0)
# Function space for displacement
V2 = VectorFunctionSpace(mesh, "CG", 2)
# Fixed boundary conditions
bc = DirichletBC(V2, Constant((0, 0)), left)

# Function space for density field
V0_std = FunctionSpace(mesh_std, "DG", 0)
# Function space for displacement
V2_std = VectorFunctionSpace(mesh_std, "CG", 2)
# Fixed boundary conditions
bc_std = DirichletBC(V2_std, Constant((0, 0)), left_std)


p = Constant(1)  # SIMP penalty exponent
exponent_counter = 0  # exponent update counter
lagrange = Constant(1)  # Lagrange multiplier for volume constraint

p_std = Constant(1)  # SIMP penalty exponent
exponent_counter_std = 0  # exponent update counter
lagrange_std = Constant(1)  # Lagrange multiplier for volume constraint


thetaold = Function(V0, name="Density")
#thetaold.interpolate(Constant(thetamoy))

random_density = np.clip(
    np.random.normal(loc=thetamoy, scale=0.05, size=thetaold.vector().size()),
    thetamin, 1.0
)
thetaold.vector()[:] = random_density
print("hello")
print(random_density)
def coeff():
    return thetaold
theta = Function(V0)

volume = assemble(Constant(1.)*dx(domain=mesh))
avg_density_0 = assemble(thetaold*dx)/volume  # initial average density
avg_density = 0.
num_elements = mesh.num_cells()
print(f"Number of elements in the mesh: {num_elements}")
def eps(v):
    return sym(grad(v))
def sigma(v):
    #return coeff*(lamda*div(v)*Identity(2)+2*mu*eps(v))
    return coeff()*(lamda*div(v)*Identity(2)+2*mu*eps(v))
def energy_density(u, v):
    return inner(sigma(u), eps(v))


thetaold_std = Function(V0_std, name="Density")
thetaold_std.interpolate(Constant(thetamoy_std))
def coeff_std():
    return thetaold_std ** p_std
theta_std = Function(V0_std)

volume_std = assemble(Constant(1.)*dx(domain=mesh_std))
avg_density_0_std = assemble(thetaold_std*dx)/volume_std  # initial average density
avg_density_std = 0.
num_elements_std = mesh_std.num_cells()
print(f"Number of elements in the mesh: {num_elements_std}")
def eps_std(v):
    return sym(grad(v))
def sigma_std(v):
    return coeff_std()*(lamda_std*div(v)*Identity(2)+2*mu_std*eps_std(v))
def energy_density_std(u, v):
    return inner(sigma_std(u), eps_std(v))


# Inhomogeneous elastic variational problem
u_ = TestFunction(V2)
du = TrialFunction(V2)
a = inner(sigma(u_), eps(du))*dx
L = dot(f, u_)*ds(1)

# Inhomogeneous elastic variational problem
u__std = TestFunction(V2_std)
du_std = TrialFunction(V2_std)
a_std = inner(sigma_std(u__std), eps_std(du_std))*dx
L_std = dot(f_std, u__std)*ds_std(1)



# Print initial parameters to output file or console
print("\n--- Initial Parameters ---")
print(f"{'Algorithm Type:':<25} Standard")
print(f"{'Total Iterations:':<25} {niter}")
print(f"{'Target Avg Density (thetamoy):':<25} {thetamoy}")
print(f"{'Min Density (thetamin):':<25} {thetamin}")
print(f"{'Mesh Size (x, y):':<25} ({mesh.num_cells()})")
print(f"{'Material Properties:':<25} E = {float(E)}, nu = {float(nu)}")
print(f"{'Load Vector:':<25} f = {tuple(f.values())}")
print("--------------------------\n")



# === Azure OpenAI Setup ===
subscription_key = keyring.get_password("my_api_key_openai_1", "raman")
endpoint = os.getenv("ENDPOINT_URL", "https://ai-ramanstudentgpt41588930900787.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
client = AzureOpenAI(azure_endpoint=endpoint, api_key=subscription_key, api_version="2024-05-01-preview")


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

def local_project_std(v, V):
    dv_std = TrialFunction(V)
    v__std = TestFunction(V)
    a_proj_std = inner(dv_std, v__std)*dx
    b_proj_std = inner(v, v__std)*dx
    solver_std = LocalSolver(a_proj_std, b_proj_std)
    solver_std.factorize()
    u_std = Function(V)
    solver_std.solve_local_rhs(u_std)
    return u_std

def update_theta_std():
    theta_std.assign(local_project_std((p_std*coeff_std()*energy_density_std(u_std, u_std)/lagrange_std)**(1/(p_std+1)), V0_std))
    thetav_std = theta_std.vector().get_local()
    theta_std.vector().set_local(np.maximum(np.minimum(1, thetav_std), thetamin_std))
    theta_std.vector().apply("insert")
    avg_density_std = assemble(theta_std*dx)/volume_std
    return avg_density_std

def update_lagrange_multiplier_std(avg_density_std):
    avg_density1_std = avg_density_std
    # Initial bracketing of Lagrange multiplier
    if (avg_density1_std < avg_density_0_std):
        lagmin_std = float(lagrange_std)
        while (avg_density_std < avg_density_0_std):
            lagrange_std.assign(Constant(lagrange_std/2))
            avg_density_std = update_theta_std()
        lagmax_std = float(lagrange_std)
    elif (avg_density1_std > avg_density_0_std):
        lagmax_std = float(lagrange_std)
        while (avg_density_std > avg_density_0_std):
            lagrange_std.assign(Constant(lagrange_std*2))
            avg_density_std = update_theta_std()
        lagmin_std = float(lagrange_std)
    else:
        lagmin_std = float(lagrange_std)
        lagmax_std = float(lagrange_std)

    # Dichotomy on Lagrange multiplier
    inddico_std = 0
    while ((abs(1.-avg_density_std/avg_density_0_std)) > tol_mass_std):
        lagrange_std.assign(Constant((lagmax_std+lagmin_std)/2))
        avg_density_std = update_theta_std()
        inddico_std += 1
        if (avg_density_std < avg_density_0_std):
            lagmin_std = float(lagrange_std)
        else:
            lagmax_std = float(lagrange_std)
    print("   Dichotomy iterations:", inddico_std)


def update_exponent_std(exponent_counter_std):
    exponent_counter_std += 1
    if (std_iter < niternp_std):
        p_std.assign(Constant(1))
    elif (std_iter >= niternp_std):
        if std_iter == niternp_std:
            print("\n Starting penalized iterations\n")
        if ((abs(compliance_std-old_compliance_std) < 0.01*compliance_history_std[0]) and 
            (exponent_counter_std > exponent_update_frequency_std)):
            # average gray level
            gray_level_std = assemble((theta_std-thetamin_std)*(1.-theta_std)*dx)*4/volume_std
            p_std.assign(Constant(min(float(p_std)*(1+0.3**(1.+gray_level_std/2)), pmax_std)))
            exponent_counter_std = 0
            print("   Updated SIMP exponent p = ", float(p_std))
    return exponent_counter_std





def save_density_image(iteration):
    plot(thetaold, cmap="bone_r")
    plt.title(f"Density Distribution - Iteration {iteration}")
    plt.savefig(f"density_iter_{iteration}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
def get_density_azure_new(density_map, energy_map, energy_map_std, iteration):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_rows = density_map.shape[0]
    predicted_rows = [None] * num_rows

    label_bins = [0.001, 0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.01]
    labels = ["void", "very-low", "low", "sub-mid", "mid", "upper-mid", "dense", "solid"]
    label_to_value = {
        "void": 0.001, "very-low": 0.1, "low": 0.2, "sub-mid": 0.3,
        "mid": 0.4, "upper-mid": 0.55, "dense": 0.7, "solid": 0.85
    }


    def encode_row(row):
        return " ".join(labels[np.digitize(val, label_bins) - 1] for val in row)

    def safe_decode_row(label_row, default_row):
        allowed_labels = {"void", "very-low", "low", "sub-mid", "mid", "upper-mid", "dense", "solid"} 
        tokens = label_row.strip().split()

        # Filter out any unknown tokens
        tokens = [tok for tok in tokens if tok in allowed_labels]

        # Handle length mismatch
        if len(tokens) < 50:
            print(f"⚠️ Only {len(tokens)} tokens. Padding with 'low'.")
            tokens += ["low"] * (50 - len(tokens))
        elif len(tokens) > 50:
            print(f"⚠️ Got {len(tokens)} tokens. Trimming to 50.")
            tokens = tokens[:50]

        try:
            return np.array([label_to_value.get(tok, 0.001) for tok in tokens])
        except:
            print("⚠️ Token decoding failed. Using default row.")
            return default_row
    # Normalize energy row (e.g., between 0 and 1)
    def normalize_energy(energy_row):
        e_min, e_max = np.min(energy_row), np.max(energy_row)
        if e_max == e_min:
            return np.zeros_like(energy_row)
        return (energy_row - e_min) / (e_max - e_min + 1e-10)

    def process_row(row_idx):
        density_row = density_map[row_idx]
        energy_row = energy_map[row_idx]
        std_density_row = energy_map_std[row_idx]

        prev_row = density_map[row_idx - 1] if row_idx > 0 else np.zeros(50)
        next_row = density_map[row_idx + 1] if row_idx < 29 else np.zeros(50)

        # Encode both density and normalized energy
        density_tokens = encode_row(density_row)
        energy_norm = normalize_energy(energy_row)
        energy_tokens = encode_row(energy_norm)

        # Optionally encode neighboring rows for continuity
        prev_tokens = encode_row(prev_row) if prev_row is not None else ""
        next_tokens = encode_row(next_row) if next_row is not None else ""

        # Reference from standard optimization (optional)
        std_density_tokens = encode_row(std_density_row)

        prompt = f"""
        You are an AI assistant optimizing material layout for a 50×30 structure. Each element belongs to one of 8 discrete density classes (tokens), based on material concentration:
        - "void"       ≈ 0.001
        - "very-low"   ≈ 0.1
        - "low"        ≈ 0.2
        - "sub-mid"    ≈ 0.3
        - "mid"        ≈ 0.4
        - "upper-mid"  ≈ 0.55
        - "dense"      ≈ 0.7
        - "solid"      ≈ 0.85

        Each tokenized row has exactly 50 elements representing horizontal slices of the design.

        ---

        ### Input Data for Row {row_idx}:
        #### Current Density Map:
        {density_tokens}

        #### Normalized Strain Energy Map:
        {energy_tokens}

        #### Standard Optimizer Reference (for guidance):
        {std_density_tokens}

        #### Neighboring Rows:
        - Previous Row: {prev_tokens}
        - Next Row:     {next_tokens}

        ---

        ### Instructions:
        1. Output exactly **50 space-separated labels**.
        2. Use only these valid labels:
        `"void"`, `"very-low"`, `"low"`, `"sub-mid"`, `"mid"`, `"upper-mid"`, `"dense"`, `"solid"`
        3. High strain energy → use higher-density labels like `"solid"` or `"dense"`.
        4. Low strain energy → use lower-density labels like `"low"` or `"void"`.
        5. Preserve structural connectivity — avoid isolated high/low density.
        6. Match average density ≈ 0.4 → prefer `"mid"`, `"low"`, `"sub-mid"`.
        7. Follow the reference pattern when confident about structural logic.
        8. Ensure smooth transitions with neighboring rows — no abrupt changes.

        ---

        ### Output Format:
        Return exactly 50 space-separated tokens as described above.
        Do NOT include any extra text — just the 50 labels.

        """
        #print(prompt)
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained to optimize tokenized density maps."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=32767,
            temperature=0.0001
        )
        #print(response)
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            reply = response.choices[0].message.content.strip()
            print(reply)
        else:
            print(f"⚠️ No valid response for row {row_idx}")
            return row_idx, density_row  # fallback

        try:
            predicted_row = safe_decode_row(reply, default_row=density_row)
            if predicted_row.shape[0] != 50:
                #print(f"⚠️ Row {row_idx} had wrong size: {predicted_row.shape[0]}")
                #print(predicted_row)
                predicted_row = density_row
        except Exception as e:
            print(f"❌ Parsing failed for row {row_idx}: {e}")
            predicted_row = density_row

        return row_idx, predicted_row

    # Run all 30 rows
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_row, i) for i in range(num_rows)]
        for future in as_completed(futures):
            row_idx, row_data = future.result()
            predicted_rows[row_idx] = row_data

    return np.vstack(predicted_rows)

# Storage for dataset
dataset = {
    'density': [],     # t
    'energy': [],      # t
    'next_density': [],  # t+1
    'compliance': [],
    'method': []   # optional
}

u = Function(V2, name="Displacement")
old_compliance = 1e30
ffile = XDMFFile("topology_optimization.xdmf")
ffile.parameters["flush_output"] = True
ffile.parameters["functions_share_mesh"] = True
compliance_history = []

u_std = Function(V2_std, name="Displacement")
old_compliance_std = 1e30
ffile_std = XDMFFile("topology_optimization.xdmf")
ffile_std.parameters["flush_output"] = True
ffile_std.parameters["functions_share_mesh"] = True
compliance_history_std = []
output_file = "topopt_dataset_llm.npz"

print("\nStarting Optimization Process...\n")
niter_llm = 20
std_iters_per_llm = 4  # linear factor = 2x standard steps
niter_std = niter_llm * std_iters_per_llm

std_iter = 0  # tracks current standard iteration

for i_llm in range(niter_llm):
    print(f"\n=== LLM Iteration {i_llm + 1} ===")

    # -- Run std_iters_per_llm steps of STANDARD in each LLM iteration --
    for _ in range(std_iters_per_llm):
        if std_iter >= niter_std:
            break  # stop if standard has finished its full plan

        print(f"\n--- Standard Iteration {std_iter + 1} ---")
        # === STANDARD OPTIMIZATION ===
        solve(a_std == L_std, u_std, bc_std, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
        #ffile_std.write(thetaold_std, std_iter)
        #ffile_std.write(u_std, std_iter)

        compliance_std = assemble(action(L_std, u_std))
        compliance_history_std.append(compliance_std)
        print(f"Compliance_std: {compliance_std:.6f}")

        avg_density_std = update_theta_std()
        print(f"Average Density_std: {avg_density_std:.6f}")
        update_lagrange_multiplier_std(avg_density_std)

        exponent_counter_std = update_exponent_std(exponent_counter_std)
        print(f"SIMP Exponent_std (p): {float(p_std):.6f}")

        densities_std = theta_std.vector().get_local().reshape((1500, 4)).mean(axis=1)
        beam_density_map_std = densities_std.reshape((30, 50))

        local_energy_density_std = local_project_std(energy_density_std(u_std, u_std), V0_std).vector().get_local()
        energy1500_std = local_energy_density_std.reshape((1500, 4)).mean(axis=1)
        energy_map_2d_std = energy1500_std.reshape((30, 50))

        thetaold_std.assign(theta_std)
        old_compliance_std = compliance_std

        std_iter += 1

    # === LLM-GUIDED OPTIMIZATION ===
    print(f"\n--- Solving Elasticity for LLM (iteration {i_llm + 1}) ---")
    solve(a == L, u, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
    #ffile.write(thetaold, i_llm)
    #ffile.write(u, i_llm)

    compliance = assemble(action(L, u))
    compliance_history.append(compliance)
    print(f"Compliance_llm: {compliance:.6f}")

    densities = thetaold.vector().get_local().reshape((1500, 4)).mean(axis=1)
    beam_density_map = densities.reshape((30, 50))

    local_energy_density = local_project(energy_density(u, u), V0).vector().get_local()
    energy1500 = local_energy_density.reshape((1500, 4)).mean(axis=1)
    energy_map_2d = energy1500.reshape((30, 50))

    # === Only every 5 iterations, give std energy map to LLM ===
    if i_llm % 1 == 0:
        print("Calling get_density_azure_new with reference from standard")
        predicted_density_map = get_density_azure_new(
            beam_density_map,
            energy_map_2d,
            beam_density_map_std,  # from standard
            i_llm
        )
    else:
        continue
        predicted_density_map = get_density_azure(beam_density_map, energy_map_2d, i_llm)

    #current_density = thetaold.vector().get_local().reshape((1500, 4)).mean(axis=1).reshape((30, 50))

    # Extract energy map
    #local_energy = local_project(energy_density(u, u), V0).vector().get_local()
    #energy_map = local_energy.reshape((1500, 4)).mean(axis=1).reshape((30, 50))

    # Save next density (after update_theta())
    #next_density = theta.vector().get_local().reshape((1500, 4)).mean(axis=1).reshape((30, 50))

    # Append current LLM step to dataset
    dataset['density'].append(beam_density_map)
    dataset['energy'].append(energy_map_2d)
    dataset['next_density'].append(predicted_density_map)
    dataset['compliance'].append(compliance)
    dataset['method'].append("llm")  # mark as LLM method

    # Save periodically
    if (i_llm + 1) % 1 == 0 or i_llm == niter_llm - 1:
        print(f"Appending dataset at LLM iteration {i_llm + 1}...")

        # Load old data if exists
        if os.path.exists(output_file):
            with np.load(output_file) as old_data:
                old_Xd = old_data['density']
                old_Xe = old_data['energy']
                old_yd = old_data['next_density']
                old_comp = old_data['compliance']
                old_meth = old_data['method'] if 'method' in old_data else np.array(['std'] * len(old_data['density']))
        else:
            old_Xd = old_Xe = old_yd = old_comp = old_meth = np.empty((0,), dtype=np.float32)

        # Convert lists to numpy arrays
        new_Xd = np.array(dataset['density'])
        new_Xe = np.array(dataset['energy'])
        new_yd = np.array(dataset['next_density'])
        new_comp = np.array(dataset['compliance'])
        new_meth = np.array(dataset['method'])

        # Normalize energy maps
        if len(new_Xe) > 0:
            X_energy_min = new_Xe.min(axis=(1,2), keepdims=True)
            X_energy_max = new_Xe.max(axis=(1,2), keepdims=True)
            new_Xe = (new_Xe - X_energy_min) / (X_energy_max - X_energy_min + 1e-8)

        # Concatenate old and new
        final_Xd = np.concatenate([old_Xd, new_Xd], axis=0)
        final_Xe = np.concatenate([old_Xe, new_Xe], axis=0)
        final_yd = np.concatenate([old_yd, new_yd], axis=0)
        final_comp = np.concatenate([old_comp, new_comp], axis=0)
        final_meth = np.concatenate([old_meth, new_meth], axis=0)

        # Save back to .npz file
        np.savez_compressed(
            output_file,
            density=final_Xd,
            energy=final_Xe,
            next_density=final_yd,
            compliance=final_comp,
            method=final_meth
        )

        print(f"Dataset appended and saved to '{output_file}'")
        print(f"Total samples now in dataset: {len(final_Xd)}")

        # Reset buffer
        dataset = {
            'density': [],
            'energy': [],
            'next_density': [],
            'compliance': [],
            'method': []
        }


    # === Process and smooth ===
    predicted_density_map = np.clip(predicted_density_map, 0.001, 1.0)
    beam_density_map_smoothed = uniform_filter(predicted_density_map, size=3)

    # if i_llm > 15:
    #     beam_density_map_smoothed = (beam_density_map_smoothed >= 0.5).astype(float)
    #     beam_density_map_smoothed = np.clip(beam_density_map_smoothed, 0.001, 1.0)  # ⚠️ prevent zero-density

    if i_llm % 5 == 0:
        save_density_image(i_llm)
    expanded_density = np.repeat(beam_density_map_smoothed.flatten(), 4)
    thetaold.vector().set_local(expanded_density)
    thetaold.vector().apply("insert")
    # plot(thetaold, cmap="bone_r")
    # plt.title("Final Density Distribution")
    # plt.show()
    avg_density = assemble(thetaold * dx) / volume
    print(f"Average Density_llm: {avg_density:.6f}")
    old_compliance = compliance
    if i_llm == 16:
        break

plot(thetaold, cmap="bone_r")
plt.title("Final Density Distribution")
plt.show()



