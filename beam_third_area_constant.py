import os
import numpy as np
from fenics import *
import re
from openai import AzureOpenAI
import keyring
import time
import json
from scipy.optimize import differential_evolution

# === Material Properties (Aluminum) ===
E = 70e9        # Young's modulus (70 GPa)
rho = 2700      # Density (2700 kg/m³)
nu = 0.33       # Poisson's ratio
# === Beam Parameters ===
Len = 1.5       # Length (m)
h_min, h_max = 0.001, 0.05  # Thickness bounds (1 mm to 5 cm)
b_min, b_max = 0.001, 0.05  # Width bounds (1 mm to 5 cm)
P = 1000        # Point load (N)
# === Fixed Cross-Section Area ===
fixed_area = 0.0001  # Fixed cross-sectional area (m²)

# === Constraints ===
max_iterations = 50
tolerance = 1e-4

# === Mesh Parameters ===
num_elements_length = lambda: max(50, int(Len / 0.01))  # Dynamic resolution
num_elements_height = lambda h: max(5, int(h / 0.005))

# === Azure OpenAI Setup ===
#subscription_key = keyring.get_password("my_api_key_openai", "raman")
#endpoint = os.getenv("ENDPOINT_URL", "https://ai-hmsmstudentsraman366400147319.openai.azure.com/")
#deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
#client = AzureOpenAI(azure_endpoint=endpoint, api_key=subscription_key, api_version="2024-05-01-preview")

subscription_key = keyring.get_password("my_api_key_openai_1", "raman")
endpoint = os.getenv("ENDPOINT_URL", "https://ai-ramanstudentgpt41588930900787.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
client = AzureOpenAI(azure_endpoint=endpoint, api_key=subscription_key, api_version="2024-05-01-preview")

# === Plane Strain Formulations ===
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    mu = E / (2 * (1 + nu))
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))  # Plane strain lambda
    return lambda_ * tr(epsilon(u)) * Identity(2) + 2 * mu * epsilon(u)

def von_mises_stress(u):
    s = sigma(u) - (1/3) * tr(sigma(u)) * Identity(2)
    return sqrt(3/2 * inner(s, s))

# === Boundary Conditions ===
class FixedEnd(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class FreeEnd(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Len) and on_boundary

# === Query Azure OpenAI for Both h and b ===
def query_azure_openai(prompt, current_h, current_b):
    try:
        messages = [
            {"role": "system", "content": "You are an AI assistant optimizing beam dimensions (h=thickness, b=width)."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=40,
            temperature=0.1,
            top_p=0.95
        )
        generated_text = response.choices[0].message.content.strip()
        print(f"Azure OpenAI Response: {generated_text}")
        # Clean response: Retain only digits, dots, spaces, minus signs, and scientific notation
        cleaned = re.sub(r"[^\d.\s\-eE]", "", generated_text)
        cleaned = " ".join(cleaned.split())  # Remove duplicate spaces
        print(f"Cleaned Response: {cleaned}")
        # Extract matches
        matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
        print(f"Extracted Matches: {matches}")
        if len(matches) >= 2:
            try:
                suggested_h = float(matches[0])
                suggested_b = float(matches[1])
                print(f"Parsed Values: h={suggested_h}, b={suggested_b}")
                # Validate bounds and area constraint
                epsilon = 1e-6
                if (h_min - epsilon <= suggested_h <= h_max + epsilon and
                    b_min - epsilon <= suggested_b <= b_max + epsilon and
                    abs(suggested_h * suggested_b - fixed_area) < 1e-4):
                    return suggested_h, suggested_b
                else:
                    raise ValueError("Suggested values out of bounds or violate area constraint")
            except ValueError as ve:
                print(f"Validation Error: {ve}")
                raise  # Re-raise the exception to trigger fallback logic
        else:
            print("Insufficient matches found.")
            raise ValueError("Invalid response format")
    except Exception as e:
        print(f"Error querying Azure OpenAI: {e}")
        print(f"Problematic Response: {generated_text}")
        print(f"Cleaned Response: {cleaned}")
        print(f"Extracted Matches: {matches}")
        print(prompt)
        # Fallback heuristic adjustment
        perturbation = 0.001  # Small perturbation to avoid stagnation
        new_h = current_h * (1 + np.random.uniform(-perturbation, perturbation))
        new_b = fixed_area / new_h  # Recompute b to enforce area constraint
        return np.clip(new_h, h_min, h_max), np.clip(new_b, b_min, b_max)

# === Objective Function ===
def objective(x):
    h, b = x
    # Enforce fixed cross-sectional area constraint
    if abs(h * b - fixed_area) > 1e-6:  # Allow small numerical tolerance
        return 1e9  # Penalize violations of the area constraint
    
    # Out-of-bounds penalty
    if not (h_min <= h <= h_max and b_min <= b <= b_max):
        return 1e9
    
    # Re-mesh based on current height
    mesh = RectangleMesh(
        Point(0, 0), 
        Point(Len, h),
        num_elements_length(),
        num_elements_height(h)
    )
    V = VectorFunctionSpace(mesh, 'Lagrange', degree=2)
    V_scalar = FunctionSpace(mesh, 'P', 1)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    FixedEnd().mark(boundaries, 1)
    FreeEnd().mark(boundaries, 2)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    bc = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant((0.0, -rho * 9.81 * b))  # Gravity
    traction = Constant((0.0, -P / h))
    a = inner(sigma(u), epsilon(v)) * dx
    L = inner(f, v) * dx + inner(traction, v) * ds(2)
    A, b_matrix = assemble_system(a, L, bc)
    u_sol = Function(V)
    solve(A, u_sol.vector(), b_matrix)
    u_mag = project(sqrt(dot(u_sol, u_sol)), V_scalar)
    u_max = float(np.amax(u_mag.vector().get_local()))
    
    # Constraint: Stress
    vm_stress_expr = von_mises_stress(u_sol)
    vm_proj = project(vm_stress_expr, V_scalar)
    max_vm = float(np.amax(vm_proj.vector().get_local()))
    yield_stress = 0.002 * E
    if max_vm > yield_stress:
        return 1e6 + max_vm
    
    # Objective: Minimize deflection
    return u_max

# === Main Optimization Loop ===
iteration_history = []
h_opt, b_opt = 0.02, 0.02  # Initial guesses
converged = False
start_time_llm = time.time()
for i in range(max_iterations):
    print(f"\n--- Iteration {i + 1} ---")
    # Query Azure OpenAI for both h and b
    prompt = f"""
    I am solving for the optimal thickness (h) and width (b) of a cantilever beam under bending.
    The goal is to minimize the maximum deflection while satisfying the following constraints:
    - Maximum von Mises stress ≤ {0.002 * E:.2f} Pa.
    - Cross-sectional area must be exactly {fixed_area} m² (h*b = {fixed_area}).
    - Thickness (h) must be between {h_min} m and {h_max} m.
    - Width (b) must be between {b_min} m and {b_max} m.
    A point load of 1000 N is applied at the free end of the beam.
    Previous iterations:
    """
    for hist in iteration_history:
        prompt += f"At h={hist[0]:.4f}, b={hist[1]:.4f}: deflection={hist[2]:.6f} m, von Mises stress={hist[3]:.2f} Pa\n"
    prompt += (
        f"Suggest the next thickness (h) and width (b) values that minimize deflection without violating the constraints. "
        f"The beam must have a thickness (h) greater than the width (b) to ensure proper structural stability."
        f"Ensure the constraints are satisfied: h ∈ [{h_min:.4f}, {h_max:.4f}], b ∈ [{b_min:.4f}, {b_max:.4f}], von Mises stress ≤ {0.002 * E:.2f}. "
        f"Make sure h*b = {fixed_area}. "
        f"Provide exactly two numbers (h and b), separated by a space, with no additional text or explanations."
    )
    print(prompt)
    h_prev, b_prev = h_opt, b_opt
    h_opt, b_opt = query_azure_openai(prompt, h_prev, b_prev)
    
    # Solve the finite element problem with updated h and b
    mesh = RectangleMesh(
        Point(0, 0), 
        Point(Len, h_opt),
        num_elements_length(),
        num_elements_height(h_opt)
    )
    V = VectorFunctionSpace(mesh, 'Lagrange', degree=2)
    V_scalar = FunctionSpace(mesh, 'P', 1)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    fixed_end = FixedEnd()
    fixed_end.mark(boundaries, 1)  # Mark fixed boundary
    free_end = FreeEnd()
    free_end.mark(boundaries, 2)  # Mark free boundary
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    bc = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant((0.0, -rho * 9.81 * b_opt))  # Body force (N/m³)
    traction = Constant((0.0, -P / h_opt))
    a = inner(sigma(u), epsilon(v)) * dx
    L = inner(f, v) * dx + inner(traction, v) * ds(2)  # Apply traction on free end
    A, b_matrix = assemble_system(a, L, bc)
    u_sol = Function(V)
    solve(A, u_sol.vector(), b_matrix)
    u_mag = project(sqrt(dot(u_sol, u_sol)), V_scalar)
    u_max = float(np.amax(u_mag.vector().get_local()))
    von_mises_expr = von_mises_stress(u_sol)
    von_mises_stress_field = project(von_mises_expr, V_scalar)
    max_stress = np.max(von_mises_stress_field.vector().get_local())
    yield_stress = 0.002 * E
    stress_exceeded = project(conditional(gt(von_mises_stress_field, yield_stress), 1.0, 0.0), V_scalar)
    File(f"results/displacement_{i}.pvd") << u_sol
    File(f"results/von_mises_{i}.pvd") << von_mises_stress_field
    File(f"results/stress_exceeded_{i}.pvd") << stress_exceeded
    
    iteration_history.append((h_opt, b_opt, u_max, max_stress))
    print(f"h={h_opt:.4f}, b={b_opt:.4f}, Deflection={u_max:.6f}, Stress={max_stress:.2f} Pa")
    
    # Check convergence
    if (abs(h_opt - h_prev) < tolerance and abs(b_opt - b_prev) < tolerance and max_stress <= yield_stress):
        print("\nConverged to optimal solution!")
        converged = True
        break

if not converged:
    print("Max iterations reached without full convergence")
time_llm = time.time() - start_time_llm
# Final results
print("\n--- Final Results ---")
print(f"Optimal Thickness: {h_opt:.4f} m")
print(f"Optimal Width: {b_opt:.4f} m")
print(f"Max Deflection: {u_max:.6f} m")
print(f"Weight: {rho*Len*h_opt*b_opt} kg")
print(f"Max von Mises Stress: {max_stress*1e-6:.2f} MPa")

# === Bounds and Initialization ===
bounds = [(h_min, h_max), (b_min, b_max)]
init_guess = [[h_opt, b_opt]]  # Use best result from AI loop

def generate_initial_population(h_opt, b_opt, bounds, size=30, noise_scale=0.05):
    pop = []
    for _ in range(size):
        h_perturbed = np.clip(h_opt * (1 + noise_scale * np.random.randn()), bounds[0][0], bounds[0][1])
        b_perturbed = np.clip(b_opt * (1 + noise_scale * np.random.randn()), bounds[1][0], bounds[1][1])
        pop.append([h_perturbed, b_perturbed])
    return np.array(pop)

init_pop = generate_initial_population(h_opt, b_opt, bounds)
print("\nStarting Differential Evolution Optimization...\n")

start_time_ai = time.time()
result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=40,
    tol=1e-4,
    polish=True,
    updating='deferred',
    init=init_pop,
    disp=True
)
time_ai = time.time() - start_time_ai
print(result)
h_best, b_best = result.x
min_deflection = result.fun
print("\n=== Local Optimization Complete ===")
print(f"Optimal Thickness (h): {h_best:.6f} m")
print(f"Optimal Width (b): {b_best:.6f} m")
print(f"Minimum Weight: {rho*Len*h_best*b_best} kg")
print(f"Deflection: {min_deflection:.6f} m")
min_weight = rho*Len*h_best*b_best


print("\n\n--- Running DE with Random Initialization ---")

def generate_initial_population(bounds, size=30):
    pop = []
    for _ in range(size):
        # Randomly sample h within bounds
        h = np.random.uniform(bounds[0][0], bounds[0][1])
        # Compute b to satisfy the fixed area constraint
        b = fixed_area / h
        
        # Ensure b is within bounds
        if b_min <= b <= b_max:
            pop.append([h, b])
    return np.array(pop)

# Generate initial population
init_pop = generate_initial_population(bounds)

start_time_random = time.time()
# Run DE with custom initial population
result_random = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=40,
    tol=1e-4,
    init=init_pop,
    polish=True,
    updating='deferred',
    disp=True
)
time_random = time.time() - start_time_random
print(result_random)
h_best_rd, b_best_rd = result_random.x
min_deflection_rd = result_random.fun
print("\n=== Local Optimization Complete ===")
print(f"Optimal Thickness (h): {h_best_rd:.6f} m")
print(f"Optimal Width (b): {b_best_rd:.6f} m")
print(f"Minimum Weight: {rho*Len*h_best_rd*b_best_rd} kg")
print(f"Deflection: {min_deflection_rd:.6f} m")
min_weight_rd = rho*Len*h_best_rd*b_best_rd

# === Collect Results for AI-Initialized DE and Random DE ===
results = {
    "LLM_optimizer": {
        "no_of_iterations": len(iteration_history),
        "time_taken": time_llm,
        "optimal_thickness_h": h_opt,
        "optimal_width_b": b_opt,
        "minimum_deflection": u_max,
        "minimum_weight": rho * Len * h_opt * b_opt
    },
    "AI_Initialized_DE": {
        "no_of_iterations": result.nit,
        "time_taken": time_ai,
        "optimal_thickness_h": h_best,
        "optimal_width_b": b_best,
        "minimum_deflection": min_deflection,
        "minimum_weight": rho * Len * h_best * b_best
    },
    "Random_DE": {
        "no_of_iterations": result_random.nit,
        "time_taken": time_random,
        "optimal_thickness_h": h_best_rd,
        "optimal_width_b": b_best_rd,
        "minimum_deflection": min_deflection_rd,
        "minimum_weight": min_weight_rd
    }
}

# Define the output file path
output_file = "optimization_results_area.txt"

# Convert the results dictionary to a JSON-formatted string
json_string = json.dumps(results, indent=4)

# Open the file in append mode and write the JSON string
with open(output_file, "a") as f:
    f.write(json_string)
    #f.write("\n")

print(f"\nResults appended to '{output_file}'.")


# === Evaluate Perturbation ===
def evaluate_perturbation(h_perturbed, b_perturbed):
    try:
        # Create mesh
        mesh = RectangleMesh(
            Point(0, 0), 
            Point(Len, h_perturbed),
            num_elements_length(),
            num_elements_height(h_perturbed)
        )
        V = VectorFunctionSpace(mesh, 'Lagrange', degree=2)
        V_scalar = FunctionSpace(mesh, 'P', 1)
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        boundaries.set_all(0)
        
        # Boundary conditions
        class FixedEnd(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0.0) and on_boundary
        
        class FreeEnd(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], Len) and on_boundary
        
        fixed_end = FixedEnd()
        fixed_end.mark(boundaries, 1)  # Mark fixed boundary
        free_end = FreeEnd()
        free_end.mark(boundaries, 2)  # Mark free boundary
        ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
        bc = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 1)
        
        # Variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant((0.0, -rho * 9.81 * b_perturbed))  # Body force (N/m³)
        traction = Constant((0.0, -P / h_perturbed))  # Traction force at free end
        a = inner(sigma(u), epsilon(v)) * dx
        L = inner(f, v) * dx + inner(traction, v) * ds(2)  # Apply traction on free end
        
        # Assemble system
        A, b_matrix = assemble_system(a, L, bc)
        
        # Solve the system
        u_sol = Function(V)
        solve(A, u_sol.vector(), b_matrix)
        
        # Post-processing
        u_mag = project(sqrt(dot(u_sol, u_sol)), V_scalar)
        u_max = float(np.amax(u_mag.vector().get_local()))
        von_mises_expr = von_mises_stress(u_sol)
        von_mises_stress_field = project(von_mises_expr, V_scalar)
        max_stress = np.max(von_mises_stress_field.vector().get_local())
        yield_stress = 0.002 * E
        constraints_satisfied = (max_stress <= yield_stress)  # Check stress constraint
        
        # Weight calculation
        weight = rho * Len * h_perturbed * b_perturbed
        
        return {
            "weight": weight,
            "deflection": u_max,
            "von_mises": max_stress,
            "constraints_satisfied": constraints_satisfied
        }
    except Exception as e:
        print(f"Error during perturbation evaluation: {e}")
        return None
    
# === Main Perturbation Analysis ===
perturbation_percentages = [
    -0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    -0.15, -0.12, -0.09, -0.06, -0.03, 0.03, 0.06, 0.09, 0.12, 0.15,
    -0.20, -0.16, -0.12, -0.08, -0.04, 0.04, 0.08, 0.12, 0.16, 0.20,
    -0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25
]
perturbation_results = []

# Optimal values from previous optimization
h_opt, b_opt = h_best, b_best  # Replace with actual results

# Evaluate perturbations for thickness (h)
for percentage in perturbation_percentages:
    h_perturbed = h_opt * (1 + percentage)
    
    # Clip h to its bounds
    h_perturbed = np.clip(h_perturbed, h_min, h_max)
    
    # Adjust b to satisfy the fixed area constraint
    b_perturbed = fixed_area / h_perturbed
    
    # Clip b to its bounds
    b_perturbed = np.clip(b_perturbed, b_min, b_max)
    
    # Recheck h*b after clipping b
    if abs(h_perturbed * b_perturbed - fixed_area) > 1e-6:  # Allow small numerical tolerance
        print(f"Area constraint violation: h={h_perturbed:.4f}, b={b_perturbed:.4f}. Skipping...")
        continue
    
    print(f"\nPerturbing Thickness: h={h_perturbed:.4f}, b={b_perturbed:.4f}")
    
    result = evaluate_perturbation(h_perturbed, b_perturbed)
    if result is None:
        print("Failed to evaluate perturbation. Skipping...")
        continue
    
    weight_perturbed = result["weight"]
    deflection_perturbed = result["deflection"]
    von_mises_perturbed = result["von_mises"]
    constraints_satisfied = result["constraints_satisfied"]
    
    print(f"Weight: {weight_perturbed} kg, Deflection: {deflection_perturbed} m, "
          f"Max von Mises Stress: {von_mises_perturbed} Pa, Constraints Satisfied: {constraints_satisfied}")
    
    perturbation_results.append({
        "parameter": "h",
        "value": h_perturbed,
        "weight": weight_perturbed,
        "deflection": deflection_perturbed,
        "von_mises": von_mises_perturbed,
        "constraints_satisfied": constraints_satisfied
    })

# Evaluate perturbations for width (b)
for percentage in perturbation_percentages:
    b_perturbed = b_opt * (1 + percentage)
    
    # Clip b to its bounds
    b_perturbed = np.clip(b_perturbed, b_min, b_max)
    
    # Adjust h to satisfy the fixed area constraint
    h_perturbed = fixed_area / b_perturbed
    
    # Clip h to its bounds
    h_perturbed = np.clip(h_perturbed, h_min, h_max)
    
    # Recheck h*b after clipping h
    if abs(h_perturbed * b_perturbed - fixed_area) > 1e-6:  # Allow small numerical tolerance
        print(f"Area constraint violation: h={h_perturbed:.4f}, b={b_perturbed:.4f}. Skipping...")
        continue
    
    print(f"\nPerturbing Width: h={h_perturbed:.4f}, b={b_perturbed:.4f}")
    
    result = evaluate_perturbation(h_perturbed, b_perturbed)
    if result is None:
        print("Failed to evaluate perturbation. Skipping...")
        continue
    
    weight_perturbed = result["weight"]
    deflection_perturbed = result["deflection"]
    von_mises_perturbed = result["von_mises"]
    constraints_satisfied = result["constraints_satisfied"]
    
    print(f"Weight: {weight_perturbed} kg, Deflection: {deflection_perturbed} m, "
          f"Max von Mises Stress: {von_mises_perturbed} Pa, Constraints Satisfied: {constraints_satisfied}")
    
    perturbation_results.append({
        "parameter": "b",
        "value": b_perturbed,
        "weight": weight_perturbed,
        "deflection": deflection_perturbed,
        "von_mises": von_mises_perturbed,
        "constraints_satisfied": constraints_satisfied
    })

# Analyze perturbation results
valid_perturbations = [res for res in perturbation_results if res["constraints_satisfied"]]
if valid_perturbations:
    best_perturbation = min(valid_perturbations, key=lambda x: x["weight"])
    print("\n--- Best Perturbation ---")
    print(f"Parameter: {best_perturbation['parameter']}, Value: {best_perturbation['value']}")
    print(f"Weight: {best_perturbation['weight']} kg, Deflection: {best_perturbation['deflection']} m, "
          f"Max von Mises Stress: {best_perturbation['von_mises']} Pa")
else:
    print("\nNo valid perturbations found. Original solution is likely optimal.")

# Conclusion
if all(round(res["weight"], 2) >= round(min_weight, 2) for res in valid_perturbations):
    print("\nVerification Complete: All valid perturbations resulted in equal or higher weight. "
          "The solution is likely a local minimum.")
else:
    print("\nVerification Complete: Some valid perturbations resulted in lower weight. "
          "The solution may not be a local minimum.")

# Preprocess perturbation_results to make it JSON-serializable
serializable_results = [
    {
        "parameter": res["parameter"],
        "value": res["value"],
        "weight": res["weight"],
        "deflection": res["deflection"],
        "von_mises": res["von_mises"],
        "constraints_satisfied": int(res["constraints_satisfied"])  # Convert bool to int
    }
    for res in perturbation_results
]

# Save perturbation results to a JSON file
with open("perturbation_results_area.json", "w") as f:
    json.dump(serializable_results, f, indent=4)

print("\nPerturbation results saved to 'perturbation_results.json'.")