import os
import numpy as np
from fenics import *
#import matplotlib.pyplot as plt
import re
from openai import AzureOpenAI
import keyring
import time
import json
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# === Material Properties (Aluminum) ===
E = 70e9        # Young's modulus (70 GPa)
rho = 2700      # Density (2700 kg/m³)
nu = 0.33       # Poisson's ratio

# === Updated Beam Parameters ===
Len = 1.5       # Length (m)
h_min, h_max = 0.001, 0.05  # Thickness bounds (1 mm to 5 cm)
b_min, b_max = 0.001, 0.05  # Width bounds (1 mm to 5 cm)
P = 1000        # Point load (N)

# === Updated Constraints ===
#deflection_limit = Len / 250  # Deflection limit remains the same
max_weight = 1.5              # Maximum allowable weight (kg)
max_iterations = 50
tolerance = 1e-4

# === 2D Mesh Parameters (Plane Strain) ===
num_elements_length = lambda: max(50, int(Len / 0.01))  # Dynamic resolution
num_elements_height = lambda h: max(5, int(h / 0.005))
'''
# === Azure OpenAI Setup ===
subscription_key = keyring.get_password("my_api_key_openai", "raman")
endpoint = os.getenv("ENDPOINT_URL", "https://ai-hmsmstudentsraman366400147319.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
client = AzureOpenAI(azure_endpoint=endpoint, api_key=subscription_key, api_version="2024-05-01-preview")
'''
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



# === Query Azure OpenAI for Both Parameters (h and b) Together ===
def query_azure_openai_both(prompt, current_values):
    try:
        h_current, b_current = current_values
        messages = [
            {"role": "system", "content": "You are an AI assistant optimizing both thickness (h) and width (b) of a cantilever beam."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=50,
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
                # Add a small tolerance for floating-point comparisons
                epsilon = 1e-6
                # Validate bounds
                if (h_min - epsilon <= suggested_h <= h_max + epsilon) and (b_min - epsilon <= suggested_b <= b_max + epsilon):
                    return np.clip(suggested_h, h_min, h_max), np.clip(suggested_b, b_min, b_max)
                else:
                    raise ValueError("Suggested values out of bounds")
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
        new_h = h_current * (1 + np.random.uniform(-perturbation, perturbation))
        new_b = b_current * (1 + np.random.uniform(-perturbation, perturbation))
        return (
            np.clip(new_h, h_min, h_max),
            np.clip(new_b, b_min, b_max)
        )

# === Objective Function ===
def objective(x):
    h, b = x
    #print(f"Evaluating h = {h:.6f}, b = {b:.6f}")
    # Out-of-bounds penalty
    if not (h_min <= h <= h_max and b_min <= b <= b_max):
        #print(" -> Out of bounds. Heavy penalty.")
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
    #f = Constant((0.0, 0.0))
    traction = Constant((0.0, -P / h))
    a = inner(sigma(u), epsilon(v)) * dx
    #L = inner(f, v) * dx
    L = inner(f, v) * dx + inner(traction, v) * ds(2)
    A, b_matrix = assemble_system(a, L, bc)
    #point = Point(Len, h / 2)
    #point_load = PointSource(V.sub(1), point, -P)#/ b)  # Negative y-direction
    #point_load.apply(b_matrix)
    u_sol = Function(V)
    solve(A, u_sol.vector(), b_matrix)
    u_mag = project(sqrt(dot(u_sol, u_sol)), V_scalar)
    u_max = float(np.amax(u_mag.vector().get_local()))
    # Constraint: Stress
    vm_stress_expr = von_mises_stress(u_sol)
    vm_proj = project(vm_stress_expr, V_scalar)
    max_vm = float(np.amax(vm_proj.vector().get_local()))
    if max_vm > yield_stress:
        #print(f" -> Stress {max_vm:.2f} exceeds yield {yield_stress}")
        return 1e6 + max_vm
    # Constraint: Weight
    weight = rho * Len * h * b
    #print(weight)
    if weight > max_weight:
        #print(f" -> Weight {weight:.2f} exceeds limit {max_weight}")
        return 1e6 + weight
    # Objective: Minimize deflection
    #print(f" -> OK | Deflection = {u_max:.6f} m")
    return u_max

def log_iteration_results(ai_iterations, de_iterations, filename="optimization_log.txt"):
    with open(filename, "a") as f:
        f.write(f"AI-driven iterations: {ai_iterations}, DE iterations: {de_iterations}\n")


iteration_data = {
    "h": [],
    "b": [],
    "deflection": [],
    "stress": [],
    "weight": []
}

start_time_llm = time.time()
# === Optimization Loop with Both Parameters Optimized Together ===
iteration_history = []
h_opt, b_opt = 0.02, 0.02  # Initial guesses
converged = False

for i in range(max_iterations):
    print(f"\n--- Iteration {i + 1} ---")
    # Step 1: Query Azure OpenAI for both h and b together
    print(f"Step 1: Solving for both thickness (h) and width (b)...")
    
    prompt_both = f"""
    I am solving for the optimal thickness (h) and width (b) of a cantilever beam under bending.
    The goal is to minimize the maximum deflection while satisfying the following constraints:
    - Maximum von Mises stress ≤ {0.002 * E:.2f} Pa.
    - Thickness (h) must be between {h_min} m and {h_max} m.
    - Width (b) must be between {b_min} m and {b_max} m.
    A point load of 1000 N is applied at the free end of the beam.
    Previous iterations:
    """
    for hist in iteration_history:
        prompt_both += f"At h={hist[0]:.4f}, b={hist[1]:.4f}: deflection={hist[2]:.6f} m, von Mises stress={hist[3]:.2f} Pa, weight={hist[4]:.2f} kg\n"
    prompt_both += (
        f"Suggest the next pair of thickness (h) and width (b) values that minimize deflection without violating the constraints. "
        f"The beam must have a thickness (h) greater than the width (b) to ensure proper structural stability."
        f"Ensure the constraints are satisfied: h ∈ [{h_min:.4f}, {h_max:.4f}], b ∈ [{b_min:.4f}, {b_max:.4f}], von Mises stress ≤ {0.002 * E:.2f}."# , weight ≤ {max_weight}. "
        f"By observing the trend you can take large steps if needed."
        f"Provide exactly two numbers (h, b), separated by a space, with no additional text or explanations."
    )
    '''
    prompt_both = f"""
    I am solving for the optimal thickness (h) and width (b) of a cantilever beam under bending.
    The goal is to minimize the maximum deflection while satisfying the following constraints:
    - Maximum von Mises stress ≤ {0.002 * E:.2f} Pa.
    - Weight of the beam ≤ {max_weight:.2f} kg.
    - Thickness (h) must be between {h_min:.4f} m and {h_max:.4f} m.
    - Width (b) must be between {b_min:.4f} m and {b_max:.4f} m.
    A point load of 1000 N is applied at the free end of the beam.
    Previous iterations:
    """
    for hist in iteration_history:
        prompt_both += f"At h={hist[0]:.4f}, b={hist[1]:.4f}: deflection={hist[2]:.6f} m, von Mises stress={hist[3]:.2f} Pa, weight={hist[4]:.2f} kg\n"
    prompt_both += (
        f"Suggest the next pair of thickness (h) and width (b) values that minimize deflection without violating the constraints. "
        f"The beam must have a thickness (h) greater than the width (b) to ensure proper structural stability."
        f"Ensure the constraints are satisfied: h ∈ [{h_min:.4f}, {h_max:.4f}], b ∈ [{b_min:.4f}, {b_max:.4f}], von Mises stress ≤ {0.002 * E:.2f}, weight ≤ {max_weight:.2f}."
        f"By observing the trend, take large steps if needed to explore feasible regions efficiently."
        f"If previous iterations suggest that increasing thickness or width reduces deflection but increases weight, prioritize reducing deflection unless the weight exceeds {max_weight:.2f}."
        f"Provide exactly two numbers (h, b), separated by a space, with no additional text or explanations."
    )'''
    print(prompt_both)
    h_prev, b_prev = h_opt, b_opt
    h_opt, b_opt = query_azure_openai_both(prompt_both, (h_prev, b_prev))

    # Step 2: Solve the finite element problem with updated h and b
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
    weight = rho * Len * h_opt * b_opt  # kg
    print(weight)
    iteration_history.append((h_opt, b_opt, u_max, max_stress, weight))

    iteration_data["h"].append(h_opt)
    iteration_data["b"].append(b_opt)
    iteration_data["deflection"].append(u_max)
    iteration_data["stress"].append(max_stress)
    iteration_data["weight"].append(weight)


    print(f"h={h_opt:.4f}, b={b_opt:.4f}, Deflection={u_max:.6f}, Stress={max_stress:.2f} Pa, Weight={weight:.2f} kg")

    # Check convergence
    if (abs(h_opt - h_prev) < tolerance and abs(b_opt - b_prev) < tolerance and weight <= max_weight and max_stress <= yield_stress):
        print(abs(h_opt - h_prev))
        print(abs(b_opt - b_prev))
        print("\nConverged to optimal solution!")
        converged = True
        break

if not converged:
    print("Max iterations reached without full convergence")


time_llm = time.time() - start_time_llm
print(f"AI-initialized DE: {len(iteration_history)} iterations, {time_llm:.2f}s")
# === Generate Convergence Plot ===
plt.figure(figsize=(12, 8))

# Plot h and b
plt.subplot(3, 1, 1)
plt.plot(range(1, len(iteration_data["h"]) + 1), iteration_data["h"], label="Thickness (h)", marker='o')
plt.plot(range(1, len(iteration_data["b"]) + 1), iteration_data["b"], label="Width (b)", marker='x')
plt.title("Evolution of Thickness (h) and Width (b)")
plt.xlabel("Iteration")
plt.ylabel("Value (m)")
plt.legend()
plt.grid()

# Plot deflection
plt.subplot(3, 1, 2)
plt.plot(range(1, len(iteration_data["deflection"]) + 1), iteration_data["deflection"], label="Deflection", color='green', marker='s')
plt.title("Evolution of Deflection")
plt.xlabel("Iteration")
plt.ylabel("Deflection (m)")
plt.legend()
plt.grid()

'''
# Plot stress and weight
plt.subplot(3, 1, 3)
plt.plot(range(1, len(iteration_data["stress"]) + 1), iteration_data["stress"], label="Stress", color='red', marker='^')
plt.plot(range(1, len(iteration_data["weight"]) + 1), iteration_data["weight"], label="Weight", color='blue', marker='d')
plt.title("Evolution of Stress and Weight")
plt.xlabel("Iteration")
plt.ylabel("Value (Pa for Stress, kg for Weight)")
plt.legend()
plt.grid()
'''
plt.tight_layout()
plt.savefig("convergence_plot.png")  # Save the plot
#plt.show()


# Final results
print("\n--- Final Results ---")
print(f"Optimal Thickness: {h_opt:.4f} m")
print(f"Optimal Width: {b_opt:.4f} m")
print(f"Weight: {rho*Len*h_opt*b_opt:.2f} kg")
print(f"Max Deflection: {u_max:.6f} m")


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
print(f"Minimum Weight: {rho*Len*h_best*b_best:.2f} kg")
print(f"Deflection: {min_deflection:.6f} m")
min_weight = rho*Len*h_best*b_best



# === Compare with Random Initialization DE ===
print("\n\n--- Running DE with Random Initialization ---")
start_time_random = time.time()
result_random = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=40,
    tol=1e-4,
    init='latinhypercube',  # Pure random initialization
    polish=True,
    updating='deferred',
    disp=True
)
time_random = time.time() - start_time_random

h_best_random, b_best_random = result_random.x
min_deflection_random = result_random.fun
print("\n=== Local Optimization Complete ===")
print(f"Optimal Thickness (h): {h_best_random:.6f} m")
print(f"Optimal Width (b): {b_best_random:.6f} m")
print(f"Minimum Weight: {rho*Len*h_best_random*b_best_random:.2f} kg")
print(f"Deflection: {min_deflection_random:.6f} m")
min_weight_random = rho*Len*h_best_random*b_best_random
# === Comparison Metrics ===
print("\n=== Performance Comparison ===")
print(f"AI-initialized DE: {result.nit} iterations, {result.nfev} evals, {time_ai:.2f}s")
print(f"Random DE: {result_random.nit} iterations, {result_random.nfev} evals, {time_random:.2f}s")
print(f"Final Deflection (AI): {result.fun:.6f} vs Random: {result_random.fun:.6f}")

log_iteration_results(result.nit, result_random.nit)

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
        "optimal_thickness_h": h_best_random,
        "optimal_width_b": b_best_random,
        "minimum_deflection": min_deflection_random,
        "minimum_weight": min_weight_random
    }
}

# Define the output file path
output_file = "optimization_results_weight1.txt"

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
        FixedEnd().mark(boundaries, 1)
        FreeEnd().mark(boundaries, 2)
        ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
        bc = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant((0.0, -rho * 9.81 * b_perturbed))  # Gravity
        traction = Constant((0.0, -P / h_perturbed))
        a = inner(sigma(u), epsilon(v)) * dx
        L = inner(f, v) * dx + inner(traction, v) * ds(2)
        A, b_matrix = assemble_system(a, L, bc)
        u_sol = Function(V)
        solve(A, u_sol.vector(), b_matrix)

        u_mag = project(sqrt(dot(u_sol, u_sol)), V_scalar)
        u_max = float(np.amax(u_mag.vector().get_local()))
        von_mises_expr = von_mises_stress(u_sol)
        von_mises_stress_field = project(von_mises_expr, V_scalar)
        max_stress = np.max(von_mises_stress_field.vector().get_local())

        weight = rho * Len * h_perturbed * b_perturbed

        constraints_satisfied = (weight <= max_weight) and (max_stress <= 0.002 * E)

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

# Optimal values from DE optimization
h_opt_perturb = h_best
b_opt_perturb = b_best

# Evaluate perturbations for thickness (h)
for percentage in perturbation_percentages:
    h_perturbed = h_opt_perturb * (1 + percentage)
    
    # Clip h to its bounds
    h_perturbed = np.clip(h_perturbed, h_min, h_max)
    
    # Adjust b to satisfy the fixed area constraint
    b_perturbed = b_best
    
    # Clip b to its bounds
    b_perturbed = np.clip(b_perturbed, b_min, b_max)
    
    
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
        "constraints_satisfied": bool(constraints_satisfied)
    })

# Evaluate perturbations for width (b)
for percentage in perturbation_percentages:
    b_perturbed = b_opt_perturb * (1 + percentage)
    
    # Clip b to its bounds
    b_perturbed = np.clip(b_perturbed, b_min, b_max)
    
    # Adjust h to satisfy the fixed area constraint
    h_perturbed = h_best
    
    # Clip h to its bounds
    h_perturbed = np.clip(h_perturbed, h_min, h_max)
    
    
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
        "constraints_satisfied": bool(constraints_satisfied)
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

# Conclusion: Check if there is a single perturbation with both lower deflection and lower or equal weight
better_solution_exists = any(
    (res["deflection"] < min_deflection) and (res["weight"] <= min_weight)
    for res in valid_perturbations
)

if better_solution_exists:
    print("\nVerification Complete: Some valid perturbations resulted in lower deflection AND lower or equal weight.")
    print("The original solution may NOT be a true local minimum.")
else:
    print("\nVerification Complete: No perturbation achieved both lower deflection and lower or equal weight.")
    print("The solution is likely a local minimum.")


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

# === Save Perturbation Results to JSON ===
with open("perturbation_results_weight_limit.json", "w") as f:
    json.dump(perturbation_results, f, indent=4)

print("\nPerturbation results saved to 'perturbation_results_weight_limit.json'.")










# === Evaluate Perturbation ===
def evaluate_perturbation(h_perturbed, b_perturbed):
    try:
        in_bounds = (h_min <= h_perturbed <= h_max) and (b_min <= b_perturbed <= b_max)
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
        #f = Constant((0.0, 0.0))
        a = inner(sigma(u), epsilon(v)) * dx
        L = inner(f, v) * dx
        
        # Assemble system without point load
        A, b_matrix = assemble_system(a, L, bc)
        point = Point(Len, h_perturbed / 2)
        point_load = PointSource(V.sub(1), point, -P)#/ b_perturbed)  # Negative y-direction
        point_load.apply(b_matrix)
        
        # Solve the system
        u_sol = Function(V)
        solve(A, u_sol.vector(), b_matrix)
        
        # Post-processing
        u_mag = project(sqrt(dot(u_sol, u_sol)), V_scalar)
        u_max = float(np.amax(u_mag.vector().get_local()))
        von_mises_expr = von_mises_stress(u_sol)
        von_mises_stress_field = project(von_mises_expr, V_scalar)
        max_stress = np.max(von_mises_stress_field.vector().get_local())
        weight = rho * Len * h_perturbed * b_perturbed
        constraints_satisfied = (weight <= max_weight and max_stress <= yield_stress and in_bounds) 
        print(yield_stress)
        print(max_stress)
        
        
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
h_opt, b_opt = h_best, b_best 

# Evaluate perturbations for thickness (h)
for percentage in perturbation_percentages:
    h_perturbed = h_opt * (1 + percentage)
    #h_perturbed = np.clip(h_perturbed, h_min, h_max)
    b_perturbed = b_opt  # Keep b constant for this perturbation
    
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
    #b_perturbed = np.clip(b_perturbed, b_min, b_max)
    h_perturbed = h_opt  # Keep h constant for this perturbation
    
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
    best_perturbation = min(valid_perturbations, key=lambda x: x["deflection"])  # Minimize deflection
    print("\n--- Best Perturbation ---")
    print(f"Parameter: {best_perturbation['parameter']}, Value: {best_perturbation['value']}")
    print(f"Deflection: {best_perturbation['deflection']} m, Weight: {best_perturbation['weight']} kg, "
          f"Max von Mises Stress: {best_perturbation['von_mises']} Pa")
else:
    print("\nNo valid perturbations found. Original solution is likely optimal.")

# Conclusion
if all(round(res["deflection"], 6) >= round(min_deflection, 6) for res in valid_perturbations):
    print(min_deflection)
    print("\nVerification Complete: All valid perturbations resulted in equal or higher deflection. "
          "The solution is likely a local minimum.")
else:
    print(min_deflection)
    for res in valid_perturbations:
        print(abs(res["deflection"] - min_deflection))
    print("\nVerification Complete: Some valid perturbations resulted in lower deflection. "
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
with open("perturbation_results.json", "w") as f:
    json.dump(serializable_results, f, indent=4)

print("\nPerturbation results saved to 'perturbation_results.json'.")

