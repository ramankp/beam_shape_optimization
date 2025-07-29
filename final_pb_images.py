import os
import numpy as np
from fenics import *
import re
from openai import AzureOpenAI
import keyring
import matplotlib.pyplot as plt
import base64
import sys  # Add this at the top of your script

# === Redirecting Output to a File ===
output_file = "optimization_results_llm_latest.txt"  # Specify the output file name
original_stdout = sys.stdout  # Save the original stdout
with open(output_file, "w") as f:
    sys.stdout = f 
    # === Algorithmic Parameters ===
    niternp = 20  # Non-penalized iterations
    niter = 80    # Total iterations
    pmax = 4      # Maximum SIMP exponent
    exponent_update_frequency = 4  # Minimum steps between exponent updates
    tol_mass = 1e-4  # Tolerance on mass when finding Lagrange multiplier
    thetamin = 0.001  # Minimum density modeling void
    
    # === Problem Parameters ===
    thetamoy = 0.4  # Target average material density
    E = Constant(1)  # Young's modulus
    nu = Constant(0.3)  # Poisson's ratio
    lamda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / (2 * (1 + nu))
    f = Constant((0, -1))  # Vertical downward force

    # === Mesh and Boundaries ===
    mesh = RectangleMesh(Point(-2, 0), Point(2, 1), 50, 30, "crossed")

    def left(x, on_boundary):
        return near(x[0], -2) and on_boundary

    def load(x, on_boundary):
        return near(x[0], 2) and near(x[1], 0.5, 0.05)

    facets = MeshFunction("size_t", mesh, 1)
    AutoSubDomain(load).mark(facets, 1)
    ds = Measure("ds", subdomain_data=facets)

    # === Function Spaces ===
    V0 = FunctionSpace(mesh, "DG", 0)  # Discontinuous Galerkin space for density field
    V2 = VectorFunctionSpace(mesh, "CG", 2)  # Continuous Galerkin space for displacement field
    bc = DirichletBC(V2, Constant((0, 0)), left)  # Fixed boundary condition

    p = Constant(1)  # SIMP penalty exponent
    exponent_counter = 0  # Exponent update counter
    lagrange = Constant(1)  # Lagrange multiplier for volume constraint

    thetaold = Function(V0, name="Density")
    thetaold.interpolate(Constant(thetamoy))
    #coeff = thetaold**p
    def coeff():
        return thetaold ** p

    theta = Function(V0)

    volume = assemble(Constant(1) * dx(domain=mesh))
    avg_density_0 = assemble(thetaold * dx) / volume  # Initial average density
    avg_density = 0.0

    # Print initial parameters to output file or console
    print("\n--- Initial Parameters ---")
    print(f"{'Algorithm Type:':<25} LLM-driven")
    print(f"{'Total Iterations:':<25} {niter}")
    print(f"{'Non-Penalized Iterations:':<25} {niternp}")
    print(f"{'Max SIMP Exponent (pmax):':<25} {pmax}")
    print(f"{'Exponent Update Frequency:':<25} {exponent_update_frequency}")
    print(f"{'Target Avg Density (thetamoy):':<25} {thetamoy}")
    print(f"{'Min Density (thetamin):':<25} {thetamin}")
    print(f"{'Mesh Size (x, y):':<25} ({mesh.num_cells()})")
    print(f"{'Material Properties:':<25} E = {float(E)}, nu = {float(nu)}")
    print(f"{'Load Vector:':<25} f = {tuple(f.values())}")
    print("--------------------------\n")



    # === Elasticity Formulations ===
    def eps(v):
        return sym(grad(v))

    def sigma(v):
        #return coeff * (lamda * div(v) * Identity(2) + 2 * mu * eps(v))
        return coeff() * (lamda * div(v) * Identity(2) + 2 * mu * eps(v))

    def energy_density(u, v):
        return inner(sigma(u), eps(v))

    # === Variational Problem ===
    u_ = TestFunction(V2)
    du = TrialFunction(V2)
    a = inner(sigma(u_), eps(du)) * dx
    L = dot(f, u_) * ds(1)

    # === Local Projection ===
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

    # === Update Density Field ===
    def update_theta():
        global avg_density
        #theta.assign(local_project((p * coeff * energy_density(u, u) / lagrange)**(1 / (p + 1)), V0))
        theta.assign(local_project((p * coeff() * energy_density(u, u) / lagrange)**(1 / (p + 1)), V0))
        thetav = theta.vector().get_local()
        theta.vector().set_local(np.maximum(np.minimum(1, thetav), thetamin))
        theta.vector().apply("insert")
        avg_density = assemble(theta * dx) / volume
        return avg_density

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def save_density_image(iteration):
        plot(thetaold, cmap="bone_r")
        plt.title(f"Density Distribution - Iteration {iteration}")
        plt.savefig(f"density_iter_{iteration}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def get_visual_history_paths(current_iter):
        return [f"density_iter_{j}.png" for j in range(0, current_iter + 1, 5)]

    def create_combined_image(image_paths, output_path):
        n = len(image_paths)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, img_path in zip(axes, image_paths):
            img = plt.imread(img_path)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Iteration {img_path.split('_')[-1].replace('.png', '')}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # === Update Lagrange Multiplier ===
    def update_lagrange_multiplier(avg_density):
        global lagrange
        avg_density1 = avg_density
        if avg_density1 < avg_density_0:
            lagmin = float(lagrange)
            while avg_density < avg_density_0:
                lagrange.assign(Constant(lagrange / 2))
                avg_density = update_theta()
            lagmax = float(lagrange)
        elif avg_density1 > avg_density_0:
            lagmax = float(lagrange)
            while avg_density > avg_density_0:
                lagrange.assign(Constant(lagrange * 2))
                avg_density = update_theta()
            lagmin = float(lagrange)
        else:
            lagmin = float(lagrange)
            lagmax = float(lagrange)

        # Dichotomy on Lagrange multiplier
        max_dichotomy_iterations = 100
        inddico = 0
        while abs(1 - avg_density / avg_density_0) > tol_mass and inddico < max_dichotomy_iterations:
            lagrange.assign(Constant((lagmax + lagmin) / 2))
            avg_density = update_theta()
            inddico += 1
            if avg_density < avg_density_0:
                lagmin = float(lagrange)
            else:
                lagmax = float(lagrange)

        if inddico == max_dichotomy_iterations:
            print("   WARNING: Dichotomy search reached maximum iterations.")
        else:
            print(f"   Dichotomy iterations: {inddico}")

    # === Update SIMP Exponent ===
    def update_exponent(exponent_counter):
        global i
        print(i)
        global p
        exponent_counter += 1
        if i < niternp:
            p.assign(Constant(1))
        elif i >= niternp:
            if i == niternp:
                print("\n Starting penalized iterations\n")
            if (abs(compliance - old_compliance) < 0.01 * compliance_history[0]) and (exponent_counter > exponent_update_frequency):
                gray_level = assemble((theta - thetamin) * (1 - theta) * dx) / volume
                p.assign(Constant(min(float(p) * (1 + 0.3**(1 + gray_level / 2)), pmax)))
                exponent_counter = 0
                print("   Updated SIMP exponent p =", float(p))
        return exponent_counter

    def get_trend(values):
        if len(values) < 3:
            return "unknown"
        if values[-1] > values[-2] and values[-2] > values[-3]:
            return "increasing"
        elif values[-1] < values[-2] and values[-2] < values[-3]:
            return "decreasing"
        else:
            return "stable"

    # === Azure OpenAI Setup ===
    subscription_key = keyring.get_password("my_api_key_openai_1", "raman")
    endpoint = os.getenv("ENDPOINT_URL", "https://ai-ramanstudentgpt41588930900787.openai.azure.com/")
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=subscription_key, api_version="2024-05-01-preview")

    # === Query Azure OpenAI for Penalty Exponent (p) ===
    def query_azure_openai_p(prompt, current_p,itre):
        if itre % 5 == 0 and itre > 0:
            combined_img_path = f"history_up_to_{itre}.png"
            history_image_paths = [f"density_iter_{j}.png" for j in range(0, itre+1) if j % 5 == 0]
            print(history_image_paths)
            create_combined_image(history_image_paths, combined_img_path)
            encoded_image = encode_image(combined_img_path)
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert in topology optimization using the SIMP method."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                            }
                        ]
                    }
                ]
                response = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=5000,
                    temperature=0.0001,
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
                
                if len(matches) >= 1:
                    suggested_p = float(matches[0])
                    return min(max(suggested_p, 1), pmax)  # Clip within [1, pmax]
                else:
                    raise ValueError("Insufficient matches found.")
            except Exception as e:
                print(f"Error querying Azure OpenAI: {e}")
                print(f"Problematic Response: {generated_text}")
                print(f"Cleaned Response: {cleaned}")
                print(f"Extracted Matches: {matches}")
                # Fallback heuristic adjustment
                perturbation = 0.01  # Small perturbation to avoid stagnation
                new_p = float(current_p) * (1 + np.random.uniform(-perturbation, perturbation))
                return np.clip(new_p, 1, pmax)        

        else:
            try:
                messages = [
                    {"role": "system", "content": "You are an expert in topology optimization using the SIMP method."},
                    {"role": "user", "content": prompt}
                ]
                response = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=5000,
                    temperature=0.0001,
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
                
                if len(matches) >= 1:
                    suggested_p = float(matches[0])
                    return min(max(suggested_p, 1), pmax)  # Clip within [1, pmax]
                else:
                    raise ValueError("Insufficient matches found.")
            except Exception as e:
                print(f"Error querying Azure OpenAI: {e}")
                print(f"Problematic Response: {generated_text}")
                print(f"Cleaned Response: {cleaned}")
                print(f"Extracted Matches: {matches}")
                # Fallback heuristic adjustment
                perturbation = 0.01  # Small perturbation to avoid stagnation
                new_p = float(current_p) * (1 + np.random.uniform(-perturbation, perturbation))
                return np.clip(new_p, 1, pmax)

    # === Optimization Loop ===
    u = Function(V2, name="Displacement")
    old_compliance = 1e30
    ffile = XDMFFile("topology_optimization.xdmf")
    ffile.parameters["flush_output"] = True
    ffile.parameters["functions_share_mesh"] = True
    compliance_history = []
    gray_level_history = []
    density_error_history = []
    p_history = []

    print("\nStarting Optimization Process...\n")

    # === Stopping Criteria Parameters ===
    convergence_tolerance = 1e-4  # Tolerance for convergence
    min_improvement = 1e-5        # Minimum improvement in compliance to continue
    max_no_improvement = 5        # Maximum number of iterations without improvement
    no_improvement_counter = 0

    for i in range(niter):
        print(f"\n--- Iteration {i + 1} ---")

        # Solve elasticity problem
        solve(a == L, u, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
        ffile.write(thetaold, i)
        ffile.write(u, i)

        # Compute compliance
        compliance = assemble(action(L, u))
        compliance_history.append(compliance)
        print(f"Compliance: {compliance:.6f}")

        # Update density field
        avg_density = update_theta()
        gray_level = assemble((theta - thetamin)*(1 - theta)*dx)/volume
        gray_level_history.append(gray_level)
        density_error = abs(avg_density - thetamoy)
        density_error_history.append(abs(avg_density - thetamoy))
        print(f"Average Density: {avg_density:.6f}")
        print(f"Lagrange Multiplier: {float(lagrange):.6f}")
        compliance_trend = get_trend(compliance_history[-min(3, len(compliance_history)):])
        gray_level_trend = get_trend(gray_level_history[-min(3, len(gray_level_history)):])
        density_error_trend = get_trend(density_error_history[-min(3, len(density_error_history)):])


        # === Query Azure OpenAI for p ===
        if i > 0:  # Start querying after the first iteration
            if i == 1:
                prompt = f"""
                You are an expert in topology optimization using the SIMP method. Your task is to suggest the next value for the penalty exponent (`p`) based on the current state and historical trends.

                ### Current State:
                - Iteration: {i + 1}
                - Current p: {float(p):.2f}
                - Compliance: {compliance:.6f} (Trend: {compliance_trend})
                - Gray Level: {gray_level:.6f} (Trend: {gray_level_trend}) [Lower is better]
                - Average Density: {avg_density:.6f}, Target: {thetamoy} (Error: {density_error:.6f}, Trend: {density_error_trend})
                - Lagrange Multiplier: {float(lagrange):.6f}
                - Maximum p: {pmax}

                ### Rules:
                1. Increase p gradually if compliance is decreasing and gray level is low.
                2. If compliance stagnates or increases, keep or slightly reduce p.
                3. Avoid large jumps in p (e.g., from 3 to 4 directly unless very stable).
                4. Prefer binary material distributions (low gray level).

                Give me exactly one number between 1 and {pmax} to one decimal place as the next value for p. No explanations needed.
                """
            elif i % 5 == 0 and i > 1:
                    list_of_iterations = [j for j in range(0, i + 1) if j % 5 == 0]
                    prompt = f"""
                    You are an expert in topology optimization using the SIMP method.
                    Your task is to suggest the next value for the penalty exponent (`p`) based on the visual evolution of the density field shown in the attached image and the latest numerical metrics.

                    The image shows the density distribution at iterations: {list_of_iterations}.

                    ### Current State:
                    - Iteration: {i + 1}
                    - Current p: {float(p):.2f}
                    - Compliance: {compliance:.6f} (Trend: {compliance_trend})
                    - Gray Level: {gray_level:.6f} (Trend: {gray_level_trend}) [Lower is better]
                    - Average Density: {avg_density:.6f}, Target: {thetamoy} (Error: {density_error:.6f}, Trend: {density_error_trend})
                    - Lagrange Multiplier: {float(lagrange):.6f}
                    - Maximum p: {pmax}

                    ### Rules:
                    1. Increase p gradually if compliance is decreasing and gray level is low.
                    2. If compliance stagnates or increases, keep or slightly reduce p.
                    3. Avoid large jumps in p (e.g., from 3 to 4 directly unless very stable).
                    4. Prefer binary material distributions (low gray level).
                    5. Use the image to detect visual artifacts like checkerboarding or disconnected paths.

                    Give me exactly one number between 1 and {pmax} to one decimal place as the next value for p. No explanations needed.

                    """
            
            else:
                prompt = f"""
                You are an expert in topology optimization using the SIMP method. Your task is to suggest the next value for the penalty exponent (`p`) based on the current state and historical trends.

                ### Current State:
                - Iteration: {i + 1}
                - Current p: {float(p):.2f}
                - Compliance: {compliance:.6f} (Trend: {compliance_trend})
                - Gray Level: {gray_level:.6f} (Trend: {gray_level_trend}) [Lower is better]
                - Average Density: {avg_density:.6f}, Target: {thetamoy} (Error: {density_error:.6f}, Trend: {density_error_trend})
                - Lagrange Multiplier: {float(lagrange):.6f}
                - Maximum p: {pmax}
                -Last suggested {float(p):.2f} =  → Compliance improved by {compliance_change}" 

                ### Rules:
                1. Increase p gradually if compliance is decreasing and gray level is low.
                2. If compliance stagnates or increases, keep or slightly reduce p.
                3. Avoid large jumps in p (e.g., from 3 to 4 directly unless very stable).
                4. Prefer binary material distributions (low gray level).

                Give me exactly one number between 1 and {pmax} to one decimal place as the next value for p. No explanations needed.
                """
            #print(prompt)
            try:
                # Query Azure OpenAI for p
                suggested_p = query_azure_openai_p(prompt, float(p),i)

                # Update p with the suggested value
                p.assign(Constant(suggested_p))
                
                if i < niternp:
                    p.assign(Constant(1))  # Force p = 1 during non-penalized iterations

                print(f"Suggested by LLM: p = {suggested_p:.6f}")

            except Exception as e:
                print(f"Error querying Azure OpenAI: {e}")
                print("Falling back to traditional updates for p.")

                # Fallback to traditional updates if LLM fails
                exponent_counter = update_exponent(exponent_counter)

        else:
            # Traditional updates for the first iteration
            exponent_counter = update_exponent(exponent_counter)

        print(f"SIMP Exponent (p): {float(p):.6f}")
        p_history.append(float(p)) 
        #coeff = thetaold**p
        # Update Lagrange multiplier using traditional method
        update_lagrange_multiplier(avg_density)
        # Check for convergence
        if i > niternp:
            compliance_change = abs(compliance - old_compliance)
            density_change = abs(avg_density - thetamoy)

            print(f"Compliance Change: {compliance_change:.6f}, Density Change: {density_change:.6f}")

            # Check if compliance improvement is negligible
            if compliance_change < min_improvement:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

            # Stopping criteria
            if (compliance_change < convergence_tolerance and density_change < 1e-4) or no_improvement_counter >= max_no_improvement:
                print("\nConvergence criteria met. Stopping optimization early.")
                break
        
        if i % 5 == 0:
            save_density_image(i)

        # === Print Pixel Densities ===
        density_values = thetaold.vector().get_local()  # Extract density values as a 1D array
        density_grid = density_values.reshape(100, 60)  # Reshape into a 2D grid (100 rows, 60 columns) 
        #print(f"Density Grid (Iteration {i + 1}):")
        #for row in density_grid:
            #print(" ".join(f"{x:.6f}" for x in row))    # Print each row with 6 decimal precision


        # Update theta field and compliance
        compliance_change = abs(compliance - old_compliance)
        print(compliance_change)
        thetaold.assign(theta)
        old_compliance = compliance

    # Final visualization
    plot(theta, cmap="bone_r")
    plt.title("Final Density Distribution")
    plt.savefig("final_density.png", dpi=300, bbox_inches='tight')
    plt.close()
    plt.show()

    # Plot convergence history
    plt.figure()
    plt.plot(np.arange(1, len(compliance_history) + 1), compliance_history)
    ax = plt.gca()
    ymax = ax.get_ylim()[1]
    plt.plot([niternp, niternp], [0, ymax], "--k")
    plt.annotate(r"$\leftarrow$ Penalized iterations $\rightarrow$", xy=[niternp + 1, ymax * 0.02], fontsize=14)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Compliance")
    plt.title("Convergence History", fontsize=16)
    plt.show()

    # === Save Final State and History for Restart ===
    np.savez("optimization_state.npz",
        # Final fields
        theta_final=thetaold.vector().get_local(),
        p_final=float(p),
        lagrange_final=float(lagrange),
        # History data
        compliance_history=np.array(compliance_history),
        p_history=np.array(p_history),
        gray_level_history=np.array(gray_level_history),
        density_error_history=np.array(density_error_history),

        # Scalar parameters
        niternp=niternp,
        niter=niter,
        pmax=pmax,
        convergence_tolerance=convergence_tolerance,
        min_improvement=min_improvement,
        max_no_improvement=max_no_improvement,

        # Mesh info (optional)
        mesh_coordinates=mesh.coordinates()
    )
    print("Optimization state saved to 'optimization_state.npz'")

