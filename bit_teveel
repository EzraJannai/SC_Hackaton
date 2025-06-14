# Julia script to solve the 1D shallow water equations as a DAE problem
using DifferentialEquations, LinearAlgebra, Parameters, Plots

# This code solves the 1D shallow water equations (SWE) as a DAE problem.
# The setup is as follows:
# 1. Define parameters for the simulation, including gravity, number of grid points,
#   spatial domain, and bottom topography.
# 2. Set up initial conditions for water height and momentum.
# 3. Define the DAE residual function that describes the SWE.
# 4. Implement a time loop to solve the DAE problem using Sundials' IDA solver.
# 5. Plot the results.
# 6. Function calls to start the simulation.

# --- 1. Parameter setup ---
function make_parameters()
    g = 9.81
    D = 10.0
    N = 200  # number of grid points
    x_start, x_end = 0.0, 5.0
    x = collect(range(x_start, x_end, length=N))
    dx = x[2] - x[1]
    tstart, tstop = 0.0, 1.0
    cf = 0.5  # friction coefficient
    
    # Bed profile (wavy)
    zb = -D .+ 0.4 .* sin.(2 * π * x / x_end * ((N - 1)/N) * 5)
    
    return (; g, D, N, x_start, x_end, x, dx, tstart, tstop, cf, zb)
end

# --- 2. Initial condition ---
function initial_conditions(params)
    @unpack N, x, x_end, zb = params
    
    # Initial water height
    h0 = 1.0 .* exp.(-100 .* ((x / x_end .- 0.5) * x_end).^2) .- zb
    
    # Initial momentum (zero)
    q0 = zeros(N)
    
    return h0, q0
end

# --- 3. GABC boundary treatment ---
function apply_gabc!(dhdt, dqdt, h, q, params)
    @unpack g, N = params
    
    # Left boundary (x=0)
    c = sqrt(g * max(h[1], 1e-6))
    dqdt_in = (1 / h[1]) * (dqdt[1] - (q[1]/h[1]) * dhdt[1] + c * dhdt[1])
    dqdt_out = (1 / h[1]) * (dqdt[1] - (q[1]/h[1]) * dhdt[1] - c * dhdt[1])
    dqdt[1] = dqdt_in + dqdt_out
    
    # Right boundary (x=L)
    c = sqrt(g * max(h[N], 1e-6))
    dqdt_in = (1 / h[N]) * (dqdt[N] - (q[N]/h[N]) * dhdt[N] - c * dhdt[N])
    dqdt_out = (1 / h[N]) * (dqdt[N] - (q[N]/h[N]) * dhdt[N] + c * dhdt[N])
    dqdt[N] = dqdt_in + dqdt_out
    
    return nothing
end

# --- 4. DAE residual function ---
function swe_dae_residual!(residual, du, u, p, t)
    @unpack g, N, dx, cf, zb = p
    
    # Extract state variables
    h = u[1:N]
    q = u[N+1:2N]
    
    # Extract time derivatives
    dhdt = du[1:N]
    dqdt = du[N+1:2N]
    
    # Compute free surface elevation
    ζ = h .+ zb
    
    # Compute spatial derivatives using centered differences with periodic wrapping
    # (Note: boundaries will be handled by GABC)
    h_p = circshift(h, -1)  # h[i+1]
    h_m = circshift(h, 1)   # h[i-1]
    q_p = circshift(q, -1)  # q[i+1]
    q_m = circshift(q, 1)   # q[i-1]
    ζ_p = circshift(ζ, -1)  # ζ[i+1]
    ζ_m = circshift(ζ, 1)   # ζ[i-1]
    
    # Continuity equation residual: ∂h/∂t + ∂q/∂x = 0
    dqdx = (q_p .- q_m) ./ (2 * dx)
    residual[1:N] = dhdt .+ dqdx
    
    # Momentum equation residual: ∂q/∂t + ∂(q²/h)/∂x + gh∂ζ/∂x + friction = 0
    dqq_h_dx = ((q_p.^2 ./ (h_p .+ 1e-6)) .- (q_m.^2 ./ (h_m .+ 1e-6))) ./ (2 * dx)
    dzetadx = (ζ_p .- ζ_m) ./ (2 * dx)
    friction = cf .* q .* abs.(q) ./ (h.^2 .+ 1e-6)
    
    residual[N+1:2N] = dqdt .+ dqq_h_dx .+ g .* h .* dzetadx .+ friction
    
    # Apply GABC boundary conditions
    apply_gabc!(dhdt, dqdt, h, q, p)
    
    # Update residual with boundary-corrected derivatives
    residual[N+1:2N] = dqdt .+ dqq_h_dx .+ g .* h .* dzetadx .+ friction
    
    return nothing
end

# --- 5. Time integration ---
function timeloop(params)
    @unpack N, tstart, tstop = params
    
    # Set up initial conditions
    h0, q0 = initial_conditions(params)
    u0 = vcat(h0, q0)
    du0 = zeros(2N)  # Initial guess for du/dt
    
    tspan = (tstart, tstop)
    
    # All variables are differential (not algebraic)
    differential_vars = trues(2N)
    
    # Create DAE problem
    dae_prob = DAEProblem(
        swe_dae_residual!, du0, u0, tspan, params;
        differential_vars=differential_vars
    )
    
    # Solve with IDA solver
    sol = solve(dae_prob, IDA(), reltol=1e-8, abstol=1e-8, 
                saveat=range(tstart, tstop, length=500))
    
    return sol
end

# --- 6. Plotting results ---
function plot_results(sol, params)
    @unpack N, x, zb = params
    
    # Extract final state
    h_final = sol.u[end][1:N]
    q_final = sol.u[end][N+1:2N]
    surface_final = h_final .+ zb
    
    # Plot initial and final states
    h_initial = sol.u[1][1:N]
    surface_initial = h_initial .+ zb
    
    p1 = plot(x, zb, label="Bed elevation", color=:black, linewidth=2)
    plot!(p1, x, surface_initial, label="Initial surface", color=:blue, linewidth=2, linestyle=:dash)
    plot!(p1, x, surface_final, label="Final surface", color=:red, linewidth=2)
    xlabel!(p1, "x [m]")
    ylabel!(p1, "Elevation [m]")
    title!(p1, "1D Shallow Water Simulation with GABC")
    
    # Plot momentum
    p2 = plot(x, q_final, label="Final momentum", color=:green, linewidth=2)
    xlabel!(p2, "x [m]")
    ylabel!(p2, "Momentum q [m²/s]")
    title!(p2, "Final Momentum Distribution")
    
    # Combined plot
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

# --- 7. Animation function ---
function create_animation(sol, params)
    @unpack N, x, zb = params
    
    # Create animation
    anim = @animate for i in 1:length(sol.t)
        h = sol.u[i][1:N]
        surface = h .+ zb
        
        plot(x, zb, label="Bed elevation", color=:black, linewidth=2, 
             ylims=(minimum(zb)-0.1, maximum([maximum(h .+ zb) for h in [sol.u[j][1:N] for j in 1:length(sol.u)]]) + 0.2))
        plot!(x, surface, label="Free surface", color=:blue, linewidth=2, fill=(0, :lightblue, 0.5))
        xlabel!("x [m]")
        ylabel!("Elevation [m]")
        title!("1D Shallow Water Simulation - t = $(round(sol.t[i], digits=3)) s")
    end
    
    return anim
end

# --- 8. Main script ---
function main()
    println("Starting 1D Shallow Water Equations simulation...")
    
    # Set up parameters
    params = make_parameters()
    println("Parameters set up successfully")
    
    # Run simulation
    println("Running time integration...")
    solution = timeloop(params)
    println("Simulation completed successfully!")
    
    # Plot results
    println("Creating plots...")
    plot_results(solution, params)

    return solution, params
end

# Run the simulation
solution, params = main()