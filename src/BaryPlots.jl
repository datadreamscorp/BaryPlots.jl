module BaryPlots

    using DifferentialEquations
	using Plots
	using LinearAlgebra
	using NLsolve
	using ForwardDiff
    using PyPlot

    export plot_evolution, ternary_coords, replicator_dynamics!, check_stability, find_equilibria, plot_simplex, ReplicatorParams

"""
    ternary_coords(x::Vector{Float64}) -> Tuple{Float64, Float64}

Transforms a vector `x` of three strategy frequencies `[x1, x2, x3]` into ternary coordinates
for plotting within a simplex. It ensures that the sum of the frequencies equals 1 and returns
the transformed coordinates `(X, Y)`.

# Arguments
- `x`: A vector of length 3, representing the frequencies of three strategies.

# Returns
- A tuple `(X, Y)` representing the ternary coordinates of the input vector.
"""
    function ternary_coords(x)
        x1, x2, x3 = x  # Strategy frequencies
        s = x1 + x2 + x3 # Make it for dummies by forcing it to sum to 1
        x1 /= s
        x2 /= s
        x3 /= s
        # Swap x1 and x2 in the coordinate mapping
        X = 0.5 * (2 * x1 + x3)  # Use x1 instead of x2
        Y = (sqrt(3)/2) * x3     # Y-coordinate remains the same
        return (X, Y)
    end

"""
    plot_simplex(; labels::Vector{String} = ["Strategy 1", "Strategy 2", "Strategy 3"], kwargs...) -> Plots.Plot

Plots the simplex for three strategies, using a triangular shape, and adds labels for the
corners of the simplex. This is used as a background for plotting evolutionary dynamics.

# Arguments
- `labels`: A vector of three strings, which label the corners of the simplex. Default: `["Strategy 1", "Strategy 2", "Strategy 3"]`.

# Returns
- A `Plot` object displaying the simplex triangle with the specified labels.
"""
    function plot_simplex(; labels::Vector{String} = ["Strategy 1", "Strategy 2", "Strategy 3"], kwargs...)
        triangle_x = [0.0, 1.0, 0.5, 0.0]
        triangle_y = [0.0, 0.0, sqrt(3)/2, 0.0]
        plot(triangle_x, triangle_y;
            seriestype = :shape,
            aspect_ratio = 1,
            fillcolor = :white,
            linecolor = :black,
            legend = false,
            xlabel = "",
            ylabel = "",
            xlims = (-0.1, 1.1),
            ylims = (-0.1, sqrt(3)/2 + 0.1),
            framestyle = :none,
            dpi = 300,
            kwargs...)
        # Corrected labels
        annotate!(0.0, -0.05, text(labels[2], :center))  # Left corner (Loner)
        annotate!(1.0, -0.05, text(labels[1], :center))  # Right corner (Sharer)
        annotate!(0.5, sqrt(3)/2 + 0.05, text(labels[3], :center)) # Top corner (Hoarder)
    end

    struct ReplicatorParams
        
        payoff_functions::Tuple{Function, Function, Function}  # Payoff functions for strategies
        extra_params::NamedTuple # Additional parameters for payoff functions
    end

"""
    replicator_dynamics!(dx::Vector{Float64}, x::Vector{Float64}, params::ReplicatorParams, t::Float64)

Computes the time derivative of the replicator dynamics for a population playing three
strategies. The payoffs for each strategy are passed as part of `ReplicatorParams`.

# Arguments
- `dx`: A vector to store the time derivative (change in frequencies).
- `x`: A vector representing the current frequencies of the three strategies.
- `params`: A `ReplicatorParams` struct containing the payoff functions and extra parameters.
- `t`: The current time in the dynamics.

# Notes
This function is used to simulate the evolution of strategies over time.
"""
    function replicator_dynamics!(dx, x, params::ReplicatorParams, t)
        # Normalize the frequencies to ensure they sum to 1
        x_total = sum(x)
        x_normalized = x / x_total
    
        # Extract the payoff functions
        payoff1, payoff2, payoff3 = params.payoff_functions
    
        # Compute the payoffs
        w1 = payoff1(x_normalized, t, params.extra_params)
        w2 = payoff2(x_normalized, t, params.extra_params)
        w3 = payoff3(x_normalized, t, params.extra_params)
    
        payoffs = [w1, w2, w3]
    
        # Compute the average payoff
        avg_payoff = dot(x_normalized, payoffs)
    
        # Replicator dynamics equation
        dx .= x_normalized .* (payoffs .- avg_payoff)
    end

    function create_steady_state_callback(steady_state_tol)
        condition(u, t, integrator) = begin
            # Re-evaluate the derivative at the current state
            du = similar(u)
            integrator.f(du, u, integrator.p, t)  # Computes the time derivative at u
            du_norm = norm(du)
            return du_norm < steady_state_tol
        end
        affect!(integrator) = terminate!(integrator)
        return DiscreteCallback(condition, affect!)
    end

    function generate_simplex_grid(resolution::Int)
        points = []
        for i in 0:resolution
            for j in 0:(resolution - i)
                k = resolution - i - j
                x1 = i / resolution
                x2 = j / resolution
                x3 = k / resolution
                push!(points, [x1, x2, x3])
            end
        end
        return points
    end
    
    function compute_average_payoffs(grid_points::Vector{Vector{Float64}}, payoff_functions::Tuple{Function, Function, Function}, params::NamedTuple)
        avg_payoffs = []
        for x in grid_points
            w1 = payoff_functions[1](x, 0.0, params)
            w2 = payoff_functions[2](x, 0.0, params)
            w3 = payoff_functions[3](x, 0.0, params)
            avg_payoff = x[1]*w1 + x[2]*w2 + x[3]*w3
            push!(avg_payoffs, avg_payoff)
        end
        return avg_payoffs
    end
    
    function get_ternary_coordinates(grid_points::Vector{Vector{Float64}})
        X = []
        Y = []
        for x in grid_points
            X_i, Y_i = ternary_coords(x)
            push!(X, X_i)
            push!(Y, Y_i)
        end
        return X, Y
    end
    
    

"""
    plot_evolution(
        payoff_functions::Tuple{Function, Function, Function},
        x0_list::Vector{<:AbstractVector{<:Real}},
        tspan::Tuple{Float64, Float64};
        labels::Vector{String} = ["Strategy 1", "Strategy 2", "Strategy 3"],
        extra_params::NamedTuple = NamedTuple(),
        steady_state_tol::Float64 = 1e-6,
        arrow_list::Vector{Vector{Int}} = Vector{Vector{Int}}(),
        trajectory_labels::Vector{String} = String[],
        trajectory_colors::AbstractVector = Any[],
        num_initial_guesses::Int = 1000,
        equilibrium_tol::Float64 = 1e-6,
        eq_size = 6,
        kwargs...
    ) -> Plots.Plot

Simulates the evolution of strategy frequencies over time and plots their trajectories within
a ternary simplex. It also computes and plots the equilibria of the system.

# Arguments
- `payoff_functions`: A tuple of three payoff functions corresponding to each strategy.
- `x0_list`: A vector of initial conditions (strategy frequencies) to simulate trajectories.
- `tspan`: A tuple representing the time span for the simulation.
- `labels`: Labels for the corners of the simplex. Default: `["Strategy 1", "Strategy 2", "Strategy 3"]`.
- `extra_params`: Extra parameters for the payoff functions (optional).
- `steady_state_tol`: Tolerance for determining when a trajectory reaches steady state.
- `arrow_list`: A list of indices indicating where to draw arrows along the trajectories.
- `trajectory_labels`: Labels for each trajectory (optional).
- `trajectory_colors`: Colors for each trajectory (optional).
- `num_initial_guesses`: Number of initial guesses for finding equilibria.
- `equilibrium_tol`: Tolerance for determining equilibria.

# Returns
- A `Plot` object displaying the simplex with the trajectories and equilibria.
"""
    function plot_evolution(
        payoff_functions::Tuple{Function, Function, Function},
        x0_list::Vector{<:AbstractVector{<:Real}},
        tspan::Tuple{Float64, Float64};
        labels::Vector{String} = ["Strategy 1", "Strategy 2", "Strategy 3"],
        extra_params::NamedTuple = NamedTuple(),
        num_initial_guesses::Int = 1000,
        steady_state_tol::Float64 = 1e-6,
        solver_tol::Float64 = 1e-8,
        equilibrium_tol::Float64 = 1e-5,
        validity_tol::Float64 = 1e-6,
        stability_tol::Float64 = 1e-6,
        arrow_list::Vector{Vector{Int}} = Vector{Vector{Int}}(),
        trajectory_labels::Vector{String} = String[],
        trajectory_colors::AbstractVector = Any[],
        equilibrium_size::Int = 6,
        colored_trajectories::Bool = false,
        contourf::Bool = false,
        contour_resolution::Int = 50,
        contour_levels::Int = 10,
        kwargs...
    )
        num_trajectories = length(x0_list)
    
        # Default labels and colors if not provided
        if isempty(trajectory_labels)
            trajectory_labels = ["Trajectory $(i)" for i in 1:num_trajectories]
        end

        if isempty(trajectory_colors)
            trajectory_colors =  colored_trajectories ? [
                palette(:tab10)[(i - 1) % 10 + 1] 
                for i in 1:num_trajectories
                ] : repeat(["black"], num_trajectories)
        end
    
        # Prepare the parameters
        params = ReplicatorParams(payoff_functions, extra_params)
    
        # Create the steady state callback
        cb = create_steady_state_callback(steady_state_tol)
    
        # Plotting
        plot_simplex(labels = labels)
    
        # Compute equilibria
        equilibria, stability_status = find_equilibria(
            payoff_functions, 
            extra_params,
            num_initial_guesses = num_initial_guesses,
            solver_tol = solver_tol,
            equilibrium_tol = equilibrium_tol,
            validity_tol = validity_tol,
            stability_tol = stability_tol
        )
    
        for i in 1:num_trajectories
            x0 = x0_list[i] / sum(x0_list[i])  # Ensure x0 sums to 1
            prob = ODEProblem(replicator_dynamics!, x0, tspan, params)
            sol = solve(prob; callback = cb,
                        saveat = tspan[1]:0.01:tspan[2], tstops = [tspan[2]],
                        reltol = 1e-8, abstol = 1e-8, kwargs...,
                        isoutofdomain = (u,p,t) -> any(x -> x < -1e-8 || x > 1+1e-8, u))
    
            # Extract the solutions
            X = Float64[]
            Y = Float64[]
            for x in sol.u
                coord = ternary_coords(x)
                push!(X, coord[1])
                push!(Y, coord[2])
            end
    
            # Plot trajectory
            plot!(X, Y;
                linecolor = trajectory_colors[i],
                label = trajectory_labels[i])
    
            # Add arrows along the trajectory if requested
            if length(arrow_list) ≥ i && !isempty(arrow_list[i])
                arrow_scale = 0.01  # Adjust this value to change arrow size
                for j in arrow_list[i] #1:arrow_interval:(length(X)-1)
                    if j > 0
                        x_start, y_start = X[j], Y[j]
                        x_end, y_end = X[j+1], Y[j+1]
                        dx_arrow = x_end - x_start
                        dy_arrow = y_end - y_start
                        # Normalize the direction vector for consistent arrow size
                        norm_factor = sqrt(dx_arrow^2 + dy_arrow^2)
                        if norm_factor != 0  # Avoid division by zero
                            dx_arrow /= norm_factor
                            dy_arrow /= norm_factor
                            # Scale the arrow size
                            dx_arrow *= arrow_scale
                            dy_arrow *= arrow_scale
                            quiver!(
                                [x_start], 
                                [y_start], 
                                quiver=(
                                    [dx_arrow], [dy_arrow]);
                                    arrow=:closed, color=trajectory_colors[i],
                                    linewidth=1, label=false
                            )
                        end
                    end
                end
            end
    
        end
    
        # Plot the equilibria
        if !isempty(equilibria)
            for (i, x_eq) in enumerate(equilibria)
                is_stable = stability_status[i]
                # Ensure the equilibrium point sums to 1
                x_eq = x_eq / sum(x_eq)
                X_eq, Y_eq = ternary_coords(x_eq)
    
                if is_stable
                    # Plot stable equilibrium as filled black circle
                    scatter!([X_eq], [Y_eq];
                        markercolor = :black,
                        markershape = :circle,
                        markersize = equilibrium_size,
                        label = false)
                else
                    # Plot unstable equilibrium as hollow circle
                    scatter!([X_eq], [Y_eq];
                        markercolor = :white,
                        markerstrokecolor = :black,
                        markershape = :circle,
                        markersize = equilibrium_size,
                        label = false)
                end
            end
        end
    
        # Show legend
        plot!(legend = false)
    end

    function generate_initial_guesses(n)
        guesses = []
        for _ in 1:n
            x = rand(3)
            x /= sum(x)  # Normalize to sum to 1
            push!(guesses, x)
        end
        return guesses
    end

    function replicator_equation(x, params::ReplicatorParams, t)
        dx = similar(x)
        replicator_dynamics!(dx, x, params, t)
        return dx
    end

    function equilibrium_exists(x_new, equilibria, equilibrium_tol)
        x_new /= sum(x_new)
        for x_eq in equilibria
            x_eq /= sum(x_eq)
            if norm(x_new - x_eq) < equilibrium_tol
                return true
            end
        end
        return false
    end

"""
    check_stability(x_eq::Vector{Float64}, params::ReplicatorParams) -> Bool

Checks the stability of an equilibrium point by computing the Jacobian matrix of the reduced
system and checking the signs of its eigenvalues. If all eigenvalues have negative real parts,
the equilibrium is stable.

# Arguments
- `x_eq`: A vector representing the strategy frequencies at the equilibrium point.
- `params`: A `ReplicatorParams` struct containing the payoff functions and extra parameters.

# Returns
- `true` if the equilibrium is stable, `false` otherwise.
"""
    function check_stability(x_eq, params::ReplicatorParams; stability_tol::Float64 = 1e-6)
        # Reduced dynamics function
        function reduced_replicator(x_reduced)
            x1, x2 = x_reduced
            x3 = 1.0 - x1 - x2
            x_full = [x1, x2, x3]
            dx = replicator_equation(x_full, params, 0.0)
            # Return only the first two equations
            return dx[1:2]
        end
    
        # Compute Jacobian of the reduced system
        J_reduced = ForwardDiff.jacobian(reduced_replicator, x_eq[1:2])
    
        # Compute eigenvalues
        eigenvalues = eigvals(J_reduced)
        #println("Eigenvalues at equilibrium $x_eq: $eigenvalues")

        # If the real parts are all near zero (neutrally stable or oscillatory), return false
        if all(abs(real(eigenvalue)) < stability_tol for eigenvalue in eigenvalues)
            return false
        end

        # Otherwise, check if all real parts are strictly negative for stability
        return all(real(eigenvalues) .< -stability_tol)
    end

"""
    find_equilibria(
        payoff_functions::Tuple{Function, Function, Function},
        params::NamedTuple;
        num_initial_guesses::Int = 1000,
        tol::Float64 = 1e-6
    ) -> Tuple{Vector{Vector{Float64}}, Vector{Bool}}

Finds the equilibria of the replicator dynamics system by numerically solving for points where
the strategy frequencies stop changing. It also checks the stability of each equilibrium.

# Arguments
- `payoff_functions`: A tuple of three payoff functions corresponding to each strategy.
- `params`: Extra parameters for the payoff functions (optional).
- `num_initial_guesses`: Number of initial guesses to use for finding equilibria.
- `tol`: Tolerance for determining equilibria.

# Returns
- A tuple containing:
    - `equilibria`: A vector of strategy frequencies representing the equilibria.
    - `stability_status`: A vector of booleans indicating whether each equilibrium is stable.
"""
    function find_equilibria(
        payoff_functions::Tuple{Function, Function, Function},
        params::NamedTuple;
        num_initial_guesses::Int = 1000,
        solver_tol::Float64 = 1e-8,
        equilibrium_tol::Float64 = 1e-5,
        validity_tol::Float64 = 1e-6,
        stability_tol::Float64 = 1e-6
    )
        equilibria = Vector{Vector{Float64}}()
        stability_status = Vector{Bool}()
    
        # Generate initial guesses within the simplex
        initial_guesses = generate_initial_guesses(num_initial_guesses)
    
        # Prepare parameters
        replicator_params = ReplicatorParams(payoff_functions, params)
    
        for x0 in initial_guesses
            # Define the function whose root we want to find
            f(x) = replicator_equation(x, replicator_params, 0.0)
            # Apply the constraint x1 + x2 + x3 = 1
            x0 = x0 / sum(x0)
    
            # Use nlsolve to find the root
            result = try
                nlsolve(f, x0; method = :trust_region, xtol = solver_tol)
            catch
                # If solver fails, skip this initial guess
                continue
            end
    
            # Extract the solution
            x_eq = result.zero
    
            # Enforce the simplex constraint
            x_eq /= sum(x_eq)
    
            # Check if the solution is within the simplex and valid
            if all(x_eq .>= -validity_tol) && all(x_eq .<= 1 + validity_tol)
                # Check if this equilibrium is already found
                if !equilibrium_exists(x_eq, equilibria, equilibrium_tol)
                    push!(equilibria, x_eq)
                    # Determine stability
                    is_stable = check_stability(x_eq, replicator_params, stability_tol=stability_tol)
                    push!(stability_status, is_stable)
                end
            end
        end
    
        # Return equilibria and their stability status
        return equilibria, stability_status
    end

end # module Baryplots
