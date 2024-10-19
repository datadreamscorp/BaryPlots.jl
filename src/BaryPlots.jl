module BaryPlots

    using DifferentialEquations
	using Plots
	using LinearAlgebra
	using NLsolve
	using ForwardDiff

    export plot_evolution, ternary_coords, replicator_dynamics!, check_stability, find_equilibria, plot_simplex, ReplicatorParams, generate_simplex_grid

"""
    ternary_coords(x::Vector{Float64}) -> Tuple{Float64, Float64}

Converts a vector `x` of three strategy frequencies `[x₁, x₂, x₃]` into ternary coordinates `(X, Y)`
for plotting within a simplex (equilateral triangle).

# Arguments
- `x`: A vector of length 3, representing the frequencies of three strategies. The frequencies can sum to any positive value; they will be normalized within the function.

# Returns
- A tuple `(X, Y)` representing the ternary coordinates of the input vector.

# Notes
- This function ensures that the sum of the frequencies equals 1 by normalizing `x`.
- The ternary coordinates are calculated based on the normalized frequencies for plotting in a 2D simplex.
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

Plots an equilateral triangle representing the simplex for three strategies. Labels are added to the corners of the triangle.

# Arguments
- `labels`: A vector of three strings that label the corners of the simplex. Default: `["Strategy 1", "Strategy 2", "Strategy 3"]`.
- `kwargs`: Additional keyword arguments passed to the `plot` function.

# Returns
- A `Plot` object displaying the simplex triangle with the specified labels.

# Notes
- This function sets up the simplex plot, which can be used as a background for plotting evolutionary dynamics.
"""
    function plot_simplex(; labels::Vector{String} = ["Strategy 1", "Strategy 2", "Strategy 3"], kwargs...)
        triangle_x = [0.0, 1.0, 0.5, 0.0]
        triangle_y = [0.0, 0.0, sqrt(3)/2, 0.0]
        plot(triangle_x, triangle_y;
            seriestype = :shape,
            aspect_ratio = 1,
            linewidth = 2,
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

"""
    struct ReplicatorParams

Holds parameters for the replicator dynamics, including payoff functions and extra parameters.

# Fields
- `payoff_functions`: A tuple of three payoff functions `(payoff1, payoff2, payoff3)`, one for each strategy.
- `extra_params`: A `NamedTuple` containing additional parameters required by the payoff functions.
"""
    struct ReplicatorParams
        
        payoff_functions::Tuple{Function, Function, Function}  # Payoff functions for strategies
        extra_params::NamedTuple # Additional parameters for payoff functions
    end

"""
    replicator_dynamics!(dx::Vector{Float64}, x::Vector{Float64}, params::ReplicatorParams, t::Float64)

Computes the time derivative of the replicator dynamics for a population playing three strategies.

# Arguments
- `dx`: A vector to store the time derivative (change in frequencies).
- `x`: A vector representing the current frequencies of the three strategies.
- `params`: A `ReplicatorParams` struct containing the payoff functions and extra parameters.
- `t`: The current time in the dynamics.

# Notes
- The function normalizes `x` to ensure the frequencies sum to 1 before computing the payoffs.
- The replicator dynamics equation is: `dxᵢ = xᵢ * (wᵢ - w̄)`, where `wᵢ` is the payoff of strategy `i` and `w̄` is the average payoff.
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

"""
    create_steady_state_callback(steady_state_tol::Float64) -> DiscreteCallback

Creates a callback function for use with DifferentialEquations.jl solvers that terminates the integration when the system reaches a steady state.

# Arguments
- `steady_state_tol`: Tolerance for determining when the system has reached a steady state (i.e., when the norm of the derivative is below this value).

# Returns
- A `DiscreteCallback` object that can be passed to the solver.

# Notes
- The callback checks if the norm of the derivative is less than `steady_state_tol` and terminates the integration if so.
"""
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

"""
    generate_simplex_grid(resolution::Int, margin::Int = 1) -> Vector{Vector{Float64}}

Generates a list of points within the simplex by discretizing the simplex into a grid, excluding points near the boundaries as specified by the margin.

# Arguments
- `resolution`: The number of divisions along each edge of the simplex. Higher resolution results in more points.
- `margin`: The number of grid layers to exclude from the edges. Defaults to 1.

# Returns
- A vector of points, where each point is a vector `[x₁, x₂, x₃]` representing the frequencies of the three strategies at that grid point.

# Notes
- This function is used to generate points for plotting contour plots within the simplex.
"""
    function generate_simplex_grid(resolution::Int, margin::Int = 1)
        points = Vector{Vector{Float64}}()
        for i in margin:(resolution - margin)
            for j in margin:(resolution - i - margin)
                k = resolution - i - j
                if k ≥ margin
                    x1 = i / resolution
                    x2 = j / resolution
                    x3 = k / resolution
                    x = [x1, x2, x3]
                    push!(points, x)
                end
            end
        end
        return points
    end
    
"""
    compute_average_payoffs(grid_points::Vector{Vector{Float64}}, payoff_functions::Tuple{Function, Function, Function}, params::NamedTuple) -> Vector{Float64}

Computes the average payoff for each point in `grid_points` given the payoff functions and additional parameters.

# Arguments
- `grid_points`: A vector of points, where each point is a vector `[x₁, x₂, x₃]` representing the frequencies of the three strategies.
- `payoff_functions`: A tuple of three payoff functions corresponding to each strategy.
- `params`: A `NamedTuple` of additional parameters required by the payoff functions.

# Returns
- A vector of average payoffs corresponding to each point in `grid_points`.

# Notes
- The average payoff at each point is calculated as `avg_payoff = x₁*w₁ + x₂*w₂ + x₃*w₃`, where `wᵢ` is the payoff of strategy `i`.
"""
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
    
"""
    get_ternary_coordinates(grid_points::Vector{Vector{Float64}}) -> Tuple{Vector{Float64}, Vector{Float64}}

Converts a list of points in strategy frequency space to their corresponding ternary plot coordinates.

# Arguments
- `grid_points`: A vector of points, where each point is a vector `[x₁, x₂, x₃]` representing the frequencies of the three strategies.

# Returns
- A tuple `(X_coords, Y_coords)`, where `X_coords` and `Y_coords` are vectors containing the ternary plot coordinates for each point.

# Notes
- Uses `ternary_coords` to compute the coordinates for each point.
"""
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
        trajectory_linewidth::Int = 2,
        num_initial_guesses::Int = 1000,
        solver_tol::Float64 = 1e-8,
        equilibrium_tol::Float64 = 1e-5,
        validity_tol::Float64 = 1e-6,
        stability_tol::Float64 = 1e-6,
        eq_size::Int = 7,
        colored_trajectories::Bool = false,
        contourf::Bool = false,
        contour_resolution::Int = 150,
        contour_levels::Int = 10,
        cbar::Bool = false,
        triangle_linewidth::Int = 2,
        legend::Bool = false,
        margin::Int = 2,
        dpi = 300,
    )::Plots.Plot

Simulates the evolution of strategy frequencies over time and plots their trajectories within a ternary simplex. It also computes and plots the equilibria of the system and optionally overlays a contour plot.

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
- `trajectory_linewidth`: Line width for the trajectory lines.
- `num_initial_guesses`: Number of initial guesses for finding equilibria.
- `solver_tol`: Tolerance for the numerical solver when finding equilibria.
- `equilibrium_tol`: Tolerance for determining equilibria uniqueness.
- `validity_tol`: Tolerance for validating equilibrium points within the simplex.
- `stability_tol`: Tolerance for stability analysis of equilibria.
- `eq_size`: Size of the markers representing equilibria.
- `colored_trajectories`: Whether to color trajectories differently.
- `contourf`: Whether to include a contour plot of average payoffs.
- `contour_resolution`: Resolution of the contour plot grid.
- `contour_levels`: Number of contour levels.
- `cbar`: Whether to include a color bar in the contour plot.
- `triangle_linewidth`: Line width for the simplex triangle.
- `legend`: Whether to include a legend.
- `margin`: Margin to exclude points near the boundaries in the contour plot.
- `dpi`: Dots per inch for the plot resolution.

# Returns
- A `Plot` object displaying the simplex with the trajectories and equilibria.

# Notes
- Trajectories are simulated using the replicator dynamics defined by the provided payoff functions.
- Equilibria are computed numerically and their stability is assessed.
- If `contourf` is `true`, a contour plot of average payoffs is overlaid on the simplex.
"""
    function plot_evolution(
        payoff_functions::Tuple{Function, Function, Function},
        x0_list::Vector{<:AbstractVector{<:Real}},
        tspan::Tuple{Float64, Float64};
        labels::Vector{String} = ["Strategy 1", "Strategy 2", "Strategy 3"],
        extra_params::NamedTuple = NamedTuple(),
        steady_state_tol::Float64 = 1e-6,
        arrow_list::Vector{Vector{Int}} = Vector{Vector{Int}}(),
        trajectory_labels::Vector{String} = String[],
        trajectory_colors::AbstractVector = Any[],
        trajectory_linewidth::Int = 2,
        num_initial_guesses::Int = 1000,
        solver_tol::Float64 = 1e-8,
        equilibrium_tol::Float64 = 1e-5,
        validity_tol::Float64 = 1e-6,
        stability_tol::Float64 = 1e-6,
        eq_size::Int = 7,
        colored_trajectories::Bool = false,
        contourf::Bool = false,
        contour_resolution::Int = 150,
        contour_levels::Int = 10,
        cbar::Bool = false,
        triangle_linewidth::Int = 2,
        legend::Bool = false,
        margin::Int = 2,
        dpi = 300,
    )::Plots.Plot

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

        # Start plotting
        if contourf
            # Generate the simplex grid without boundary points
            grid_points = generate_simplex_grid(contour_resolution, margin)
            X_coords, Y_coords = get_ternary_coordinates(grid_points)
            Z_values = compute_average_payoffs(grid_points, payoff_functions, extra_params)
    
            # Create the contour plot using scatter with color mapping
            p = scatter(X_coords, Y_coords;
                zcolor = Z_values,
                markersize = 5,
                markerstrokewidth = 0,
                colorbar = cbar,
                legend = false,
                xlabel = "",
                ylabel = "",
                xlims = (-0.1, 1.1),
                ylims = (-0.1, sqrt(3)/2 + 0.1),
                aspect_ratio = 1,
                framestyle = :none,
                c = :viridis,
                dpi = dpi)
    
            # Plot the simplex boundaries
            triangle_x = [0.0, 1.0, 0.5, 0.0]
            triangle_y = [0.0, 0.0, sqrt(3)/2, 0.0]
            plot!(p, triangle_x, triangle_y, color=:black, linewidth=triangle_linewidth, legend=false)
    
            # Add labels
            annotate!(p, 0.0, -0.05, text(labels[2], :center))
            annotate!(p, 1.0, -0.05, text(labels[1], :center))
            annotate!(p, 0.5, sqrt(3)/2 + 0.05, text(labels[3], :center))
        else
            # Existing code to plot the simplex without contour
            p = plot_simplex(labels = labels, linewidth=triangle_linewidth)
        end

        # Plotting trajectories
        for i in 1:num_trajectories
            x0 = x0_list[i] / sum(x0_list[i])  # Ensure x0 sums to 1
            prob = ODEProblem(replicator_dynamics!, x0, tspan, params)
            sol = solve(prob; callback = cb,
                        saveat = tspan[1]:0.01:tspan[2], tstops = [tspan[2]],
                        reltol = 1e-8, abstol = 1e-8,
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
            plot!(
                p, X, Y;
                linecolor = trajectory_colors[i],
                legend = legend,
                label = trajectory_labels[i],
                linewidth = trajectory_linewidth,
                )

            # Add arrows along the trajectory if requested
            if length(arrow_list) ≥ i && !isempty(arrow_list[i])
                arrow_scale = 0.01  # Adjust this value to change arrow size
                for j in arrow_list[i]
                    if j > 0 && j < length(X)
                        x_start, y_start = X[j], Y[j]
                        x_end, y_end = X[j+1], Y[j+1]
                        dx_arrow = x_end - x_start
                        dy_arrow = y_end - y_start
                        norm_factor = sqrt(dx_arrow^2 + dy_arrow^2)
                        if norm_factor != 0
                            dx_arrow /= norm_factor
                            dy_arrow /= norm_factor
                            dx_arrow *= arrow_scale
                            dy_arrow *= arrow_scale
                            quiver!(
                                p,
                                [x_start], 
                                [y_start], 
                                quiver=(
                                    [dx_arrow], [dy_arrow]);
                                    arrow=:closed, color=trajectory_colors[i],
                                    linewidth=trajectory_linewidth, label=false
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
                    scatter!(p, [X_eq], [Y_eq];
                        markercolor = :black,
                        markershape = :circle,
                        markersize = eq_size,
                        label = false)
                else
                    # Plot unstable equilibrium as hollow circle
                    scatter!(p, [X_eq], [Y_eq];
                        markercolor = :white,
                        markerstrokecolor = :black,
                        markershape = :circle,
                        markersize = eq_size,
                        label = false)
                end
            end
        end

        # Show legend if trajectories are labeled
        if any(.!isempty.(trajectory_labels))
            plot!(p, legend = legend)
        else
            plot!(p, legend = legend)
        end

        return p
    end

"""
    generate_initial_guesses(n::Int) -> Vector{Vector{Float64}}

Generates random initial guesses within the simplex for finding equilibria.

# Arguments
- `n`: The number of initial guesses to generate.

# Returns
- A vector of vectors, each representing a point `[x₁, x₂, x₃]` within the simplex (summing to 1).

# Notes
- The generated points are uniformly random within the simplex.
"""
    function generate_initial_guesses(n)
        guesses = []
        for _ in 1:n
            x = rand(3)
            x /= sum(x)  # Normalize to sum to 1
            push!(guesses, x)
        end
        return guesses
    end

"""
    replicator_equation(x::Vector{Float64}, params::ReplicatorParams, t::Float64) -> Vector{Float64}

Computes the replicator dynamics equation at a given point.

# Arguments
- `x`: A vector representing the strategy frequencies.
- `params`: A `ReplicatorParams` struct containing the payoff functions and extra parameters.
- `t`: The current time.

# Returns
- A vector `dx` representing the time derivatives of the strategy frequencies.

# Notes
- This function is used in numerical root-finding to find equilibria.
"""
    function replicator_equation(x, params::ReplicatorParams, t)
        dx = similar(x)
        replicator_dynamics!(dx, x, params, t)
        return dx
    end

"""
    equilibrium_exists(x_new::Vector{Float64}, equilibria::Vector{Vector{Float64}}, equilibrium_tol::Float64) -> Bool

Checks if a newly found equilibrium already exists in the list of equilibria.

# Arguments
- `x_new`: The new equilibrium point to check.
- `equilibria`: A vector of existing equilibria.
- `equilibrium_tol`: Tolerance for considering two equilibria as the same.

# Returns
- `true` if the equilibrium already exists, `false` otherwise.

# Notes
- Normalizes `x_new` and existing equilibria to sum to 1 before comparison.
"""
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
    check_stability(x_eq::Vector{Float64}, params::ReplicatorParams; stability_tol::Float64 = 1e-6) -> Bool

Determines the stability of an equilibrium point by analyzing the eigenvalues of the Jacobian matrix of the reduced system.

# Arguments
- `x_eq`: A vector representing the strategy frequencies at the equilibrium point.
- `params`: A `ReplicatorParams` struct containing the payoff functions and extra parameters.
- `stability_tol`: Tolerance for determining the stability based on eigenvalues.

# Returns
- `true` if the equilibrium is stable, `false` otherwise.

# Notes
- The reduced system excludes one variable due to the simplex constraint (sum to 1).
- Stability is determined by the sign of the real parts of the eigenvalues.
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
        solver_tol::Float64 = 1e-8,
        equilibrium_tol::Float64 = 1e-5,
        validity_tol::Float64 = 1e-6,
        stability_tol::Float64 = 1e-6
    ) -> Tuple{Vector{Vector{Float64}}, Vector{Bool}}

Finds the equilibria of the replicator dynamics system by numerically solving for points where the strategy frequencies stop changing.

# Arguments
- `payoff_functions`: A tuple of three payoff functions corresponding to each strategy.
- `params`: Extra parameters for the payoff functions (optional).
- `num_initial_guesses`: Number of initial guesses to use for finding equilibria.
- `solver_tol`: Tolerance for the numerical solver.
- `equilibrium_tol`: Tolerance for considering equilibria as unique.
- `validity_tol`: Tolerance for validating equilibria within the simplex.
- `stability_tol`: Tolerance for stability analysis.

# Returns
- A tuple containing:
    - `equilibria`: A vector of strategy frequencies representing the equilibria.
    - `stability_status`: A vector of booleans indicating whether each equilibrium is stable.

# Notes
- Uses `nlsolve` to find roots of the replicator equation.
- Filters out equilibria that are outside the simplex or are duplicates.
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
