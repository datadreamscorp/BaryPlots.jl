module BaryPlots

    using DifferentialEquations
	using Plots
	using LinearAlgebra
	using NLsolve
	using ForwardDiff
    using LaTeXStrings

    export plot_evolution, ternary_coords, replicator_dynamics!, check_stability, find_equilibria, plot_simplex, ReplicatorParams, generate_simplex_grid, identify_neutral_edges

"""
    get_vertices() -> Vector{Vector{Float64}}

Returns the corner points (vertices) of the 3-strategy simplex.

# Returns
- A vector of three vectors, `[ [1,0,0], [0,1,0], [0,0,1] ]`, each representing a pure strategy in the simplex.

# Notes
- These vertices are used as initial guesses for pure-strategy equilibria when searching for equilibria numerically.
"""
    function get_vertices()::Vector{Vector{Float64}}
        return [
            [1.0, 0.0, 0.0],  # Vertex for Strategy 1
            [0.0, 1.0, 0.0],  # Vertex for Strategy 2
            [0.0, 0.0, 1.0]   # Vertex for Strategy 3
        ]
    end

"""
    arrow0!(x, y, u, v; as=0.07, lw=1, lc=:black, la=1)

Draws a custom arrow on an existing plot, starting at `(x, y)` and extending by `(u, v)`.

# Arguments
- `x`, `y`: Coordinates of the arrow's starting point.
- `u`, `v`: The change in x- and y- coordinates from the start to the tip of the arrow.
- `as`: Arrow size scaling factor. Default is `0.07`.
- `lw`: Line width for the arrow body and head. Default is `1`.
- `lc`: Line color for the arrow body and head. Default is `:black`.
- `la`: Line alpha (transparency) for the arrow. Default is `1` (fully opaque).

# Notes
- The arrow head is drawn with two short lines forming a closed arrow tip.
- `arrow0!` is used internally for drawing direction indicators on replicator dynamics trajectories.
- This function modifies the plot in-place via `plot!`.
"""
    function arrow0!(x, y, u, v; as=0.07, lw=1, lc=:black, la=1)
        nuv = sqrt(u^2 + v^2)
        v1, v2 = [u;v] / nuv,  [-v;u] / nuv
        v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
        v5 = v4 - 2*(v4'*v2)*v2
        v4, v5 = as*nuv*v4, as*nuv*v5
        plot!([x,x+u], [y,y+v], lw=lw, lc=lc, la=la)
        plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], lw=lw, lc=lc, la=la)
        plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], lw=lw, lc=lc, la=la)
    end

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
    plot_simplex(; 
        labels::Vector{String} = ["Strategy 1", "Strategy 2", "Strategy 3"], 
        neutral_edges::Vector{Int} = Int[], 
        plot_neutral_dots::Bool = false, 
        p = nothing, 
        kwargs...
    ) -> Plots.Plot

Plots an equilateral triangle representing the simplex for three strategies. Optionally highlights certain edges as "neutral" by adding dotted markers if requested.

# Arguments
- `labels`: A vector of three strings labeling the corners of the simplex. Default: `["Strategy 1", "Strategy 2", "Strategy 3"]`.
- `neutral_edges`: A vector of integers indicating which edges are neutral. Edges are indexed as:
    - `1`: Edge between Strategy 1 and Strategy 2
    - `2`: Edge between Strategy 2 and Strategy 3
    - `3`: Edge between Strategy 3 and Strategy 1
- `plot_neutral_dots`: If `true`, draws a sequence of white dots on each specified neutral edge to visually mark them. Default: `false`.
- `p`: An existing `Plots.Plot` object to draw on. If `nothing`, a new plot is created.
- `kwargs`: Additional keyword arguments passed to `plot` for further customization.

# Returns
- A `Plots.Plot` object displaying the simplex with labeled corners. Edges in `neutral_edges` may be visually highlighted.

# Notes
- When `plot_neutral_dots` is `false`, the edges in `neutral_edges` are simply not drawn (or drawn in a minimal style, as configured in the source).
- If you want to add further customizations, you can pass typical Plots.jl arguments (e.g., `linecolor`, `linestyle`, etc.) through `kwargs`.
"""
    function plot_simplex(; 
        labels = ["Strategy 1", "Strategy 2", "Strategy 3"], 
        neutral_edges::Vector{Int} = Int[], 
        plot_neutral_dots::Bool = false, 
        p = nothing, 
        kwargs...)::Plots.Plot

        if isnothing(p)
            p = plot(; 
                aspect_ratio = 1, 
                xlims = (-0.1, 1.1), 
                ylims = (-0.1, sqrt(3)/2 + 0.1), 
                framestyle = :none, 
                xlabel = "", 
                ylabel = "", 
                kwargs...)
        end

        # Define the three edges of the simplex based on updated ordering
        edges = [
            ([0.0, 1.0], [0.0, 0.0]),                # Edge 1: Strategy 2 ↔ Strategy 1 (Bottom Left ↔ Bottom Right)
            ([1.0, 0.5], [0.0, sqrt(3)/2]),         # Edge 2: Strategy 1 ↔ Strategy 3 (Bottom Right ↔ Top)
            ([0.5, 0.0], [sqrt(3)/2, 0.0])          # Edge 3: Strategy 3 ↔ Strategy 2 (Top ↔ Bottom Left)
        ]

        # Define number of dots for neutral edges
        num_dots = 15  # Adjust as needed for spacing

        for (idx, (x_coords, y_coords)) in enumerate(edges)
            if plot_neutral_dots
                if idx in neutral_edges
                    plot!(p, x_coords, y_coords; 
                        seriestype = :path, 
                        linewidth = 2, 
                        linecolor = :black, 
                        linestyle = :solid, 
                        legend = false)
                    
                    # Plot neutral edges as a sequence of white dots
                    x_start, x_end = x_coords
                    y_start, y_end = y_coords

                    # Generate evenly spaced points along the edge
                    for i in 1:num_dots
                        frac = (i - 1) / (num_dots - 1)
                        x = x_start + frac * (x_end - x_start)
                        y = y_start + frac * (y_end - y_start)
                        scatter!(p, [x], [y]; 
                            markercolor = :white, 
                            markersize = 9, 
                            markerstrokewidth = 2,
                            label = false)
                    end
                end
            else
                # Plot regular edges
                if !(idx in neutral_edges)  # Plot only non-neutral edges
                    plot!(p, x_coords, y_coords; 
                        seriestype = :path, 
                        linewidth = 2, 
                        linecolor = :black, 
                        linestyle = :solid, 
                        legend = false)
                else
                    # Optionally, plot regular lines for neutral edges with reduced opacity
                    # Uncomment the following lines if desired
                    # plot!(p, x_coords, y_coords; 
                    #     seriestype = :path, 
                    #     linewidth = 2, 
                    #     linecolor = :black, 
                    #     linestyle = :solid, 
                    #     alpha = 0.3, 
                    #     legend = false)
                end
            end
        end

        # Add labels to the corners
        if !plot_neutral_dots
            annotate!(p, 0.0, -0.1, text(labels[2], :center))  # Left corner (Strategy 2)
            annotate!(p, 1.0, -0.1, text(labels[1], :center))  # Right corner (Strategy 1)
            annotate!(p, 0.5, sqrt(3)/2 + 0.1, text(labels[3], :center)) # Top corner (Strategy 3)
        end

        return p
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
    identify_neutral_edges(params::ReplicatorParams; tol::Float64 = 1e-8, samples::Int = 100) -> Vector{Int}

Identifies "neutral" edges in the simplex under the given payoff functions. An edge is neutral if, for all sampled points on that edge, the payoffs of the two involved strategies are effectively equal.

# Arguments
- `params`: A `ReplicatorParams` struct containing the payoff functions and extra parameters.
- `tol`: Tolerance for payoff difference. If `|wᵢ - wⱼ|` < `tol` across all samples, that edge is considered neutral.
- `samples`: Number of points to sample along each edge. Default is `100`.

# Returns
- A vector of edge indices `{1,2,3}`, each corresponding to:
  - `1`: Edge between Strategy 1 and Strategy 2
  - `2`: Edge between Strategy 2 and Strategy 3
  - `3`: Edge between Strategy 3 and Strategy 1

# Notes
- This function is used inside `plot_evolution` to highlight edges where the game exhibits neutral stability. 
"""
    function identify_neutral_edges(params::ReplicatorParams; tol::Float64 = 1e-8, samples::Int = 100)::Vector{Int}
        edges = [(2, 1), (1, 3), (3, 2)]
        neutral_edges = Int[]  # List to store indices of neutral edges

        for (idx, (i, j)) in enumerate(edges)
            is_neutral = true
            for s in 0:(samples - 1)
                frac = s / (samples - 1)
                # Define frequencies for strategies i and j
                x = zeros(3)
                x[i] = 1.0 - frac
                x[j] = frac
                # Compute payoffs for strategies i and j
                w_i = params.payoff_functions[i](x, 0.0, params.extra_params)
                w_j = params.payoff_functions[j](x, 0.0, params.extra_params)
                # Check if payoffs are equal within tolerance
                if abs(w_i - w_j) > tol
                    is_neutral = false
                    break
                end
            end
            if is_neutral
                push!(neutral_edges, idx)
            end
        end

        return neutral_edges
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
        markersize::Int = 9,
        markerstrokewidth::Int = 2,
        colored_trajectories::Bool = false,
        contour::Bool = false,
        contour_color::Symbol = :roma,
        contour_resolution::Int = 150,
        contour_levels::Int = 10,
        colorbar::Bool = false,
        triangle_linewidth::Int = 2,
        legend::Bool = false,
        margin::Int = 2,
        dpi::Int = 300,
    )::Plots.Plot

Simulates replicator dynamics trajectories within a 3-strategy simplex, locates equilibria, and creates a ternary plot of the results.

# Arguments
- `payoff_functions`: A tuple of three payoff functions `(payoff1, payoff2, payoff3)`, each accepting `(x, t, params)` where `x` is the normalized frequency vector.
- `x0_list`: A collection of initial conditions in the simplex (each a length-3 vector). Each trajectory is simulated from these initial points.
- `tspan`: A tuple `(tstart, tfinal)` specifying the integration window.
- `labels`: Labels for the triangle corners. Default is `["Strategy 1", "Strategy 2", "Strategy 3"]`.
- `extra_params`: Additional parameters passed to the payoff functions, packaged in a `NamedTuple`.
- `steady_state_tol`: If the norm of the derivative falls below this threshold, the solver stops early (via a callback).
- `arrow_list`: An array-of-arrays specifying indices at which to draw an arrow along each trajectory.
- `trajectory_labels`: Labels for each trajectory if you want a legend. Defaults to "Trajectory i" for i in `1:length(x0_list)`.
- `trajectory_colors`: Colors for trajectories. If empty, defaults to black for all or a tab10 palette if `colored_trajectories` is true.
- `trajectory_linewidth`: Width of the lines used for plotting trajectories.
- `num_initial_guesses`: Number of random initial guesses for root-finding to locate equilibria.
- `solver_tol`: Tolerance used internally by the nonlinear solver for equilibrium-finding.
- `equilibrium_tol`: Tolerance for declaring two equilibria to be effectively the same.
- `validity_tol`: Tolerance for points to be considered inside the simplex (0 ≤ xᵢ ≤ 1).
- `stability_tol`: Tolerance used when deciding if all eigenvalues have negative real parts (stable).
- `markersize`: Size of equilibrium markers.
- `markerstrokewidth`: Stroke width of the equilibrium marker boundary.
- `colored_trajectories`: If `true`, color each trajectory differently using a color palette; otherwise, use a single color.
- `contour`: Whether to overlay a contour plot of average payoffs. If `true`, uses `scatter` to color each simplex grid point by its average payoff.
- `contour_color`: A symbol representing the color gradient (e.g. `:viridis`, `:roma`) for the contour plot.
- `contour_resolution`: Resolution of the simplex grid for contour plotting.
- `contour_levels`: Number of contour levels to display (used in color scaling).
- `colorbar`: Whether to display a color bar when `contour` is `true`.
- `triangle_linewidth`: Line width for the simplex boundary lines.
- `legend`: Whether to display a legend for trajectory labels.
- `margin`: How many rows/columns of the simplex grid to skip from the edges when computing and plotting contour points (avoid overlapping boundaries).
- `dpi`: Resolution for the final plot.

# Returns
- A `Plots.Plot` object with:
  1. An (optional) contour plot of average payoffs in the simplex.
  2. Ternary simplex edges (with neutral edges optionally highlighted).
  3. Simulated trajectories from the provided initial conditions.
  4. Detected equilibria, marked as filled circles if stable or hollow circles if unstable.

# Notes
- Internally calls `identify_neutral_edges` to highlight edges where payoffs between two strategies are equal for all points along that edge.
- Equilibria are located with `find_equilibria`, and each one is tested for stability by analyzing the Jacobian.
- The arrow drawing is handled by `arrow0!`, with user-specified arrow positions in `arrow_list`.
"""
    function plot_evolution(
        payoff_functions::Tuple{Function, Function, Function},
        x0_list::Vector{<:AbstractVector{<:Real}},
        tspan::Tuple{Float64, Float64};
        labels::Vector{} = ["Strategy 1", "Strategy 2", "Strategy 3"],
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
        markersize::Int = 9,
        markerstrokewidth::Int = 2,
        colored_trajectories::Bool = false,
        contour::Bool = false,
        contour_color::Symbol = :roma,
        contour_resolution::Int = 150,
        contour_levels::Int = 10,
        colorbar::Bool = false,
        triangle_linewidth::Int = 2,
        legend::Bool = false,
        margin::Int = 2,
        dpi::Int = 300,
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

        # Identify neutral edges
        neutral_edges = identify_neutral_edges(params)

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
        if contour
            # Generate the simplex grid without boundary points
            grid_points = generate_simplex_grid(contour_resolution, margin)
            X_coords, Y_coords = get_ternary_coordinates(grid_points)
            Z_values = compute_average_payoffs(grid_points, payoff_functions, extra_params)
    
            # Create the contour plot using scatter with color mapping
            p = scatter(
                X_coords, Y_coords;
                zcolor = Z_values,
                markersize = 5,
                markerstrokewidth = 0,
                colorbar = colorbar,
                legend = false,
                xlabel = "",
                ylabel = "",
                xlims = (-0.1, 1.1),
                ylims = (-0.1, sqrt(3)/2 + 0.1),
                aspect_ratio = 1,
                framestyle = :none,
                c = contour_color,
                dpi = dpi
                )
    
            # Plot the simplex boundaries
            triangle_x = [0.0, 1.0, 0.5, 0.0]
            triangle_y = [0.0, 0.0, sqrt(3)/2, 0.0]
            plot!(p, triangle_x, triangle_y, color=:black, linewidth=triangle_linewidth, legend=false)
    
            # Add labels
            annotate!(p, 0.0, -0.1, text(labels[2], :center))
            annotate!(p, 1.0, -0.1, text(labels[1], :center))
            annotate!(p, 0.5, sqrt(3)/2 + 0.1, text(labels[3], :center))
        else
            # Existing code to plot the simplex without contour
            p = plot_simplex(labels = labels, neutral_edges = neutral_edges, linewidth=triangle_linewidth)
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
                            arrow0!(x_start, y_start, dx_arrow, dy_arrow; as=4.0, lc=:black, la=1, lw=2)
                        end
                    end
                end
            end
        end

        p = plot_simplex(
            labels = labels, 
            neutral_edges = neutral_edges, 
            plot_neutral_dots = true, 
            p = p, 
            triangle_linewidth = triangle_linewidth
        )

        # Plot the equilibria
        if !isempty(equilibria)
            for (i, x_eq) in enumerate(equilibria)
                is_stable = stability_status[i]
                # Ensure the equilibrium point sums to 1
                x_eq = x_eq / sum(x_eq)
                X_eq, Y_eq = ternary_coords(x_eq)

                 # Determine if equilibrium lies on a neutral edge
                zero_tol = 1e-2  # Tolerance for zero frequency
                edge_idx = 0  # Initialize edge index

                if abs(x_eq[3]) < zero_tol
                    edge_idx = 1  # Edge between Strategy 2 and Strategy 1
                elseif abs(x_eq[2]) < zero_tol
                    edge_idx = 2  # Edge between Strategy 1 and Strategy 3
                elseif abs(x_eq[1]) < zero_tol
                    edge_idx = 3  # Edge between Strategy 3 and Strategy 2
                end     
                
                # Check if equilibrium is at a vertex (two frequencies near zero)
                num_zero = sum([abs(x_eq[1]) < zero_tol, 
                abs(x_eq[2]) < zero_tol, 
                abs(x_eq[3]) < zero_tol])
                is_vertex = num_zero >= 2

                # If equilibrium is on a neutral edge and is unstable, skip plotting
                if !( (edge_idx != 0 && edge_idx in neutral_edges) ) || is_vertex

                    if is_stable
                        # Plot stable equilibrium as filled black circle
                        scatter!(p, [X_eq], [Y_eq];
                            markercolor = :black,
                            markershape = :circle,
                            markersize = markersize,
                            markerstrokewidth = markerstrokewidth,
                            label = false)
                    else
                        # Plot unstable equilibrium as hollow circle
                        scatter!(p, [X_eq], [Y_eq];
                            markercolor = :white,
                            markerstrokecolor = :black,
                            markershape = :circle,
                            markersize = markersize,
                            markerstrokewidth = markerstrokewidth,
                            label = false)
                    end

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

        # Get vertices and add to initial guesses
        vertices = get_vertices()
        initial_guesses = vcat(vertices, initial_guesses)
    
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
