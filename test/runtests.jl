using Test
using Plots, BaryPlots
using LinearAlgebra

@testset "ternary_coords tests" begin
    @test ternary_coords([1.0, 0.0, 0.0]) == (1.0, 0.0)  # Testing corner
    @test ternary_coords([0.0, 1.0, 0.0]) == (0.0, 0.0)  # Testing corner
    @test ternary_coords([0.0, 0.0, 1.0]) == (0.5, sqrt(3)/2)  # Top corner
    @test ternary_coords([0.5, 0.5, 0.0]) == (0.5, 0.0)  # Testing edge
    @test ternary_coords([0.0, 0.5, 0.5]) == (1/4, sqrt(3)/4)  # Testing edge
    @test ternary_coords([0.5, 0.0, 0.5]) == (3/4, sqrt(3)/4)  # Testing edge
end

@testset "ternary_coords normalization tests" begin
    @test ternary_coords([2.0, 0.0, 0.0]) == (1.0, 0.0)  # Should normalize to [1.0, 0.0, 0.0]
    @test ternary_coords([0.0, 2.0, 0.0]) == (0.0, 0.0)
    @test ternary_coords([0.0, 0.0, 2.0]) == (0.5, sqrt(3)/2)
    @test ternary_coords([1.0, 1.0, 1.0]) == (0.5, sqrt(3)/6)
end

#=
@testset "ternary_coords invalid input tests" begin
    @test_throws DomainError ternary_coords([-1.0, 1.0, 1.0])  # Negative frequency
    @test_throws DomainError ternary_coords([0.0, 0.0, 0.0])    # All zeros
end
=#

@testset "ternary_coords precision tests" begin
    @test all( isapprox.(ternary_coords([1e-10, 1.0 - 1e-10, 0.0]), [0.0, 0.0]; atol=1e-8) )
    @test all( isapprox.(ternary_coords([1.0 - 1e-10, 1e-10, 0.0]), [1.0, 0.0]; atol=1e-8) )
end

@testset "inverse_ternary_coords tests" begin
    # Corners
    @test isapprox(inverse_ternary_coords(1.0, 0.0), [1.0, 0.0, 0.0]; atol=1e-12)
    @test isapprox(inverse_ternary_coords(0.0, 0.0), [0.0, 1.0, 0.0]; atol=1e-12)
    @test isapprox(inverse_ternary_coords(0.5, sqrt(3)/2), [0.0, 0.0, 1.0]; atol=1e-12)
    # Centroid
    @test isapprox(inverse_ternary_coords(0.5, sqrt(3)/6), [1/3, 1/3, 1/3]; atol=1e-12)
end

@testset "inverse_ternary_coords round-trip tests" begin
    test_points = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
        [1/3, 1/3, 1/3], [0.6, 0.3, 0.1], [0.15, 0.05, 0.8],
    ]
    for x in test_points
        X, Y = ternary_coords(x)
        @test isapprox(inverse_ternary_coords(X, Y), x; atol=1e-12)
        # Recovered coordinates are a valid distribution
        @test isapprox(sum(inverse_ternary_coords(X, Y)), 1.0; atol=1e-12)
    end
end

@testset "inverse_ternary_coords outside-simplex tests" begin
    # Points in the bounding box but outside the triangle must have a negative component
    @test any(inverse_ternary_coords(0.05, sqrt(3)/2 - 0.05) .< 0)  # above the left edge
    @test any(inverse_ternary_coords(0.95, sqrt(3)/2 - 0.05) .< 0)  # above the right edge
    # Components still sum to 1 even outside the simplex
    @test isapprox(sum(inverse_ternary_coords(0.05, sqrt(3)/2 - 0.05)), 1.0; atol=1e-12)
end

# Basin of attraction tests.
#
# The game wᵢ(x) = xᵢ makes every strategy do best when it is common, so all three
# vertices are stable and the replicator dynamics carry any interior point to the
# vertex with the largest initial frequency. That gives an exact prediction for the
# basin labels: argmax(x0).

@testset "classify_basin tests" begin
    coord_payoff_1(x, t, params) = x[1]
    coord_payoff_2(x, t, params) = x[2]
    coord_payoff_3(x, t, params) = x[3]

    params = ReplicatorParams((coord_payoff_1, coord_payoff_2, coord_payoff_3), NamedTuple())
    stable_eq_points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    tspan = (0.0, 100.0)

    # Each initial condition flows to the vertex it is closest to dominating
    @test BaryPlots.classify_basin([0.6, 0.2, 0.2], params, stable_eq_points, tspan) == 1
    @test BaryPlots.classify_basin([0.2, 0.6, 0.2], params, stable_eq_points, tspan) == 2
    @test BaryPlots.classify_basin([0.2, 0.2, 0.6], params, stable_eq_points, tspan) == 3
    @test BaryPlots.classify_basin([0.45, 0.4, 0.15], params, stable_eq_points, tspan;
                                   steady_state_tol = 1e-6) == 1

    # Starting at a stable equilibrium stays there
    @test all(
        BaryPlots.classify_basin(copy(eq), params, stable_eq_points, tspan) == i
        for (i, eq) in enumerate(stable_eq_points)
    )

    # No stable equilibria supplied => label 0
    @test BaryPlots.classify_basin([0.6, 0.2, 0.2], params, Vector{Vector{Float64}}(), tspan) == 0
end

@testset "classify_basin outside-all-basins tests" begin
    coord_payoff_1(x, t, params) = x[1]
    coord_payoff_2(x, t, params) = x[2]
    coord_payoff_3(x, t, params) = x[3]

    params = ReplicatorParams((coord_payoff_1, coord_payoff_2, coord_payoff_3), NamedTuple())
    tspan = (0.0, 100.0)

    # Only vertex 1 is declared stable, but [0.2, 0.6, 0.2] flows to vertex 2.
    # With a single stable point supplied, the "outside all basins" label is n+1 = 2.
    n = 1
    @test BaryPlots.classify_basin(
        [0.2, 0.6, 0.2], params, [[1.0, 0.0, 0.0]], tspan
    ) == n + 1
    @test BaryPlots.classify_basin(
        [0.6, 0.2, 0.2], params, [[1.0, 0.0, 0.0]], tspan
    ) == 1
end

@testset "classify_basin error handling tests" begin
    stable_eq_points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    tspan = (0.0, 100.0)

    # A DomainError from the payoff functions is a numerical failure: report the point
    # as unclassified rather than aborting the whole basin sweep.
    domain_1(x, t, params) = throw(DomainError(x[1], "boom"))
    domain_rest(x, t, params) = 0.0
    domain_params = ReplicatorParams((domain_1, domain_rest, domain_rest), NamedTuple())
    @test BaryPlots.classify_basin(
        [0.6, 0.2, 0.2], domain_params, stable_eq_points, tspan
    ) == 0

    # Any other exception is a bug and must propagate, not be relabelled as 0.
    # (This is what previously hid a genuine UndefVarError behind a `0` label.)
    bug_1(x, t, params) = params.nonexistent_field
    bug_rest(x, t, params) = 0.0
    bug_params = ReplicatorParams((bug_1, bug_rest, bug_rest), NamedTuple())
    @test_throws Exception BaryPlots.classify_basin(
        [0.6, 0.2, 0.2], bug_params, stable_eq_points, tspan
    )
end

@testset "basin_label_grid tests" begin
    coord_payoff_1(x, t, params) = x[1]
    coord_payoff_2(x, t, params) = x[2]
    coord_payoff_3(x, t, params) = x[3]

    params = ReplicatorParams((coord_payoff_1, coord_payoff_2, coord_payoff_3), NamedTuple())
    stable_eq_points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    tspan = (0.0, 100.0)
    res = 11

    xs, ys, Z = BaryPlots.basin_label_grid(
        params, stable_eq_points, tspan; resolution = res
    )

    # Shapes and ranges
    @test length(xs) == res
    @test length(ys) == res
    @test size(Z) == (res, res)
    @test isapprox(first(xs), 0.0) && isapprox(last(xs), 1.0)
    @test isapprox(first(ys), 0.0) && isapprox(last(ys), sqrt(3)/2)

    # Labels are 0 (outside the simplex) or a valid basin index
    @test all(0 .<= Z .<= length(stable_eq_points) + 1)

    # Points outside the simplex are unlabelled, points inside are labelled
    @test all(
        (all(inverse_ternary_coords(X, Y) .>= -1e-8)) == (Z[ix, iy] != 0)
        for (ix, X) in enumerate(xs), (iy, Y) in enumerate(ys)
    )

    # The top corners of the grid lie outside the simplex
    @test Z[1, res] == 0    # (X=0, Y=√3/2): above the left edge
    @test Z[res, res] == 0  # (X=1, Y=√3/2): above the right edge

    # Interior labels match the argmax prediction. Skip the boundary, where argmax is
    # ambiguous, and near-ties, where the trajectory can land either way.
    interior = [
        (Z[ix, iy], argmax(inverse_ternary_coords(X, Y)))
        for (ix, X) in enumerate(xs), (iy, Y) in enumerate(ys)
        if let x = inverse_ternary_coords(X, Y), s = sort(inverse_ternary_coords(X, Y); rev = true)
            all(x .>= 1e-3) && s[1] - s[2] > 1e-2
        end
    ]
    @test !isempty(interior)
    @test all(label == predicted for (label, predicted) in interior)
end

@testset "plot_evolution basin plotting tests" begin
    coord_payoff_1(x, t, params) = x[1]
    coord_payoff_2(x, t, params) = x[2]
    coord_payoff_3(x, t, params) = x[3]

    payoffs = (coord_payoff_1, coord_payoff_2, coord_payoff_3)
    x0_list = [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]
    tspan = (0.0, 50.0)

    # Separatrix only
    p = plot_evolution(payoffs, x0_list, tspan;
        plot_basins = true, basin_resolution = 9,
        equilibria_kwargs = (num_initial_guesses = 100,))
    @test p isa Plots.Plot

    # Separatrix plus shading
    p_shaded = plot_evolution(payoffs, x0_list, tspan;
        plot_basins = true, basin_shade = true, basin_resolution = 9,
        basin_alpha = 0.2, equilibria_kwargs = (num_initial_guesses = 100,))
    @test p_shaded isa Plots.Plot
end

@testset "plot_evolution basin warning tests" begin
    # Rock-Paper-Scissors has no stable equilibrium, so basin plotting is skipped
    rps_payoff_R(x, t, params) = -x[2] + x[3]
    rps_payoff_P(x, t, params) = x[1] - x[3]
    rps_payoff_S(x, t, params) = -x[1] + x[2]

    p = @test_logs (:warn,) match_mode=:any plot_evolution(
        (rps_payoff_R, rps_payoff_P, rps_payoff_S),
        [[0.4, 0.4, 0.2]],
        (0.0, 20.0);
        plot_basins = true, basin_resolution = 8,
        equilibria_kwargs = (num_initial_guesses = 100,),
    )
    @test p isa Plots.Plot
end

@testset "integer and rational input tests" begin
    # Any numeric type that converts unambiguously to a float should be accepted
    @test isapprox(inverse_ternary_coords(0, 0), [0.0, 1.0, 0.0]; atol=1e-12)
    @test isapprox(inverse_ternary_coords(1//2, 0), [0.5, 0.5, 0.0]; atol=1e-12)
    @test all(isapprox.(ternary_coords([1, 0, 0]), (1.0, 0.0); atol=1e-12))

    coord_payoff_1(x, t, params) = x[1]
    coord_payoff_2(x, t, params) = x[2]
    coord_payoff_3(x, t, params) = x[3]

    payoffs = (coord_payoff_1, coord_payoff_2, coord_payoff_3)
    params = ReplicatorParams(payoffs, NamedTuple())

    # Integer initial condition, integer stable equilibria, integer tspan.
    # [3, 1, 1] does not sum to 1; classify_basin normalises it onto the simplex.
    @test BaryPlots.classify_basin(
        [3, 1, 1], params, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0, 100)
    ) == 1

    # Integer tolerances and grid resolution
    xs, ys, Z = BaryPlots.basin_label_grid(
        params, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0, 100);
        resolution = Int32(7), steady_state_tol = 1//1000000, convergence_tol = 1//10,
    )
    @test size(Z) == (7, 7)
    @test eltype(xs) == Float64

    # Integer styling and time span through the top-level plotting function
    p = plot_evolution(payoffs, [[1, 1, 2]], (0, 20);
        markersize = 10, markerstrokewidth = 2, trajectory_linewidth = 3,
        triangle_linewidth = 1, steady_state_tol = 1//1000000,
        equilibria_kwargs = (num_initial_guesses = 50,))
    @test p isa Plots.Plot

    # check_stability and find_equilibria with integer input / tolerances
    @test check_stability([0, 0, 1], params) isa Bool
    eqs, stab = find_equilibria(payoffs; num_initial_guesses = 50, stability_tol = 1//1000000)
    @test length(eqs) == length(stab)
end

@testset "API surface tests" begin
    coord_payoff_1(x, t, params) = x[1]
    coord_payoff_2(x, t, params) = x[2]
    coord_payoff_3(x, t, params) = x[3]
    payoffs = (coord_payoff_1, coord_payoff_2, coord_payoff_3)

    # find_equilibria: keyword form and the deprecated positional form agree
    eqs_kw, stab_kw = find_equilibria(payoffs; num_initial_guesses = 20)
    eqs_pos, stab_pos = find_equilibria(payoffs, NamedTuple(); num_initial_guesses = 20)
    @test length(eqs_kw) == length(stab_kw)
    @test length(eqs_pos) == length(stab_pos)

    # extra_params reaches the payoff functions through the keyword
    scaled_1(x, t, params) = params.a * x[1]
    scaled_rest(x, t, params) = 0.0
    eqs_extra, _ = find_equilibria((scaled_1, scaled_rest, scaled_rest);
        extra_params = (a = 2.0,), num_initial_guesses = 20)
    @test !isempty(eqs_extra)

    # generate_simplex_grid: keyword and positional margin agree
    @test generate_simplex_grid(10; margin = 2) == generate_simplex_grid(10, 2)
    @test generate_simplex_grid(10) == generate_simplex_grid(10; margin = 1)

    # equilibria_kwargs is forwarded to find_equilibria
    p = plot_evolution(payoffs, [[0.5, 0.3, 0.2]], (0.0, 10.0);
        equilibria_kwargs = (num_initial_guesses = 20, stability_tol = 1e-8))
    @test p isa Plots.Plot

    # contour_levels was removed
    @test_throws MethodError plot_evolution(payoffs, [[0.5, 0.3, 0.2]], (0.0, 10.0);
        contour_levels = 10)
end

@testset "replicator_dynamics tests" begin
    # Constant payoff functions that don't depend on strategy or time
    payoff1(x, t, params) = 1.0
    payoff2(x, t, params) = 0.5
    payoff3(x, t, params) = 0.2
    
    params = ReplicatorParams((payoff1, payoff2, payoff3), NamedTuple())
    
    x0 = [0.3, 0.3, 0.4]
    dx = zeros(3)
    
    replicator_dynamics!(dx, x0, params, 0.0)
    
    @test length(dx) == 3
    @test isapprox(sum(dx), 0.0; atol=1e-15)  # The sum of changes should be zero
end

@testset "replicator_dynamics expected values tests" begin
    payoff1(x, t, params) = 1.0
    payoff2(x, t, params) = 0.5
    payoff3(x, t, params) = 0.2
    
    params = ReplicatorParams((payoff1, payoff2, payoff3), NamedTuple())

    x0 = [0.3, 0.3, 0.4]
    dx = zeros(3)
    replicator_dynamics!(dx, x0, params, 0.0)
    
    # Expected average payoff
    x_norm = x0 / sum(x0)
    w = [1.0, 0.5, 0.2]
    avg_payoff = dot(x_norm, w)
    
    # Expected dx
    expected_dx = x_norm .* (w .- avg_payoff)
    
    @test isapprox(dx, expected_dx; atol=1e-15)
end

@testset "replicator_dynamics variable payoff tests" begin
    payoff1(x, t, params) = x[2]
    payoff2(x, t, params) = x[3]
    payoff3(x, t, params) = x[1]

    params = ReplicatorParams((payoff1, payoff2, payoff3), NamedTuple())

    x0 = [0.4, 0.4, 0.2]
    dx = zeros(3)
    replicator_dynamics!(dx, x0, params, 0.0)
    
    # Compute expected payoffs
    x_norm = x0 / sum(x0)
    w = [x_norm[2], x_norm[3], x_norm[1]]
    avg_payoff = dot(x_norm, w)
    expected_dx = x_norm .* (w .- avg_payoff)
    
    @test isapprox(dx, expected_dx; atol=1e-15)
end

@testset "find_equilibria tests" begin
    # Simple payoff functions
    payoff1(x, t, params) = 1.0
    payoff2(x, t, params) = 0.5
    payoff3(x, t, params) = 0.2
    
    equilibria, stability = find_equilibria((payoff1, payoff2, payoff3), NamedTuple(), num_initial_guesses=1000)
    
    @test length(equilibria) > 0  # Should find at least one equilibrium
    @test length(equilibria) == length(stability)  # Stability should match equilibria
end

@testset "find_equilibria expected values test" begin
    payoff1(x, t, params) = 1.0
    payoff2(x, t, params) = 0.5
    payoff3(x, t, params) = 0.2

    equilibria, stability = find_equilibria((payoff1, payoff2, payoff3), NamedTuple(), num_initial_guesses=1000)
    expected_eq = [1.0, 0.0, 0.0]
    found = false
    for (i, x_eq) in enumerate(equilibria)
        if isapprox(x_eq, expected_eq; atol=1e-2)
            found = true
            @test stability[i] == true
        end
    end
    @test found == true
end

@testset "find_equilibria multiple equilibria test" begin
    # Payoff functions that lead to multiple equilibria
    payoff1(x, t, params) = x[1]
    payoff2(x, t, params) = x[2]
    payoff3(x, t, params) = x[3]

    equilibria, stability = find_equilibria((payoff1, payoff2, payoff3), NamedTuple(), num_initial_guesses=1000)
    
    @test length(equilibria) ≥ 3  # Should find at least three equilibria
end

@testset "plot_simplex tests" begin
    @test plot_simplex(labels=["A", "B", "C"]) isa Plots.Plot
end

@testset "plot_simplex invalid labels tests" begin
    @test_throws BoundsError plot_simplex(labels=["A", "B"])  # Insufficient labels
    @test_throws BoundsError plot_simplex(labels=[])          # No labels
end

@testset "plot_simplex additional kwargs tests" begin
    p = plot_simplex(labels=["A", "B", "C"], fillcolor=:red, linecolor=:blue)
    @test p isa Plots.Plot
end

#=
@testset "check_stability function tests" begin
    x_eq = [1.0, 0.0, 0.0]
    is_stable = check_stability(x_eq, params)
    @test is_stable == true  # Assuming the equilibrium is stable in the given game
end
=#

@testset "identify_neutral_edges tests" begin
    # Edge indices are 1 = {1,2}, 2 = {1,3}, 3 = {2,3}. Each game below makes exactly
    # one pair of payoffs equal along its shared edge and unequal along the others.
    nofix(x, t, params) = 5.0

    # w1 == w2 along the x3 = 0 edge
    a1(x, t, params) = x[1]
    a2(x, t, params) = x[1]
    @test identify_neutral_edges(ReplicatorParams((a1, a2, nofix), NamedTuple())) == [1]

    # w1 == w3 along the x2 = 0 edge
    b1(x, t, params) = x[1]
    b3(x, t, params) = x[1]
    @test identify_neutral_edges(ReplicatorParams((b1, nofix, b3), NamedTuple())) == [2]

    # w2 == w3 along the x1 = 0 edge
    c2(x, t, params) = x[2]
    c3(x, t, params) = x[2]
    @test identify_neutral_edges(ReplicatorParams((nofix, c2, c3), NamedTuple())) == [3]

    # All three payoffs identical => every edge is neutral
    flat(x, t, params) = 1.0
    @test identify_neutral_edges(ReplicatorParams((flat, flat, flat), NamedTuple())) == [1, 2, 3]

    # Rock-Paper-Scissors has no neutral edge
    rps_R(x, t, params) = -x[2] + x[3]
    rps_P(x, t, params) = x[1] - x[3]
    rps_S(x, t, params) = -x[1] + x[2]
    @test isempty(identify_neutral_edges(ReplicatorParams((rps_R, rps_P, rps_S), NamedTuple())))

    # tol controls how close payoffs must be to count as neutral
    near1(x, t, params) = x[1]
    near2(x, t, params) = x[1] + 1e-4
    near_params = ReplicatorParams((near1, near2, nofix), NamedTuple())
    @test isempty(identify_neutral_edges(near_params))              # default tol = 1e-8
    @test identify_neutral_edges(near_params; tol = 1e-2) == [1]
end

@testset "plot_simplex neutral edge tests" begin
    @test plot_simplex(neutral_edges = [1, 2, 3], plot_neutral_dots = true) isa Plots.Plot
    @test plot_simplex(neutral_edges = [2], plot_neutral_dots = false) isa Plots.Plot
    @test plot_simplex(neutral_edges = Int[]) isa Plots.Plot
end

@testset "generate_simplex_grid tests" begin
    grid_points = generate_simplex_grid(10)
    @test all(sum.(grid_points) .≈ 1.0)
    @test all(all.(x .>= 0 for x in grid_points))
    @test length(grid_points) > 0
end

#Stability tests with classic games

@testset "Prisoner's Dilemma tests" begin
    # Payoff for Cooperation (C)
    function pd_payoff_C(x, t, params)
        C = x[1]
        D = x[2]
        return 3 * C + 0 * D
    end

    # Payoff for Defection (D)
    function pd_payoff_D(x, t, params)
        C = x[1]
        D = x[2]
        return 5 * C + 1 * D
    end

    # Dummy strategy with zero payoff
    function pd_payoff_dummy(x, t, params)
        return 0.0
    end

    # Add the dummy strategy in the tuple
    params = ReplicatorParams((pd_payoff_C, pd_payoff_D, pd_payoff_dummy), NamedTuple())
    
    # Test equilibrium finding and stability
    equilibria, stability_status = find_equilibria((pd_payoff_C, pd_payoff_D, pd_payoff_dummy), NamedTuple(), num_initial_guesses=100)
    
    # Expected equilibrium at [0.0, 1.0, 0.0]
    expected_eq = [0.0, 1.0, 0.0]
    found = false
    for (i, x_eq) in enumerate(equilibria)
        is_stable = stability_status[i]
        println("Equilibrium: $x_eq, Stable: $is_stable")
        if isapprox(x_eq, expected_eq; atol=1e-2)
            found = true
            @test is_stable == true
        else
            @test is_stable == false
        end
    end
    @test found == true
end

@testset "Hawk-Dove Game tests" begin
    # Payoff for Hawk (H)
    function hd_payoff_H(x, t, params)
        H = x[1]
        D = x[2]
        return 0 * H + 7 * D
    end

    # Payoff for Dove (D)
    function hd_payoff_D(x, t, params)
        H = x[1]
        D = x[2]
        return 1 * H + 3 * D
    end

    # Dummy strategy with zero payoff
    function hd_payoff_dummy(x, t, params)
        return 0.0
    end

    # Add the dummy strategy in the tuple
    params = ReplicatorParams((hd_payoff_H, hd_payoff_D, hd_payoff_dummy), NamedTuple())
    
    # Test equilibrium finding and stability
    equilibria, stability_status = find_equilibria((hd_payoff_H, hd_payoff_D, hd_payoff_dummy), NamedTuple(), num_initial_guesses=100)
    
    # Expected mixed equilibrium at [0.8, 0.2, 0.0]
    expected_eq = [0.8, 0.2, 0.0]
    found = false
    for (i, x_eq) in enumerate(equilibria)
        is_stable = stability_status[i]
        println("Equilibrium: $x_eq, Stable: $is_stable")
        if isapprox(x_eq, expected_eq; atol=1e-2)
            found = true
            @test is_stable == true
        else
            @test is_stable == false
        end
    end
    @test found == true
end

@testset "Rock-Paper-Scissors tests" begin
    # Payoff for Rock (R)
    function rps_payoff_R(x, t, params)
        R = x[1]
        P = x[2]
        S = x[3]
        return 0 * R + (-1) * P + 1 * S
    end

    # Payoff for Paper (P)
    function rps_payoff_P(x, t, params)
        R = x[1]
        P = x[2]
        S = x[3]
        return 1 * R + 0 * P + (-1) * S
    end

    # Payoff for Scissors (S)
    function rps_payoff_S(x, t, params)
        R = x[1]
        P = x[2]
        S = x[3]
        return (-1) * R + 1 * P + 0 * S
    end

    params = ReplicatorParams((rps_payoff_R, rps_payoff_P, rps_payoff_S), NamedTuple())
    
    # Test equilibrium finding and stability
    equilibria, stability_status = find_equilibria((rps_payoff_R, rps_payoff_P, rps_payoff_S), NamedTuple(), num_initial_guesses=100)
    
    # Expected mixed equilibrium at [1/3, 1/3, 1/3]
    expected_eq = [1/3, 1/3, 1/3]
    found = false
    for (i, x_eq) in enumerate(equilibria)
        is_stable = stability_status[i]
        println("Equilibrium: $x_eq, Stable: $is_stable")
        if isapprox(x_eq, expected_eq; atol=1e-2)
            found = true
            @test !is_stable  # Should be neutrally stable
        else
            @test !is_stable  # Other equilibria are also unstable
        end
    end
    @test found == true
end

@testset "Public Goods Game tests" begin
    # Payoff for Contribute (C)
    function pg_payoff_C(x, t, params)
        C = x[1]
        F = x[2]
        return 3 * C + 0 * F
    end

    # Payoff for Free Ride (F)
    function pg_payoff_F(x, t, params)
        C = x[1]
        F = x[2]
        return 4 * C + 1 * F
    end

    function pg_payoff_dummy(x, t, params)
        return 0.0
    end

    params = ReplicatorParams((pg_payoff_C, pg_payoff_F, pg_payoff_dummy), NamedTuple())
    
    # Test equilibrium finding and stability
    equilibria, stability_status = find_equilibria((pg_payoff_C, pg_payoff_F, pg_payoff_dummy), NamedTuple(), num_initial_guesses=100)
    
    # Expected equilibrium at [0.0, 1.0, 0.0]
    expected_eq = [0.0, 1.0, 0.0]
    found = false
    for (i, x_eq) in enumerate(equilibria)
        is_stable = stability_status[i]
        println("Equilibrium: $x_eq, Stable: $is_stable")
        if isapprox(x_eq, expected_eq; atol=1e-2)
            found = true
            @test is_stable == true
        else
            @test is_stable == false
        end
    end
    @test found == true
end







