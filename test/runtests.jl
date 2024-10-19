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

#=
@testset "replicator_dynamics negative frequencies test" begin
    payoff1(x, t, params) = x[2]
    payoff2(x, t, params) = x[3]
    payoff3(x, t, params) = x[1]

    params = ReplicatorParams((payoff1, payoff2, payoff3), NamedTuple())

    x0 = [-0.1, 0.6, 0.5]
    dx = zeros(3)
    @test_throws DomainError replicator_dynamics!(dx, x0, params, 0.0)
end
=#

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
    @test_throws TypeError plot_simplex(labels=[])          # No labels
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







