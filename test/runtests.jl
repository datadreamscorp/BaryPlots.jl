using Test
using Plots, BaryPlots

@testset "ternary_coords tests" begin
    @test ternary_coords([1.0, 0.0, 0.0]) == (1.0, 0.0)  # Testing corner
    @test ternary_coords([0.0, 1.0, 0.0]) == (0.0, 0.0)  # Testing corner
    @test ternary_coords([0.0, 0.0, 1.0]) == (0.5, sqrt(3)/2)  # Top corner
    @test ternary_coords([0.5, 0.5, 0.0]) == (0.5, 0.0)  # Testing edge
    @test ternary_coords([0.0, 0.5, 0.5]) == (1/4, sqrt(3)/4)  # Testing edge
    @test ternary_coords([0.5, 0.0, 0.5]) == (3/4, sqrt(3)/4)  # Testing edge
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

@testset "find_equilibria tests" begin
    # Simple payoff functions
    payoff1(x, t, params) = 1.0
    payoff2(x, t, params) = 0.5
    payoff3(x, t, params) = 0.2
    
    equilibria, stability = find_equilibria((payoff1, payoff2, payoff3), NamedTuple(), num_initial_guesses=100)
    
    @test length(equilibria) > 0  # Should find at least one equilibrium
    @test length(equilibria) == length(stability)  # Stability should match equilibria
end

@testset "plot_simplex tests" begin
    @test plot_simplex(labels=["A", "B", "C"]) isa Plots.Plot
end

#Stability tests with classic games

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

@testset "Prisoner's Dilemma tests" begin
    # Add the dummy strategy in the tuple
    params = ReplicatorParams((pd_payoff_C, pd_payoff_D, pd_payoff_dummy), NamedTuple())
    
    # Test equilibrium finding and stability
    equilibria, stability_status = find_equilibria((pd_payoff_C, pd_payoff_D, pd_payoff_dummy), NamedTuple(), num_initial_guesses=100)
    
    for (i, x_eq) in enumerate(equilibria)
        is_stable = stability_status[i]
        println("Equilibrium: $x_eq, Stable: $is_stable")
        
        # Expected behavior:
        # Mutual defection (D,D) should be stable.
        # Mutual cooperation (C,C) or any mixed equilibria should be unstable.
        if isapprox(x_eq, [0.0, 1.0, 0.0]; atol=1e-2)  # Approximation for (D,D)
            @test is_stable == true  # Mutual defection is expected to be stable
        else
            @test is_stable == false  # Other equilibria should be unstable
        end
    end
end


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

@testset "Hawk-Dove Game tests" begin
    # Add the dummy strategy in the tuple
    params = ReplicatorParams((hd_payoff_H, hd_payoff_D, hd_payoff_dummy), NamedTuple())
    
    # Test equilibrium finding and stability
    equilibria, stability_status = find_equilibria((hd_payoff_H, hd_payoff_D, hd_payoff_dummy), NamedTuple(), num_initial_guesses=100)
    
    for (i, x_eq) in enumerate(equilibria)
        is_stable = stability_status[i]
        println("Equilibrium: $x_eq, Stable: $is_stable")
        
        # Expected behavior:
        # The mixed strategy equilibrium should be stable.
        # Pure Hawk or Pure Dove equilibria should be unstable.
        
        # Example: A mixed strategy equilibrium typically has both Hawks and Doves (but not all in one strategy).
        if isapprox(x_eq[1], 0.8, atol=1e-2) && isapprox(x_eq[2], 0.2, atol=1e-2)
            @test is_stable == true  # The mixed strategy should be stable
        else
            @test is_stable == false  # Pure strategies should be unstable
        end
    end
end


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

@testset "Rock-Paper-Scissors tests" begin
    params = ReplicatorParams((rps_payoff_R, rps_payoff_P, rps_payoff_S), NamedTuple())
    
    # Test equilibrium finding and stability
    equilibria, stability_status = find_equilibria((rps_payoff_R, rps_payoff_P, rps_payoff_S), NamedTuple(), num_initial_guesses=100)
    
    for (i, x_eq) in enumerate(equilibria)
        is_stable = stability_status[i]
        println("Equilibrium: $x_eq, Stable: $is_stable")
        @test !is_stable  # Rock-paper-scissors has no stable equilibrium
    end
end


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

@testset "Public Goods Game tests" begin
    params = ReplicatorParams((pg_payoff_C, pg_payoff_F, pg_payoff_dummy), NamedTuple())
    
    # Test equilibrium finding and stability
    equilibria, stability_status = find_equilibria((pg_payoff_C, pg_payoff_F, pg_payoff_dummy), NamedTuple(), num_initial_guesses=100)
    
    for (i, x_eq) in enumerate(equilibria)
        is_stable = stability_status[i]
        println("Equilibrium: $x_eq, Stable: $is_stable")
        if isapprox(x_eq[1], 0.0, atol=1e-2) && isapprox(x_eq[2], 1.0, atol=1e-2)
            @test is_stable == true  # The mixed strategy should be stable
        else
            @test is_stable == false  # Pure strategies should be unstable
        end
    end
end







