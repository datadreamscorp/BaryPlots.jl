### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ c4a80484-f794-11ef-021d-0b9b152128e7
begin
	using Pkg
	Pkg.activate(".")
	using BaryPlots, LaTeXStrings
end

# ╔═╡ 7686cb67-73b4-418d-b208-bd35802bc791
md"""
### Example 1: Hawk-Dove Game
"""

# ╔═╡ b7565f12-c41f-449d-9302-f3cd1ed56467
begin
	# Payoff functions for Hawk-Dove game
	function hd_payoff_H(x, t, params)
	    H, D = x[1], x[2]
	    return (-1) * H + 4 * D  # Payoff for Hawk
	end
	
	function hd_payoff_D(x, t, params)
	    H, D = x[1], x[2]
	    return 0 * H + 2 * D  # Payoff for Dove
	end
	
	function hd_payoff_dummy(x, t, params)
	    return 0.0  # Dummy payoff function for the third strategy
	end
	
	payoff_functions = (
	    hd_payoff_H, 
	    hd_payoff_D, 
	    hd_payoff_dummy
	    )
	
	initial_conditions_HD = [
	        [0.99, 0.01, 0.0], 
	        [0.01, 0.99, 0.0]
	        ]# Two starting points
	
	# Simulate and plot
	p = plot_evolution(
	    payoff_functions,
	    initial_conditions_HD,
	    (0.0, 100.0);
	    labels = [L"\mathrm{Hawk}", L"\mathrm{Dove}", L"\mathrm{Loner}"],
	    arrow_list = [ [300], [300] ],
	    contour = true,              # Include contour plot
	)
end

# ╔═╡ 6c53a314-8560-4b66-9101-d5077949f9b6
md"""
### Example 2: Rock-Paper-Scissors
"""

# ╔═╡ d7a877bc-a0e3-42d6-812e-020d9b37647f
begin
	# Payoff for Rock (R)
	function rps_payoff_R(x, t, params)
	    R, P, S = x[1], x[2], x[3]
	    return 0 * R + 1 * S - 1 * P  # Rock beats Scissors
	end
	
	# Payoff for Paper (P)
	function rps_payoff_P(x, t, params)
	    R, P, S = x[1], x[2], x[3]
	    return 0 * P + 1 * R - 1 * S # Paper beats Rock
	end
	
	# Payoff for Scissors (S)
	function rps_payoff_S(x, t, params)
	    R, P, S = x[1], x[2], x[3]
	    return 0 * S + 1 * P - 1 * R  # Scissors beat Paper
	end
	
	# Initial conditions
	initial_conditions_RPS = [
	    [0.6, 0.3, 0.1], 
	    [0.2, 0.4, 0.4]
	    ]
		
	# Simulate and plot
	p2 = plot_evolution(
	    (rps_payoff_R, rps_payoff_P, rps_payoff_S),
	    initial_conditions_RPS,
	    (0.0, 100.0),
	    labels=[L"\mathrm{Rock}", L"\mathrm{Paper}", L"\mathrm{Scissors}"],
	    arrow_list = [ [100], [100] ],
	    colored_trajectories = true
	)

end

# ╔═╡ f0e4e9f5-1580-476e-a5e8-bc902d52c0e6
md"""
### Example 3: Public Good Game
"""

# ╔═╡ 218a13ac-f211-445c-8f02-476a1a11ead6
begin
	# Define the payoffs for Contribute (C) and Free Ride (F)
	function pg_payoff_C(x, t, params)
	    b = params.b
	    C = x[1]
	    F = x[2]
	    return b * C + 0 * F
	end
	
	function pg_payoff_F(x, t, params)
	    b = params.b
	    C = x[1]
	    F = x[2]
	    return (b/2) * C + 1 * F
	end
	
	function pg_payoff_dummy(x, t, params)
	    return 0.0
	end
	
	# Set up the parameterized game with benefit multiplier b = 2.0
	params = (b = 2.0, )
	payoff_functions_PG = (pg_payoff_C, pg_payoff_F, pg_payoff_dummy)
	
	# Initial conditions for different starting frequencies of strategies
	initial_conditions_PG = [
	    [0.51, 0.49, 0.0],
	    [0.49, 0.51, 0.0],
		[0.1, 0.1, 0.80]
	]
	
	# Simulate and plot the dynamics
	plot_evolution(
	    payoff_functions_PG,
	    initial_conditions_PG,
	    (0.0, 100.0);
	    labels = [L"\mathrm{Contribute}", L"\mathrm{Free\ Ride}", L"\mathrm{Loner}"],
	    extra_params = params,
	    colored_trajectories = true,
	    arrow_list = [ [500, 1000], [500, 1000], [200] ],
	    equilibrium_tol = 1e-3
	)
end

# ╔═╡ Cell order:
# ╟─c4a80484-f794-11ef-021d-0b9b152128e7
# ╟─7686cb67-73b4-418d-b208-bd35802bc791
# ╟─b7565f12-c41f-449d-9302-f3cd1ed56467
# ╟─6c53a314-8560-4b66-9101-d5077949f9b6
# ╟─d7a877bc-a0e3-42d6-812e-020d9b37647f
# ╟─f0e4e9f5-1580-476e-a5e8-bc902d52c0e6
# ╟─218a13ac-f211-445c-8f02-476a1a11ead6
