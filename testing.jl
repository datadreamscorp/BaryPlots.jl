### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 74a423ce-8ca9-11ef-23f8-b7c74b402092
begin
	using Pkg
	Pkg.activate(".")
	using Revise
	using Plots, Distributions, Statistics
	using BaryPlots
end

# ╔═╡ 1247a60f-f860-40af-93e3-65c51fdc7773
function sharer_payoff(x, t, params)
	(params.u2 - params.γ)*x[1] + params.u1*x[2] + ( (1 - params.w)*(params.u12 - params.γ) + params.w*params.u1 )*x[3] 
end

# ╔═╡ 2a16d826-5bdb-48cb-aef0-6e7091cdf11c
function loner_payoff(x, t, params)
	params.u1
end

# ╔═╡ 67ac2ef3-b75d-4ca2-a89b-dd796c9fd37a
function hoarder_payoff(x, t, params)
	( (1 - params.w)*(params.u32 - params.γ) + params.w*params.u1 )*x[1] + params.u1*x[2] + ( params.u1 - (1 - params.w)*params.γ )*x[3] 
end

# ╔═╡ c81a6992-c130-46db-87a0-f25b66dbbac6
begin
	Eu(n, f; m=n) = mean([f( (1/m)*sum(rand(n)) ) for _ in 1:10000000])
	
	Eu32(f) = mean(
		[
			f(rand() + rand()/2) 
			for _ in 1:10000000
		]
	)
end

# ╔═╡ 2ceed703-e44b-4b12-bb09-dc22fba79225
begin
	CARA(x; a=1) = 1 - exp(-a*x)
	CRRA(x; η=1) = η == 1 ? log(x) : ( (x)^(1 - η) ) / (1 - η)
end

# ╔═╡ 130a4a1c-f671-4bd0-ad52-fa51e6ef6f8a
f = CRRA

# ╔═╡ 79b33ae9-9ff8-429c-9839-94320d407925
params = (
	u1 = Eu(1, f),
	u2 = Eu(2, f),
	u12 = Eu(1, f, m=2),
	u32 = Eu32(f),
	γ = 0.01,
	w = 0.75
)

# ╔═╡ d3a0748c-020f-4d63-baf7-cbcd93276180
begin
	labels = ["Sharer", "Loner", "Hoarder"]
	payoff_functions = (sharer_payoff, loner_payoff, hoarder_payoff)
	tspan = (0.0, 10000.0)
end

# ╔═╡ 8363150f-249e-4637-9cdd-a9624e911f9f
plot(f, xlim=(1, 10), legend=false, lw=2)

# ╔═╡ 17c8cf6b-11f4-412d-83e4-b55e65f429eb
plot_evolution(
    payoff_functions,
    [
		[0.0, 0.01, 0.99],
    	[0.99, 0.01, 0.0],
		[0.99, 0.0, 0.01],
		[0.1, 0.05, 0.85]
	],
    tspan;
    labels = labels,
    extra_params = params,
	solver_tol = 1e-10,
	stability_tol = 1e-3,
	validity_tol = 1e-8,
	equilibrium_tol = 1e-2,
	contourf = true
)

# ╔═╡ 0d778e28-f271-4a49-abbf-75a12406fcd0
function trap(x; a=0, b=1)
	if x > (3*a/2) && x < a + b/2
		return (2/(b - a)^2)*(x - 3*a/2)
	elseif x > (a + b/2) && x < (b + a/2)
		return 1/(b - a)
	elseif x > (b + a/2) && x < (3*b/2)
		(2/(b-a)^2)*((3*b/2) - x) 
	else
		return 0
	end
end

# ╔═╡ 6f6a2638-4efa-4867-be2d-52cca33fdf3d
plot(trap, xlim=(-1, 2), legend=false, lw=2)

# ╔═╡ Cell order:
# ╠═74a423ce-8ca9-11ef-23f8-b7c74b402092
# ╠═1247a60f-f860-40af-93e3-65c51fdc7773
# ╠═2a16d826-5bdb-48cb-aef0-6e7091cdf11c
# ╠═67ac2ef3-b75d-4ca2-a89b-dd796c9fd37a
# ╠═c81a6992-c130-46db-87a0-f25b66dbbac6
# ╠═2ceed703-e44b-4b12-bb09-dc22fba79225
# ╠═130a4a1c-f671-4bd0-ad52-fa51e6ef6f8a
# ╠═79b33ae9-9ff8-429c-9839-94320d407925
# ╠═d3a0748c-020f-4d63-baf7-cbcd93276180
# ╠═8363150f-249e-4637-9cdd-a9624e911f9f
# ╠═17c8cf6b-11f4-412d-83e4-b55e65f429eb
# ╠═0d778e28-f271-4a49-abbf-75a12406fcd0
# ╠═6f6a2638-4efa-4867-be2d-52cca33fdf3d
