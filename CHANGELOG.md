# Changelog

All notable changes to BaryPlots.jl are documented here.

## v0.4.0

A breaking release. It adds basin-of-attraction plotting, replaces the heavyweight
DifferentialEquations dependency with the ODE solver the package actually uses, and tidies
the `plot_evolution` keyword surface.

### Breaking

- **`plot_evolution` no longer takes `num_initial_guesses`, `solver_tol`, `equilibrium_tol`,
  `validity_tol` or `stability_tol` directly.** These five were pure pass-throughs to
  `find_equilibria` and are now bundled into a single `equilibria_kwargs` NamedTuple:

  ```julia
  # before
  plot_evolution(payoffs, x0_list, tspan; solver_tol = 1e-12, equilibrium_tol = 1e-3)

  # after
  plot_evolution(payoffs, x0_list, tspan;
      equilibria_kwargs = (solver_tol = 1e-12, equilibrium_tol = 1e-3))
  ```

- **`contour_levels` removed from `plot_evolution`.** It was documented as "number of contour
  levels to display" but was never read anywhere in the package — the contour is drawn with
  `scatter` and `zcolor`, which has no notion of levels. Passing it now raises a `MethodError`
  rather than being silently ignored.

- **`markerstrokewidth` default changed from `2` to `1.0`**, giving thinner equilibrium and
  neutral-edge markers by default. Pass `markerstrokewidth = 2` to restore the old look.

- **Dependency change: `DifferentialEquations` → `OrdinaryDiffEq` + `SciMLBase`.** The package
  used exactly four symbols (`ODEProblem`, `solve`, `DiscreteCallback`, `terminate!`) while
  depending on the meta-package that pulls in the stochastic, DAE, jump-process and
  boundary-value solver stacks. This drops 43 transitive dependencies and cuts the test
  suite's wall time from about 7 minutes to under 2.

- **Minimum Julia version raised from 1.6 to 1.10**, matching what current OrdinaryDiffEq and
  SciMLBase require. The old `julia = "^1.6"` bound had become a claim the dependency tree
  could not honour.

### Added

- **Basins of attraction.** `plot_evolution(...; plot_basins = true)` classifies every point of
  the simplex by which stable equilibrium it flows to, shades the resulting basins and draws the
  separatrices between them. Controlled by `basin_resolution`, `basin_convergence_tol`,
  `basin_shade`, `basin_colors`, `basin_alpha`, `basin_boundary_color`,
  `basin_boundary_linewidth` and `basin_boundary_linestyle`.

  Works with a single stable equilibrium — the separatrix then divides its basin from the region
  flowing to unstable boundary points. Games with no stable equilibrium (Rock-Paper-Scissors,
  say) skip basin plotting with a warning. Cost is quadratic in `basin_resolution`: roughly
  4,300 ODE solves at `100`, and 108,000 at `500`.

- **`inverse_ternary_coords(X, Y)` is now exported**, alongside its existing counterpart
  `ternary_coords`. It maps a point in plot coordinates back to barycentric coordinates;
  points outside the triangle come back with at least one negative component.

- **Numeric arguments accept any real type that converts unambiguously to a float.** Scalar
  parameters take `Real`, counts take `Integer`, and vector arguments take
  `AbstractVector{<:Real}`, so `markersize = 10`, `tspan = (0, 100)` and
  `steady_state_tol = 1//1000000` all work. Previously several of these were pinned to
  `Float64` or `Int` and rejected otherwise-valid input.

### Changed (non-breaking)

- **`find_equilibria` takes its payoff parameters as the `extra_params` keyword** rather than a
  second positional argument, matching `plot_evolution`'s naming. The positional form
  `find_equilibria(payoffs, params)` still works and forwards to the keyword form.

- **`generate_simplex_grid` takes `margin` as a keyword.** The positional form
  `generate_simplex_grid(resolution, margin)` still works.

### Fixed

- **`classify_basin` now normalises its initial condition onto the simplex.** Because the
  solver's `isoutofdomain` check rejects any component above 1, an unnormalised initial
  condition could never integrate and was silently reported as "outside all basins".

- **Solver failures are now detected properly.** `classify_basin` checks
  `successful_retcode` before using the trajectory endpoint. Previously a solve that failed
  without throwing — instability, dt underflow, max iterations — was classified on whatever
  state it happened to stall in.

- **Genuine errors are no longer swallowed.** `classify_basin` wrapped its `solve` in a bare
  `catch` that reported *any* exception as an unclassified point. It now tolerates
  `DomainError` (a numerical blow-up in user payoff functions, where continuing the sweep is
  the right call) and rethrows everything else, so bugs surface as stack traces instead of
  plausible-looking but wrong pictures.

- **Docstring defaults corrected.** `basin_resolution` was documented as `60` but defaults to
  `100`; `basin_shade` as `true` but defaults to `false`; `basin_boundary_linewidth` as `1` but
  defaults to `2`. The `plot_evolution` and `plot_simplex` docstring signature blocks had also
  drifted out of sync with the real signatures.

- **README fix:** the parameter list documented a `cbar` option; the actual keyword is
  `colorbar`.

- Removed a dead conditional in `plot_evolution` whose two branches were identical.

### Internal

- `Test` moved from `[deps]` to `[extras]`/`[targets]`, the standard layout for a package's
  test-only dependency.
- `Manifest.toml` is no longer tracked. BaryPlots is a library: consumers resolve their own
  environments from `Project.toml`, so a committed lockfile was never read by them and only
  went stale.
- Test suite expanded from 17 to 28 testsets, covering the coordinate helpers, basin
  classification, the error-handling contract, numeric-type flexibility and the retained
  positional call forms.

## v0.3.1 and earlier

See the commit history.
