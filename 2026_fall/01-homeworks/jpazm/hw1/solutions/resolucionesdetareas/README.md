# HW3 — Coffee: demand, conduct, counterfactuals (solution)

## Run
```bash
python coffee_solution.py --data coffee.csv --out out/
```
No dependencies beyond `numpy`, `pandas`, `matplotlib` (2SLS is implemented by hand).

## Expected data schema (long, one row per country-year)
`year, country, price_y, qe_yc, importGDP_y, teaPrices_y, tmax_ye, tmin_ye, farm_prices_yc, fert_y`
— adjust the `COLS` dict at the top of the script if names differ. Leaders are set
in `LEADERS`; the cooperative bloc in `BLOC`.

## What it produces (in `out/`)
- `demand_estimates.csv`, `elasticities.png` — Section 2
- `marginal_costs.csv`, `mc_markup_{Brazil,Vietnam,Colombia,Indonesia}.png` — Section 3
- `table_CF{1,2,3,4}.csv`, `profit_changes.png` — Section 4

## Key modeling choices
- Supply uses **spec 2** linear demand (temperature-only IV): `b = -aQ`.
- **CF1 uses partial internalization `KAPPA = 0.5`, not a merger.** The bloc are
  followers in a homogeneous good, where full internalization (κ=1) makes
  `11'+ΩF` singular / the split indeterminate. κ=0.5 keeps all members interior;
  ΔΠ^PHM is the value of *partial* coordination. See the solution note.
- Stackelberg leaders differ from Cournot only via the fringe pass-through
  `τ = 1/(1+N_F)`; followers coincide with Cournot.
- Negative recovered MC is floored at the year's p25 of non-negative MC.
- **Non-negativity (shut-down) rule:** if the FOC solution gives a country q<0,
  that country produces nothing — set q=0, drop it, and re-solve the reduced
  game; iterate until all survivors have q>=0. Shut-down countries earn zero
  profit; the count is reported per CF. Because exit is a discrete margin, some
  comparative statics can be non-monotone (a price rise can pull a shut country
  back in) — read signs from the tables.
- `RECOVER_WITH` flag: `"observed"` (default; baseline may differ from data, per
  the HW's baseline-vs-observed framing) or `"fitted"` (baseline reproduces
  observed quantities exactly — used as the internal consistency check, which
  passes to ~1e-13).

## Validation (test_pipeline.py)
τ matches the closed form; IV recovers the true demand slope; recover→re-solve
round-trips to machine precision; the shut-down rule zeroes exactly the
unprofitable countries (all survivors q>=0, all shut-down c>=P, survivor re-solve
matches); κ=0.5 verified interior with a positive bloc gain while κ→1 degenerates.
