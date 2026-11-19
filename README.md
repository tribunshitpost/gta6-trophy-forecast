# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tribunshitpost/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tribunshitpost/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles per trophy, from the house's reads of the trophy page up to the deadline. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## The model

An earn rate is a ratio, `rate = E / N`, and its two halves move for different reasons in launch week. The tracked-player count `N` explodes after launch (638 on launch night, 3,868 at the first refresh, ~10,000 by that afternoon, from the live Exophase page) and then its growth decays, diluting every rate. The earners `E` grow only as players actually reach each trophy, on that trophy's own timescale: an early fishing challenge within hours, a story chapter over days, a 250-stunt grind over weeks. Extrapolating the rate itself (the house trend baseline) conflates the two, so it overshoots fast-rising trophies and misjudges every trophy when the denominator's growth bends. This program forecasts `N` and `E` separately and divides at the end:

- **Joining.** The join rate `J(t) = dN/dt` is fitted as a power law in time since release on the last 48 hours of distinct reads; `N` at the target is the last count plus the integral of `J`.
- **Earning.** A player who joined at hour `u` earns trophy `i` after a Weibull delay (shape pooled across trophies, scale fitted per trophy on a grid) with eventual share `p_i` solved by least squares. `lambda_i` is shrunk toward trophies with a similar last rate, weighted by observed earners.
- **Spread.** Quantiles sit on the logit scale around the median with a sigma combining a base term, the trophy's fit residual, uncertainty in `N` at the target, and a refresh-lag term (the answer is the page as of Exophase's last refresh, which can be hours old).
- **Frozen history.** If the reads never move (as in the practice cases), the median steps down a little and the spread widens, because the next refresh dilutes.

Validation: on 60 simulated launch-week worlds (cohort simulator calibrated to the real anchors above, spanning player-growth, progression-speed, and refresh-timing scenarios), mean pinball loss on the logit scale **0.155**, vs **0.257** for the house trend baseline and **0.372** for the persistence starter on the same worlds — beating the trend baseline in 54 of 60 worlds and persistence in all 60.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, no network, deterministic, and finishes in about 0.3 s a case (limit 8 s). It reads every `<id>.in` in the input directory and writes `<id>.out` (`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=24pn59hn&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
