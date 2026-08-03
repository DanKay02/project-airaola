# Project Airaola

## Data. Decisions. Domination.

Project Airaola is a statistics-driven autonomous Fantasy Premier League manager.

It is designed to:

- analyse player and fixture data
- build legal FPL squads
- project player points
- optimise transfers and captaincy
- plan across multiple Gameweeks
- explain every decision using measurable evidence
- generate automated statistical reports

## The Airaola Principle

> Every decision made by Project Airaola must be explainable through measurable evidence.

## Current Version

### v0.1.0: The Training Ground

The first release will:

- load player data
- validate FPL squad rules
- build a legal 15-player squad
- respect budget and club limits

## Manager Philosophy

- maximise total season points
- plan five Gameweeks ahead
- prioritise secure minutes
- avoid unnecessary transfer hits
- use moderate-risk captaincy
- preserve chips until statistically valuable

## Technology

- Python
- pandas
- OR-Tools
- pytest
- GitHub Actions

## Manager's Notes

### v0.1.0: The Training Ground

> The training ground is open.  
> Project Airaola now has a defined identity, development environment, version history and public home. Recruitment operations can begin.

### v0.1.1: Recruitment Department

> The recruitment department is open.  
> Project Airaola can now retrieve, validate and process the complete FPL player pool. Every registered player is assigned a club, position, price and performance profile, ready for squad evaluation.

### v0.1.2: Squad Registration

> The registration office is operational.  
> Project Airaola can now assemble and validate a complete 15-player squad while enforcing positional quotas, budget restrictions, unique-player selection and club limits.

### v0.1.3: First Team Selection

> The manager has named his first squad.  
> Project Airaola can now evaluate the full player pool and select the highest-scoring legal 15-player squad under all FPL budget, positional and club constraints.

### v0.1.4: The Projection Engine

> The analysis department has opened its doors.  
> Project Airaola now produces forward-looking player projections using recent form, points per game, points per 90, expected minutes and availability. Squad selection is now driven by projected performance rather than points already scored.

### v0.1.5: Fixture Intelligence

> The fixture analysts have joined the backroom staff.  
> Project Airaola can now inspect the live FPL calendar, map every club’s opponents and venues across the planning horizon, and automatically detect normal, blank, and double Gameweeks.

### v0.1.6: Fixture-Adjusted Projections

> The tactical analysis unit is operational.  
> Project Airaola now adjusts every player projection according to fixture difficulty, venue, and fixture count. Blank Gameweeks contribute no fixture projection, while Double Gameweeks are evaluated as multiple independent matches.

### v0.1.7: Position-Aware Minutes Security

> Project Airaola now models start security, positional involvement, sample confidence, and position-specific scoring routes. Goalkeepers face stricter eligibility rules, preventing backup players from being selected based on misleading small samples.

### v0.1.8: Matchday Selection

> Project Airaola can now convert its optimised 15-player squad into a legal starting XI, choose a formation, rank its substitutes, and assign captain and vice-captain duties using projected output and minutes security.

### v0.1.9: Single-Gameweek Decision Engine

> Project Airaola now separates long-term squad planning from weekly matchday decisions. The 15-player squad is optimised across a five-Gameweek horizon, while the starting XI, bench order, captain and vice-captain are selected using only the immediate Gameweek projection.

### v0.1.10: Captaincy-Aware Squad Optimisation

> Project Airaola now incorporates captaincy value directly into initial squad construction. The projection engine creates separate player forecasts for each Gameweek in the planning horizon, while the optimiser selects one projected captain per Gameweek and rewards squads containing strong rotating armband options.

> The selected squad records each player's projected captain Gameweeks and total captaincy appearances. Long-term squad strength remains part of the objective, but premium players can now justify their price through realistic captaincy utility rather than being assessed only on ordinary points and value.

### v0.1.11: Lineup-Aware Horizon Optimisation

> Project Airaola now optimises a legal starting XI for every Gameweek in the planning horizon alongside the 15-player squad. Weekly starter selections obey FPL formation requirements, while projected captains and vice-captains must be included in that Gameweek's starting XI.

> The optimisation objective now rewards projected points from weekly starters, captaincy bonuses and reduced bench-cover value. This prevents substitutes from receiving the same strategic value as active starters and produces a connected five-Gameweek squad, lineup and captaincy plan.

> The selected squad now records projected start Gameweeks, projected start counts, captain Gameweeks and vice-captain Gameweeks for inspection. Future vice-captain scoring remains a planned refinement because the current model treats vice-captaincy as a legal assignment rather than a weighted strategic contribution.

### v0.1.12: Single-Transfer Planner

> Project Airaola can now evaluate legal one-player transfer opportunities using the current projected squad and full player pool. The first transfer-planning model considers one player out and one player in while preserving FPL position requirements, the £100.0m budget and the maximum of three players per club.

> Transfer targets are filtered using availability and minutes security before being compared across the five-Gameweek planning horizon. Each legal move is assessed using its projected long-term gain, next-Gameweek gain, remaining money in the bank and target reliability.

> Airaola now avoids unnecessary transfers by requiring a projected improvement of at least 1.5 points across the planning horizon. When no legal move clears this threshold, the engine recommends holding the transfer rather than making a low-value change.

> Transfer recommendations now report the proposed player out, player in, transfer cost, projected gain, money remaining and recommendation strength. The current version assumes each player's present FPL price is also their selling price. Persistent purchase-price and official selling-value calculations remain planned future improvements.

### v0.1.13: Transfer-Bank Strategy Engine

> Project Airaola can now evaluate transfer plans ranging from zero to five moves in a single Gameweek. The engine understands that managers may bank up to five free transfers and can compare the value of acting immediately against preserving flexibility for a later squad rebuild.

> Each proposed strategy is assessed using its gross five-Gameweek projected gain, next-Gameweek improvement, transfer-bank opportunity cost and any points hits incurred beyond the available free-transfer allowance. Additional transfers beyond the current bank are charged at four points each.

> Airaola now distinguishes between executing transfers, rolling a transfer and holding at the five-transfer cap. When no available plan clears the minimum net strategic-gain threshold, the manager preserves the transfer bank rather than making a low-value move.

> Recommended transfer plans explicitly list every player to sell and buy, along with position, price, individual projected gain, hit cost, remaining money in the bank and expected free transfers for the following Gameweek.

> The reporting logic now preserves the best rejected plan by net strategic value. This ensures that ROLL and HOLD decisions display the same gross gain, strategic costs, hit cost and net result that were actually used by the decision engine.

> The current version uses manually supplied free-transfer availability and assumes current FPL prices are also selling prices. Persistent squad state, automatic transfer-bank tracking and official purchase-price selling-value calculations remain planned future upgrades.

### v0.1.14: Persistent Season State

> Project Airaola now maintains a persistent manager-state file across runs. The system stores the active 15-player squad, player purchase prices, money in the bank, free-transfer availability, chip availability and transfer-decision history.

> On first setup, Airaola detects an empty state file, generates an optimised initial squad and registers the selected players as the permanent starting team. Future runs reconstruct that saved squad using refreshed FPL data instead of rebuilding a new team from scratch.

> Transfer planning now uses the saved free-transfer bank and current persistent squad. Confirmed ROLL and HOLD decisions update the transfer bank while preserving the squad, and confirmed EXECUTE decisions update the owned players, purchase prices, bank and transfer history.

> Three run modes are now supported:
>
> - `python main.py` runs interactively and asks for confirmation before changing manager state.
> - `python main.py --dry-run` performs a complete analysis without writing any state changes.
> - `python main.py --auto-apply` applies and saves Airaola's recommendation automatically for unattended scheduled runs.

> State changes are written safely through a temporary file before replacing the active JSON file. This reduces the risk of corrupting manager state during an interrupted save.

> Reconstructed persistent squads now retain purchase-price information while using refreshed prices, projections, fixtures and availability data for weekly analysis.

> This release turns Project Airaola from a fresh weekly optimiser into a continuous season-long manager with memory.

### v0.1.15: Chip Strategy Engine

> Project Airaola now includes its first dedicated chip-strategy engine.

> The system evaluates the active half-season chip period and reads chip availability from persistent manager state. Chip data is now stored separately for the first and second halves of the season, allowing Airaola to track each available Wildcard, Free Hit, Bench Boost and Triple Captain independently.

> This release introduces live evaluation for:
>
> - `NO CHIP`
> - `TRIPLE CAPTAIN`
> - `BENCH BOOST`

> Triple Captain value is calculated from the selected captain's next-Gameweek projected points, minutes security and proximity to first-half chip expiry.

> Bench Boost value is calculated from the combined projected points of all four substitutes. The engine also checks minutes security across the complete bench and requires the substitute goalkeeper to meet the minimum reliability threshold.

> Each chip candidate now reports:
>
> - availability
> - eligibility
> - projected gain
> - adjusted strategic gain
> - execution threshold
> - recommendation strength
> - a written reason for the decision

> When no candidate clears its execution threshold, Airaola recommends `NO CHIP` and identifies the strongest rejected chip option.

> Persistent manager state now stores:
>
> - nested first-half and second-half chip availability
> - the last Free Hit Gameweek
> - confirmed chip decisions
> - projected and adjusted chip gains
> - captain and bench context
> - chip recommendation history

> The state loader remains backwards compatible with the previous flat chip format and automatically migrates old manager-state data into the new nested structure.

> Chip decisions respect all existing run modes:
>
> - `python main.py` asks for manual confirmation
> - `python main.py --dry-run` evaluates chips without changing state
> - `python main.py --auto-apply` saves the selected chip decision automatically

> Free Hit and Wildcard evaluation are intentionally deferred until temporary and permanent squad re-optimisation are integrated.

> This release gives Project Airaola its first true chip brain, allowing it to preserve chips when ordinary Gameweeks fail to justify their use.

### v0.1.16 | Price and Squad Value Engine

> Airaola now understands that a player’s market price is not always the amount available when selling them.

>The finance engine calculates official FPL selling prices using each player’s saved purchase price and current market price. Price drops are absorbed in full, while profits from price rises are shared according to FPL selling-value rules.

>Transfer planning now uses:

> - official selling prices for outgoing players
> - current market prices for incoming players
> - the manager’s saved money in the bank
> - accurate affordability across one-to-five-transfer plans
> - correct post-transfer bank calculations

>A new Finance Department report displays the squad’s original purchase value, current market value, official selling value, usable budget, retained profit and losses caused by price falls.

>The numbers now know the difference between looking rich and being able to spend it.