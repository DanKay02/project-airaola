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