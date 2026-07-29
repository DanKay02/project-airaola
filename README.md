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