# Testing

Test

## Inactives data

The game-day pipeline now attempts to merge an **inactives** table into the
game bundle. Each record should include the following columns:

* `team` – team abbreviation
* `player_name` – full name of the player
* `status` – roster designation (e.g. `INACTIVE`)
* `reason` – optional description
* `game_id` – identifier matching the schedule table

`fetch_inactives(kickoff_ts)` currently returns an empty table as a placeholder
until a real data source is integrated. If the automated fetch fails on game
day, create a CSV with the columns above and load it manually before calling
`read_inactives`.
