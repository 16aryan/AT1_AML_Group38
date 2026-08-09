# Data Dictionary & Feature Reference

This reference document outlines the features present in the raw, interim, and processed datasets for predicting NBA career longevity (playing at least 3 seasons).

---

## 1. Core Cleaned Features
* **`ht_inches`**: Player's height converted from feet-and-inches string representation (e.g., `"6-8"`) to a continuous float (inches).
* **`yr_clean`**: Player's college academic year standardized to an ordinal scale:
  * `1` = Freshman (Fr)
  * `2` = Sophomore (So)
  * `3` = Junior (Jr)
  * `4` = Senior (Sr)
* **`age`**: Estimated age of the player during the season, calculated by subtracting the Year of Birth (from `dob`) from the basketball season calendar `year`.

---

## 2. Advanced Basketball Efficiency Features
* **`ast_tov_ratio_clean`**: Calculated assist-to-turnover ratio, with turnovers estimated as `ast / (ast/tov)` and zero-turnover cases safely imputed to avoid division errors.
* **`true_shooting_per_clean`**: Overall shooting efficiency, taking into account 2-pointers, 3-pointers, and free throws:
  $$\text{True Shooting } \% = \frac{\text{Points}}{2 \times (\text{Field Goal Attempts} + 0.44 \times \text{Free Throw Attempts})}$$
* **`defensive_impact_score`**: Weighted index capturing defensive presence:
  $$\text{Defensive Impact} = (1.5 \times \text{steals}) + (1.5 \times \text{blocks}) + (1.0 \times \text{defensive rebounds})$$
* **`scoring_efficiency_score`**: Overall scoring rate relative to played time percentage.
* **`usage_ortg_interaction`**: Interaction feature multiplying player usage rate (`usg`) by offensive rating (`ORtg`) to capture high-volume, high-efficiency offensive performance.

---

## 3. Cumulative Player Career Features
These features capture the player's development trajectory by calculating historical running aggregates grouped by player ID (`pid`) and sorted chronologically:
* **`career_season_num`**: Ordinal number of the season in the player's tracked college career (e.g., `1` for freshman year, `2` for sophomore year).
* **`cum_GP` / `cum_pts` / `cum_ast` / `cum_stl` / `cum_blk` / `cum_treb` / `cum_stops`**: Cumulative total games played, points, assists, steals, blocks, rebounds, and defensive stops accumulated up to that season.
* **`cum_mean_bpm` / `cum_mean_porpag` / `cum_mean_dporpag`**: Running career averages for Box Plus-Minus (BPM), Points Over Replacement Player (porpag), and Defensive Points Over Replacement Player (dporpag).
* **`max_pts_to_date` / `max_bpm_to_date`**: The maximum points and Box Plus-Minus score recorded by the player up to that season.
* **`pts_progression` / `bpm_progression` / `porpag_progression`**: Year-over-year difference (current season minus previous season) to capture development velocity.

---

## 4. Recruiting & Conference Strength Features
* **`has_rec_rank`**: Binary flag indicating if the player was ranked as a high school recruit.
* **`rec_rank_filled`**: Recruiting rank, with unranked players set to a default value of `150.0`.
* **`is_major_conf`**: Binary indicator showing if the player's college belongs to a major basketball power conference (`ACC`, `SEC`, `B10`, `B12`, `BE`, `P10`, `P12`) loaded from `data/external/major_conferences.csv`.
* **`pts_per_min` / `ast_per_min` / `stl_per_min` / `blk_per_min` / `treb_per_min`**: Direct court productivity stats divided by total minutes played (`mp`).
