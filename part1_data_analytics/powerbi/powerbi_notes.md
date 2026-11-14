# Power BI Traffic Intelligence Dashboard: Preparation Notes and Findings

**Author:** Kyi Htoo Aung | Part 1, Task 4 | Files: `traffic_dashboard.pbix`, `traffic_dashboard.pdf`, `Screenshot 2026-11-13 003316.png`, `Screenshot 2026-11-13 003340.png`, `Screenshot 2026-11-13 003457.png`

---

## 4.1 Data quality and preparation (Power Query Editor)

The raw file `Metro_Interstate_Traffic_Volume.csv` was imported with **Get data > Text/CSV** and prepared in Power Query Editor. The query was renamed **Traffic**. Column quality, distribution and profile were switched on with profiling based on the **entire data set** (screenshots `Screenshot 2026-11-13 003316.png` and `Screenshot 2026-11-13 003340.png` show column quality and distribution for the original and added columns; `Screenshot 2026-11-13 003457.png` shows the full Power Query window).

### Rows and columns

| Stage | Rows | Columns |
|---|---|---|
| Raw import | 48,204 | 9 |
| After removing duplicate rows | 48,187 | 9 |
| After adding derived columns (final) | 48,187 | 14 |

The 48,187 rows cover **40,575 unique hours**: some hours appear on more than one row because the same hour was recorded with several weather descriptions. The KPI therefore counts distinct hours rather than rows.

### Null / missing values

| Finding | Columns affected | Action |
|---|---|---|
| No true nulls or errors in the raw data: every column showed 100% Valid, 0% Error, 0% Empty | All 9 columns | None needed |
| The text "None" is used as a placeholder for "not a holiday" on 48,143 rows (only 61 rows name a real holiday) | `holiday` | Kept as text; it is a label, not missing data |
| 10 temperature readings of **0 K** (absolute zero, a sensor error) | `temp` | Treated as missing in the derived `Temperature (C)` column, so `Temperature (C)` shows under 1% Empty (10 rows) by design |

### Data types

| Column | Auto-detected | Appropriate? | Final type |
|---|---|---|---|
| holiday, weather_main, weather_description | Text | Yes | Text |
| temp | Decimal number | Yes | Decimal number |
| **rain_1h, snow_1h** | **Whole number** | **No**: these hold decimals such as 0.25 mm, which would be rounded away | Changed to **Decimal number** (Replace current, so the original values are kept) |
| clouds_all, traffic_volume | Whole number | Yes | Whole number |
| date_time | Date/Time | Yes | Date/Time |

After the fix `rain_1h` shows 372 distinct values, confirming the decimals were preserved.

### Cleaning and derived columns

| Step | Method | Result |
|---|---|---|
| Remove duplicate rows | Table menu > Remove Duplicates | 17 exact duplicate rows removed |
| **Hour** | Custom column `Time.Hour([date_time])`, Whole number | 24 distinct values (0-23) |
| **Temperature (C)** | Custom column `if [temp] = 0 then null else [temp] - 273.15`, Decimal number | Celsius temperature; 0 K errors become blank |
| **Traffic Category** | Custom column `if [traffic_volume] < 4500 then "Low" else if [traffic_volume] <= 5500 then "Medium" else "High"`, Text | Low 31,229 · Medium 9,858 · High 7,100 rows |
| Year | Custom column `Date.Year([date_time])`, Whole number | Used for filters and the legend |
| Date | Custom column `DateTime.Date([date_time])`, Date | Used for the daily trend chart |

In the report view, **Year** and **Hour** were set to *Don't summarize* so they act as categories rather than being added up.

### DAX measures

```
Total Hours Analysed    = DISTINCTCOUNT ( Traffic[date_time] )
Average Traffic Volume  = AVERAGE ( Traffic[traffic_volume] )
Average Temperature (C) = AVERAGE ( Traffic[Temperature (C)] )
Highest Traffic Weather = MAXX ( TOPN ( 1, VALUES ( Traffic[weather_main] ), [Average Traffic Volume], DESC ), Traffic[weather_main] )
Lowest Traffic Weather  = MAXX ( TOPN ( 1, VALUES ( Traffic[weather_main] ), [Average Traffic Volume], ASC ), Traffic[weather_main] )
Weather Traffic Gap     = MAXX ( VALUES ( Traffic[weather_main] ), [Average Traffic Volume] ) - MINX ( VALUES ( Traffic[weather_main] ), [Average Traffic Volume] )
```

---

## 4.2 Dashboard analysis

### A. Daily traffic trends (2015, 2016, 2017), line chart

| Year | Days with data | Average daily traffic (vehicles/hour) | Lowest day |
|---|---|---|---|
| 2015 | 195 (data starts 11 June) | 3,233 | 26 Dec 2015 (1,478) |
| 2016 | 366 | 3,199 | 23 Jul 2016 (193) |
| 2017 | 365 | 3,347 | 25 Dec 2017 (1,791) |

The daily line repeats a weekly rhythm (busy weekdays, quieter weekends). 2017 is the busiest year. The sharp drop on **23 July 2016** (only 193 vehicles/hour on average) is a sensor fault or road closure rather than real demand, and the other low points fall on holidays such as Christmas.

### B. Hourly traffic patterns (2017), column chart

Traffic is lowest at **03:00 (about 382 vehicles)**, rises steeply from 05:00 to a morning peak at **07:00 (about 4,880)**, dips slightly at midday, and reaches its highest point at **16:00 (about 5,798)**, the only hour above the 5,500 congestion level on average across all days, before falling after 18:00. This is a classic two-peak commuter pattern, with the evening peak the strongest.

### C. Weather impact, bar chart and cards

| Question | Answer |
|---|---|
| Which weather condition has the highest average traffic? | **Clouds: 3,618 vehicles/hour** (15,158 rows) |
| Which has the lowest? | **Squall: 2,062 vehicles/hour** (only 4 rows) |
| Difference between highest and lowest | **1,556 vehicles/hour** |

Squall has only 4 records, so it is not a reliable comparison. Among conditions with at least 100 records, the lowest is **Fog (2,704 vehicles/hour, 912 rows)**, a gap of **914** from Clouds. These differences are partly about *when* each weather occurs: fog and mist tend to form overnight and in the early morning, when traffic is naturally lower, so they are not purely weather effects.

### D. Temperature and traffic, scatter plot with trend line

- **Is there a visible relationship?** Only a weak positive one. The trend line rises gently (Pearson r = 0.13), but the cloud of points is very wide at every temperature, because the hour of the day affects volume far more than temperature does.
- **Temperature range associated with higher traffic:** average traffic is clearly higher above about **25 °C**: 4,404 vehicles/hour at 25-30 °C and 4,786 above 30 °C, compared with roughly 3,000-3,500 between -20 °C and 25 °C. Warm readings mostly occur on summer afternoons, when commuter traffic is high.
- **Notable outliers:**
  - The 10 readings of 0 K (-273 °C) would sit far off the chart; they were blanked in `Temperature (C)` so they do not distort it.
  - A band of near-zero traffic at ordinary temperatures, mostly night hours, plus sensor-fault hours such as 23-24 July 2016.
  - The coldest hours (below -25 °C, down to -29.8 °C) average only about 1,910 vehicles/hour, so extreme cold suppresses travel.

---

## 4.3 KPI cards and slicers

| KPI card | Value (no filters) |
|---|---|
| Total Hours Analysed | 40,575 (shown as 41K) |
| Average Traffic Volume | 3,260 vehicles/hour (shown as 3.26K) |
| Average Temperature (C) | 8.11 °C |

Additional cards answer section C directly: Highest Traffic Weather (Clouds), Lowest Traffic Weather (Squall) and Weather Traffic Gap (1,556).

| Slicer | Style | Example use |
|---|---|---|
| Hour range | Between (slider) | Setting 7-9 raises Average Traffic Volume to about 4,573 |
| Weather condition | Dropdown | Isolate Rain or Snow hours |
| Traffic category | Tiles (High / Low / Medium) | Selecting High shows an average of about 6,061 |

All slicers filter every visual and card on the page. Clicking a bar or point also cross-filters the page, which is useful for exploration.

---

## Key takeaways for the Smart City Mobility Team

1. **Congestion is a weekday commuting problem**, peaking at 07:00 and especially 16:00. Time of day matters far more than weather or temperature.
2. **Weather differences are modest and partly explained by timing.** The 1,556-vehicle gap depends on a 4-record category; the reliable gap is about 900.
3. **Data quality needs attention before reporting:** duplicate rows, impossible 0 K readings, wrongly auto-detected data types and sensor outages were all found and handled during preparation.
