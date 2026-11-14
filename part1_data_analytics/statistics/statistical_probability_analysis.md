# Part 1: Written Answers for Tasks 1 to 3

Task 1 numbers come from `traffic_analysis.sql` (run against `traffic.db`); Task 2 and 3 numbers were computed in Python (pandas and SciPy) by `stats_probability_analysis.py` in this folder (run `python stats_probability_analysis.py`; it writes `stats_probability_results.md`, `stats_probability_results.json` and three figures) and cross-checked with the SQL queries X1 to X5 in the same file. All analysis uses the raw dataset of 48,204 records unless stated otherwise.

---

## Task 1: SQL-based traffic analysis

### 1.1 Load and verify

The CSV was loaded into SQLite with the `.import` command at the top of `traffic_analysis.sql`, then copied into a typed table `traffic` (REAL, INTEGER and TEXT columns, plus `year` and `date` helper columns). Running `sqlite3 traffic.db < traffic_analysis.sql` repeats the whole load and analysis. Verification queries 1.1a to 1.1h confirm:

- **48,204 rows** loaded, matching the 48,205 lines in the CSV minus the header.
- All 9 source columns are present with correct types (REAL for temperature and precipitation, INTEGER for cloud cover and volume, TEXT for labels and timestamps).
- The date range is 2 Oct 2012 09:00 to 30 Sep 2018 23:00.
- There are **no NULL or blank values** in any column.
- Data-quality issues to note: 10 rows with temp = 0 K, 1 row with 9,831.3 mm of rain in one hour, 17 exact duplicate rows, and 7,629 rows that repeat a timestamp (the same hour recorded with more than one weather description).

### 1.2 Annual traffic trends

| Year | Total volume | Change | % change | Direction | Hours covered |
|---|---|---|---|---|---|
| 2012 | 8,208,767 | | | baseline | 24% |
| 2013 | 28,177,412 | +19,968,645 | +243.3% | increase | 83% |
| 2014 | 15,731,289 | -12,446,123 | -44.2% | decrease | 51% |
| 2015 | 14,181,206 | -1,550,083 | -9.9% | decrease | 41% |
| 2016 | 29,494,821 | +15,313,615 | +108.0% | increase | 89% |
| 2017 | 35,428,156 | +5,933,335 | +20.1% | increase | 99.5% |

**Increases:** 2013, 2016, 2017. **Decreases:** 2014, 2015.

**Observation 1: the raw changes are driven by data coverage, not by real demand.** 2012 only starts on 2 October, 2014 stops on 8 August and 2015 restarts on 11 June (a 10-month gap). The years with the biggest "drops" are exactly the years with the least data. A total of 35.4 million in 2017 versus 14.2 million in 2015 does not mean traffic more than doubled.

**Observation 2: on a coverage-adjusted basis, demand was flat from 2012 to 2016 and grew in 2017.** Counting each hour once and averaging gives 3,227, 3,310, 3,270, 3,258, 3,194 and 3,377 vehicles per hour for 2012 to 2017. 2016 was the lowest year and 2017 the highest (+5.7%). A like-for-like October to December comparison (the only months covered in every year except 2014) supports this: 3,087 in 2016 rising to 3,322 in 2017 (+7.6%). Average daily volume on full days also peaks in 2017 (80,740).

**Observation 3 (data management):** duplicated timestamps inflate totals by 7% (2014) to 21% (2012 and 2015) (for example, 2017 falls from 35.4 to 29.4 million when each hour is counted once). Any official reporting should de-duplicate first.

### 1.3 Temperature around holidays (2015 to 2017)

Holiday labels appear only on the midnight record, so the label was used to find the date and then every hour on that date was analysed. Repeated timestamps were collapsed first so each hour counts once. Comparable days are the other days in the same month with the same weekday, excluding all holiday dates.

| Holiday | Date | Avg temp | Min / max | YoY change | Avg traffic/hour | vs comparable days |
|---|---|---|---|---|---|---|
| New Year's Day 2015 | (no data: falls in the 2014 to 2015 gap) | | | | | |
| New Year's Day 2016 | 2016-01-01 (Friday) | -6.1 °C | -9.2 / -1.2 °C | | 1,832 | -48.4% |
| New Year's Day 2017 | 2017-01-02 (observed Monday) | -1.3 °C | -2.7 / -0.1 °C | +4.8 °C | 2,091 | -35.5% |
| Labor Day 2015 | 2015-09-07 | 22.2 °C | 18.7 / 26.5 °C | | 2,430 | -34.5% |
| Labor Day 2016 | 2016-09-05 | 21.9 °C | 18.7 / 24.8 °C | -0.3 °C | 2,152 | -34.7% |
| Labor Day 2017 | 2017-09-04 | 18.1 °C | 11.8 / 23.1 °C | -3.8 °C | 2,395 | -32.7% |

Notable changes: New Year's 2017 was 4.8 °C milder than 2016 (the calendar day, Sunday 1 January 2017, averaged -3.1 °C). Labor Day cooled each year, with a sharp 3.8 °C drop in 2017, making it 4.1 °C cooler than 2015. New Year's Day 2016 has only 18 recorded hours.

**Relevance to traffic:** temperature changes appear to have little effect. Every holiday carried 33% to 48% less traffic than comparable normal days, which is a calendar effect. Within Labor Day, traffic does not follow temperature: 2017 was 4.1 °C cooler than 2015 yet carried almost as much traffic (2,395 vs 2,430 vehicles per hour), while 2016, the day with rain and thunderstorms, carried the least (2,152). The milder New Year 2017 did carry more traffic than 2016, but it was a Monday observed holiday (versus a Friday), which confounds the comparison. Precipitation and the weekday of the holiday look more relevant than temperature, though five holiday days (each compared against only three or four normal days) are too few to draw firm conclusions.

---

## Task 2: Descriptive statistics and correlation

### 2.1 Traffic volume statistics

| Statistic | Value |
|---|---|
| Mean | 3,259.82 vehicles/hour |
| Median | 3,380 |
| Standard deviation (sample, n − 1) | 1,986.86 |
| Variance (sample, n − 1) | 3,947,615.32 |
| Range | 7,280 (min 0, max 7,280) |

SQL query X1 reports the population variance (dividing by n rather than n − 1), 3,947,533; the two differ only because of that divisor.

**Interpretation:**
- The typical hour carries about 3,300 vehicles, but the standard deviation is about 61% of the mean, so variability is very high.
- The mean is slightly below the median (and skewness is slightly negative at -0.09) because quiet night hours pull the average down.
- The distribution is **bimodal**: one cluster of night hours under about 1,000 and a broad daytime cluster between 4,000 and 6,000. A single average therefore describes almost no real hour well, which is why later analysis splits by hour of day.
- The range of 7,280 includes two hours with zero vehicles, which look like sensor faults or closures rather than real demand.
- The variance is in squared units (vehicles squared), so the standard deviation is the more interpretable spread measure.

### 2.2 Correlation between temperature and traffic volume

| Measure | Value |
|---|---|
| Pearson r (all rows) | 0.130 |
| Pearson r (excluding 10 rows at 0 K) | 0.132 |
| Spearman rho | 0.133 |
| R squared | 0.018 |

- **Direction:** positive. Warmer hours tend to have slightly more traffic.
- **Strength:** weak. Temperature explains under 2% of the variation in volume. The p-value is tiny only because the sample is very large; statistical significance is not the same as practical importance.
- **Why correlation does not imply causation:**
  1. **Confounding by time of day:** temperatures are higher in the afternoon, when commuters travel, and lower overnight, when roads are empty.
  2. **Confounding by season:** summer brings both warmth and extra leisure and tourism trips.
  3. **No mechanism or experiment:** we observe co-occurrence only; nothing in the data shows that changing temperature would change how many people drive.
  4. **Outliers and data quality:** ten 0 K readings are impossible values that would distort any naive analysis.

---

## Task 3: Probability and congestion (congestion = volume > 5,500)

### 3.1 Basic probability

| Probability | Value |
|---|---|
| P(Congestion) | 0.1473 (7,100 of 48,204 hours) |
| P(Clear Weather) | 0.2778 |
| P(Congestion AND Clear) | 0.0366 |

### 3.2 Conditional probability

| Probability | Value |
|---|---|
| P(Clear \| Congestion) | 0.2483 |
| P(High Temp \| Congestion), high temp > 292 K | 0.2630 (compared with P(High Temp) = 0.2438) |

**Independence test:** if congestion and clear weather were independent, P(C ∩ Clear) would equal P(C) × P(Clear) = 0.1473 × 0.2778 = **0.0409**. The observed value is **0.0366**, so they are **not strictly independent**. Equivalently, P(Clear | Congestion) = 0.248 is lower than P(Clear) = 0.278. A chi-square test confirms that the difference is statistically significant (χ² = 35.9, p < 0.001), but the observed joint probability is 89% of the expected value, so the dependence is weak.

**Odds ratio (clear vs cloudy):**

| | Congested | Not congested | Odds |
|---|---|---|---|
| Clear | 1,763 | 11,628 | 0.1516 |
| Clouds | 2,592 | 12,572 | 0.2062 |

Odds ratio = 0.1516 / 0.2062 = **0.735** (95% CI 0.689 to 0.785). The odds of congestion in clear weather are about 26% lower than in cloudy weather.

**Conclusion:** weather and congestion are statistically associated, but weakly, and the association mostly reflects *when* each type of weather occurs. Clear skies account for 33% of night hours but only 23% of peak hours. When clear and cloudy hours are compared at the same hour of day (Mantel-Haenszel adjustment), the odds ratio becomes **1.006**, meaning essentially no difference. High temperature raises the chance of congestion only slightly (26.3% vs 24.4%). Congestion is mainly a function of time of day and day type; weather alone is a poor predictor.
