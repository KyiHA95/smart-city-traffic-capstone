-- =====================================================================
-- Smart City Traffic Intelligence - Part 1, Task 1: SQL-Based Analysis
-- Engine : SQLite 3.25+ (window functions LAG / OVER are used)
--
-- HOW TO RUN (sqlite3 command-line shell, from the part1_data_analytics/sql folder):
--     sqlite3 traffic.db < traffic_analysis.sql
-- This loads the CSV (Task 1.1), builds the typed `traffic` table and runs
-- every analysis query, printing each result with its title.
--
-- Using DB Browser for SQLite instead? Use File > Import > Table from CSV
-- (table name traffic_raw, "column names in first line" ticked), then run
-- this file from the "CREATE TYPED TABLE" step onwards; the dot-commands
-- (lines starting with ".") are shell-only and can be skipped.
-- =====================================================================

.headers on
.mode column

-- =====================================================================
-- TASK 1.1  LOAD THE DATASET
-- =====================================================================
-- Step 1: import the comma-delimited CSV as-is (all columns arrive as TEXT)
DROP TABLE IF EXISTS traffic_raw;
.mode csv
.import ../../data/Metro_Interstate_Traffic_Volume.csv traffic_raw
.mode column

-- Step 2: CREATE TYPED TABLE with correct data types and two helper columns
DROP TABLE IF EXISTS traffic;
CREATE TABLE traffic (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    holiday             TEXT,
    temp                REAL,      -- Kelvin
    rain_1h             REAL,      -- mm in the hour
    snow_1h             REAL,      -- mm in the hour
    clouds_all          INTEGER,   -- % cloud cover
    weather_main        TEXT,
    weather_description TEXT,
    date_time           TEXT,      -- 'YYYY-MM-DD HH:MM:SS'
    traffic_volume      INTEGER,   -- vehicles per hour, westbound I-94
    year                INTEGER,   -- derived from date_time
    date                TEXT       -- derived from date_time ('YYYY-MM-DD')
);

INSERT INTO traffic (holiday, temp, rain_1h, snow_1h, clouds_all, weather_main,
                     weather_description, date_time, traffic_volume, year, date)
SELECT holiday,
       CAST(temp AS REAL), CAST(rain_1h AS REAL), CAST(snow_1h AS REAL),
       CAST(clouds_all AS INTEGER), weather_main, weather_description, date_time,
       CAST(traffic_volume AS INTEGER),
       CAST(substr(date_time, 1, 4) AS INTEGER), substr(date_time, 1, 10)
FROM traffic_raw;

DROP TABLE traffic_raw;
CREATE INDEX idx_year ON traffic(year);
CREATE INDEX idx_date ON traffic(date);
CREATE INDEX idx_dt   ON traffic(date_time);

-- =====================================================================
-- TASK 1.1  VERIFY THE LOADED DATA
-- =====================================================================

-- @name: 1.1a Row count (expected 48,204 = CSV lines minus header)
.print
.print '== 1.1a Row count (expected 48,204 = CSV lines minus header) =='
SELECT COUNT(*) AS total_rows FROM traffic;

-- @name: 1.1b Table structure (9 source columns + id, year, date helpers)
.print
.print '== 1.1b Table structure (9 source columns + id, year, date helpers) =='
SELECT name AS column_name, type AS data_type
FROM pragma_table_info('traffic');

-- @name: 1.1c Date range covered
.print
.print '== 1.1c Date range covered =='
SELECT MIN(date_time) AS first_record,
       MAX(date_time) AS last_record,
       COUNT(DISTINCT date) AS distinct_days,
       COUNT(DISTINCT date_time) AS distinct_hours
FROM traffic;

-- @name: 1.1d Null / blank check per column
.print
.print '== 1.1d Null / blank check per column =='
SELECT SUM(holiday IS NULL OR holiday = '')                     AS holiday_missing,
       SUM(temp IS NULL)                                         AS temp_missing,
       SUM(rain_1h IS NULL)                                      AS rain_missing,
       SUM(snow_1h IS NULL)                                      AS snow_missing,
       SUM(clouds_all IS NULL)                                   AS clouds_missing,
       SUM(weather_main IS NULL OR weather_main = '')            AS weather_main_missing,
       SUM(weather_description IS NULL OR weather_description='') AS weather_desc_missing,
       SUM(date_time IS NULL OR date_time = '')                  AS date_time_missing,
       SUM(traffic_volume IS NULL)                               AS volume_missing
FROM traffic;

-- @name: 1.1e Numeric ranges (sanity check for impossible values)
.print
.print '== 1.1e Numeric ranges (sanity check for impossible values) =='
SELECT MIN(temp) AS min_temp_K, MAX(temp) AS max_temp_K,
       MAX(rain_1h) AS max_rain_mm, MAX(snow_1h) AS max_snow_mm,
       MIN(clouds_all) AS min_clouds, MAX(clouds_all) AS max_clouds,
       MIN(traffic_volume) AS min_volume, MAX(traffic_volume) AS max_volume
FROM traffic;

-- @name: 1.1f Data-quality flags (impossible readings and duplicates)
.print
.print '== 1.1f Data-quality flags (impossible readings and duplicates) =='
SELECT
  (SELECT COUNT(*) FROM traffic WHERE temp = 0)          AS temp_zero_kelvin_rows,
  (SELECT COUNT(*) FROM traffic WHERE rain_1h > 1000)    AS rain_over_1000mm_rows,
  (SELECT COUNT(*) - COUNT(DISTINCT date_time) FROM traffic) AS repeated_timestamp_rows,
  (SELECT COUNT(*) FROM (
      SELECT holiday, temp, rain_1h, snow_1h, clouds_all, weather_main,
             weather_description, date_time, traffic_volume, COUNT(*) AS n
      FROM traffic
      GROUP BY holiday, temp, rain_1h, snow_1h, clouds_all, weather_main,
               weather_description, date_time, traffic_volume
      HAVING n > 1))                                     AS exact_duplicate_groups;

-- @name: 1.1g Sample of loaded rows
.print
.print '== 1.1g Sample of loaded rows =='
SELECT * FROM traffic ORDER BY id LIMIT 5;

-- @name: 1.1h Weather categories loaded
.print
.print '== 1.1h Weather categories loaded =='
SELECT weather_main, COUNT(*) AS records
FROM traffic GROUP BY weather_main ORDER BY records DESC;


-- =====================================================================
-- TASK 1.2  ANNUAL TRAFFIC TRENDS (2012-2017)
-- =====================================================================

-- @name: 1.2a Total yearly traffic volume with year-on-year change
.print
.print '== 1.2a Total yearly traffic volume with year-on-year change =='
WITH yearly AS (
    SELECT year, SUM(traffic_volume) AS total_volume
    FROM traffic
    WHERE year BETWEEN 2012 AND 2017
    GROUP BY year
)
SELECT year,
       total_volume,
       total_volume - LAG(total_volume) OVER (ORDER BY year) AS change_vs_prev_year,
       ROUND(100.0 * (total_volume - LAG(total_volume) OVER (ORDER BY year))
             / LAG(total_volume) OVER (ORDER BY year), 1)     AS pct_change,
       CASE
         WHEN LAG(total_volume) OVER (ORDER BY year) IS NULL THEN 'baseline'
         WHEN total_volume > LAG(total_volume) OVER (ORDER BY year) THEN 'increase'
         ELSE 'decrease'
       END AS direction
FROM yearly
ORDER BY year;

-- @name: 1.2b Data coverage per year (explains why raw totals mislead)
.print
.print '== 1.2b Data coverage per year (explains why raw totals mislead) =='
SELECT year,
       COUNT(*)                   AS records,
       COUNT(DISTINCT date_time)  AS distinct_hours,
       COUNT(DISTINCT date)       AS days_with_data,
       MIN(date_time)             AS first_record,
       MAX(date_time)             AS last_record,
       ROUND(100.0 * COUNT(DISTINCT date_time) /
             (CASE WHEN year % 4 = 0 THEN 8784 ELSE 8760 END), 1) AS pct_hours_covered
FROM traffic
WHERE year BETWEEN 2012 AND 2017
GROUP BY year
ORDER BY year;

-- @name: 1.2c Coverage-adjusted trend (one row per hour, then average)
.print
.print '== 1.2c Coverage-adjusted trend (one row per hour, then average) =='
-- Repeated timestamps (same hour, several weather descriptions) are
-- collapsed so each hour is counted once. Average hourly and average
-- daily volume are comparable across years with different coverage.
WITH hourly AS (
    SELECT date_time, year, date, MAX(traffic_volume) AS volume
    FROM traffic
    WHERE year BETWEEN 2012 AND 2017
    GROUP BY date_time
),
daily AS (
    SELECT year, date, SUM(volume) AS day_volume, COUNT(*) AS hours_in_day
    FROM hourly GROUP BY year, date
),
yr AS (
    SELECT h.year,
           SUM(h.volume)                    AS dedup_total_volume,
           ROUND(AVG(h.volume), 1)          AS avg_hourly_volume,
           (SELECT ROUND(AVG(day_volume), 0) FROM daily d
             WHERE d.year = h.year AND d.hours_in_day >= 20) AS avg_daily_volume_full_days
    FROM hourly h GROUP BY h.year
)
SELECT year, dedup_total_volume, avg_hourly_volume,
       ROUND(avg_hourly_volume - LAG(avg_hourly_volume) OVER (ORDER BY year), 1) AS hourly_avg_change,
       ROUND(100.0 * (avg_hourly_volume - LAG(avg_hourly_volume) OVER (ORDER BY year))
             / LAG(avg_hourly_volume) OVER (ORDER BY year), 2) AS hourly_avg_pct_change,
       avg_daily_volume_full_days
FROM yr ORDER BY year;

-- @name: 1.2d Like-for-like comparison: same months in every year (Oct-Dec)
.print
.print '== 1.2d Like-for-like comparison: same months in every year (Oct-Dec) =='
-- Oct-Dec is covered in every year except 2014 (data gap). Each hour is
-- counted once, consistent with query 1.2c.
WITH hourly AS (
    SELECT date_time, year, MAX(traffic_volume) AS volume
    FROM traffic
    WHERE year BETWEEN 2012 AND 2017
      AND CAST(strftime('%m', date_time) AS INTEGER) BETWEEN 10 AND 12
    GROUP BY date_time
)
SELECT year,
       COUNT(*)                AS hours,
       SUM(volume)             AS total_oct_dec,
       ROUND(AVG(volume), 1)   AS avg_hourly_volume_oct_dec,
       ROUND(100.0 * (AVG(volume) - LAG(AVG(volume)) OVER (ORDER BY year))
             / LAG(AVG(volume)) OVER (ORDER BY year), 1) AS pct_change_vs_prev_available_year
FROM hourly GROUP BY year ORDER BY year;

-- @name: 1.2e Monthly record counts around the 2014-2015 gap
.print
.print '== 1.2e Monthly record counts around the 2014-2015 gap =='
SELECT strftime('%Y-%m', date_time) AS month, COUNT(*) AS records
FROM traffic
WHERE date_time BETWEEN '2014-06-01' AND '2015-08-01'
GROUP BY month ORDER BY month;


-- =====================================================================
-- TASK 1.3  TEMPERATURE AROUND HOLIDAYS (2015, 2016, 2017)
-- In this dataset the holiday label is set only on the midnight record
-- of the (observed) holiday, so the label is used to find the holiday
-- DATE, and every hourly record on that date is then analysed.
-- Temperatures are converted from Kelvin to Celsius (K - 273.15).
-- Repeated timestamps are collapsed first (one row per hour, temperature
-- averaged within the hour) so no hour is counted twice.
-- =====================================================================

-- @name: 1.3a Holiday dates present in the data (all years)
.print
.print '== 1.3a Holiday dates present in the data (all years) =='
SELECT DISTINCT holiday, date
FROM traffic
WHERE holiday IN ('New Years Day', 'Labor Day')
ORDER BY holiday, date;

-- @name: 1.3b Full-day temperature and traffic on New Year's Day and Labor Day, 2015-2017
.print
.print '== 1.3b Full-day temperature and traffic on New Years Day and Labor Day, 2015-2017 =='
WITH hourly AS (
    SELECT date_time, date, year, AVG(temp) AS temp, MAX(traffic_volume) AS volume
    FROM traffic GROUP BY date_time
),
holiday_dates AS (
    SELECT DISTINCT holiday, date, year
    FROM traffic
    WHERE holiday IN ('New Years Day', 'Labor Day')
      AND year BETWEEN 2015 AND 2017
),
day_stats AS (
    SELECT h.holiday, h.year, h.date,
           COUNT(*)                                 AS hours_recorded,
           ROUND(AVG(t.temp), 2)                    AS avg_temp_K,
           ROUND(AVG(t.temp) - 273.15, 1)           AS avg_temp_C,
           ROUND(MIN(t.temp) - 273.15, 1)           AS min_temp_C,
           ROUND(MAX(t.temp) - 273.15, 1)           AS max_temp_C,
           ROUND(AVG(t.volume), 0)                  AS avg_hourly_traffic,
           (SELECT GROUP_CONCAT(DISTINCT w.weather_main)
              FROM traffic w WHERE w.date = h.date)  AS weather_seen
    FROM holiday_dates h
    JOIN hourly t ON t.date = h.date
    GROUP BY h.holiday, h.year, h.date
)
SELECT holiday, year, date, hours_recorded, avg_temp_K, avg_temp_C,
       min_temp_C, max_temp_C,
       ROUND(avg_temp_C - LAG(avg_temp_C) OVER (PARTITION BY holiday ORDER BY year), 1)
           AS yoy_temp_change_C,
       avg_hourly_traffic, weather_seen
FROM day_stats
ORDER BY holiday DESC, year;

-- @name: 1.3c New Year 2015 check (not in dataset because of the data gap)
.print
.print '== 1.3c New Year 2015 check (not in dataset because of the data gap) =='
SELECT COUNT(*) AS records_on_2015_01_01
FROM traffic WHERE date = '2015-01-01';

-- @name: 1.3d New Year 2017: calendar day (Sun 1 Jan) vs observed holiday (Mon 2 Jan)
.print
.print '== 1.3d New Year 2017: calendar day (Sun 1 Jan) vs observed holiday (Mon 2 Jan) =='
WITH hourly AS (
    SELECT date_time, date, AVG(temp) AS temp, MAX(traffic_volume) AS volume
    FROM traffic WHERE date IN ('2017-01-01', '2017-01-02') GROUP BY date_time
)
SELECT date,
       CASE strftime('%w', date) WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' END AS weekday,
       ROUND(AVG(temp) - 273.15, 1)   AS avg_temp_C,
       ROUND(AVG(volume), 0)          AS avg_hourly_traffic
FROM hourly GROUP BY date;

-- @name: 1.3e Holiday traffic vs. comparable normal days (same year, month and weekday)
.print
.print '== 1.3e Holiday traffic vs. comparable normal days (same year, month and weekday) =='
-- Comparison days exclude every holiday date (e.g. Martin Luther King Jr
-- Day on Monday 16 Jan 2017 is not treated as a normal Monday).
WITH hourly AS (
    SELECT date_time, date, MAX(traffic_volume) AS volume
    FROM traffic GROUP BY date_time
),
all_holiday_dates AS (
    SELECT DISTINCT date FROM traffic WHERE holiday <> 'None'
),
target AS (
    SELECT DISTINCT holiday, date, year
    FROM traffic
    WHERE holiday IN ('New Years Day', 'Labor Day') AND year BETWEEN 2015 AND 2017
)
SELECT g.holiday, g.year,
       ROUND(AVG(CASE WHEN t.date = g.date THEN t.volume END), 0)  AS holiday_avg_hourly,
       ROUND(AVG(CASE WHEN t.date <> g.date THEN t.volume END), 0) AS normal_same_weekday_avg,
       COUNT(DISTINCT CASE WHEN t.date <> g.date THEN t.date END)  AS comparison_days,
       ROUND(100.0 * AVG(CASE WHEN t.date = g.date THEN t.volume END) /
             AVG(CASE WHEN t.date <> g.date THEN t.volume END) - 100, 1) AS pct_difference
FROM target g
JOIN hourly t
  ON strftime('%Y-%m', t.date) = strftime('%Y-%m', g.date)
 AND strftime('%w', t.date) = strftime('%w', g.date)
 AND (t.date = g.date OR t.date NOT IN (SELECT date FROM all_holiday_dates))
GROUP BY g.holiday, g.year
ORDER BY g.holiday DESC, g.year;


-- =====================================================================
-- SQL CROSS-CHECKS FOR TASKS 2 AND 3 (main analysis is in Python;
-- these confirm the same numbers are reproducible directly in SQL)
-- =====================================================================

-- @name: X1 Traffic volume summary statistics (population variance shown)
.print
.print '== X1 Traffic volume summary statistics (population variance shown) =='
SELECT COUNT(*)                                           AS n,
       ROUND(AVG(traffic_volume), 2)                      AS mean_volume,
       MIN(traffic_volume)                                AS min_volume,
       MAX(traffic_volume)                                AS max_volume,
       MAX(traffic_volume) - MIN(traffic_volume)          AS range_volume,
       ROUND(AVG(traffic_volume * traffic_volume) -
             AVG(traffic_volume) * AVG(traffic_volume), 1) AS population_variance
FROM traffic;

-- @name: X2 Median traffic volume
.print
.print '== X2 Median traffic volume =='
SELECT AVG(traffic_volume) AS median_volume FROM (
    SELECT traffic_volume FROM traffic ORDER BY traffic_volume
    LIMIT 2 - (SELECT COUNT(*) FROM traffic) % 2
    OFFSET (SELECT (COUNT(*) - 1) / 2 FROM traffic));

-- @name: X3 Basic and conditional probabilities (congestion = volume > 5500)
.print
.print '== X3 Basic and conditional probabilities (congestion = volume > 5500) =='
SELECT
  ROUND(AVG(traffic_volume > 5500), 4)                                  AS p_congestion,
  ROUND(AVG(weather_main = 'Clear'), 4)                                 AS p_clear,
  ROUND(AVG(traffic_volume > 5500 AND weather_main = 'Clear'), 4)       AS p_congestion_and_clear,
  ROUND(1.0 * SUM(traffic_volume > 5500 AND weather_main = 'Clear')
        / SUM(traffic_volume > 5500), 4)                                AS p_clear_given_congestion,
  ROUND(1.0 * SUM(traffic_volume > 5500 AND temp > 292)
        / SUM(traffic_volume > 5500), 4)                                AS p_hightemp_given_congestion
FROM traffic;

-- @name: X4 Congestion odds: Clear vs Clouds
.print
.print '== X4 Congestion odds: Clear vs Clouds =='
SELECT weather_main,
       SUM(traffic_volume > 5500)                        AS congested,
       SUM(traffic_volume <= 5500)                       AS not_congested,
       ROUND(1.0 * SUM(traffic_volume > 5500) /
             SUM(traffic_volume <= 5500), 4)             AS odds_of_congestion
FROM traffic
WHERE weather_main IN ('Clear', 'Clouds')
GROUP BY weather_main;

-- @name: X5 Average traffic by weather condition (feeds Power BI section C)
.print
.print '== X5 Average traffic by weather condition (feeds Power BI section C) =='
SELECT weather_main,
       COUNT(*)                       AS records,
       ROUND(AVG(traffic_volume), 1)  AS avg_traffic
FROM traffic
GROUP BY weather_main
ORDER BY avg_traffic DESC;
