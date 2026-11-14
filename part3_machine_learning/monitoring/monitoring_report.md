# Model monitoring report

**Overall system status: ALERT / Requires investigation**

Model monitored: champion `traffic_volume_regressor` (see `models/MODEL_VERSIONS.md`). Error reference: the same calendar month of 2017 (first out-of-sample year). Feature reference: the 2016-2017 normal for that calendar month (earlier years used a different recording scheme for cloud cover and weather labels). Alert rules: MAE more than 25% above the reference, or PSI above 0.25 for temperature, cloud cover, weather type or the model's predictions (PSI 0.10-0.25 is noted as moderate).

| Batch | Hours | MAE | Ref. MAE | MAE ratio | PSI temp | PSI clouds | PSI weather | PSI predictions | Status | Action |
|---|---|---|---|---|---|---|---|---|---|---|
| 2018-01 | 742 | 313 | 259 | 1.21 | 0.13 | 0.21 | 0.21 | 0.01 | **PASS / Normal** | None |
| 2018-02 | 672 | 262 | 233 | 1.12 | 0.78 | 0.03 | 0.22 | 0.01 | **ALERT / Requires investigation** | Watch: inputs are unusual but accuracy is within limits |
| 2018-03 | 733 | 227 | 211 | 1.07 | 1.11 | 0.06 | 0.26 | 0.00 | **ALERT / Requires investigation** | Watch: inputs are unusual but accuracy is within limits |
| 2018-04 | 720 | 324 | 208 | 1.56 | 0.97 | 0.12 | 0.61 | 0.01 | **ALERT / Requires investigation** | Investigate now: accuracy has degraded |
| 2018-05 | 743 | 184 | 196 | 0.94 | 0.50 | 0.17 | 0.14 | 0.01 | **ALERT / Requires investigation** | Watch: inputs are unusual but accuracy is within limits |
| 2018-06 | 719 | 173 | 190 | 0.91 | 0.15 | 0.32 | 0.14 | 0.00 | **ALERT / Requires investigation** | Watch: inputs are unusual but accuracy is within limits |
| 2018-07 | 744 | 214 | 250 | 0.85 | 0.03 | 0.06 | 0.13 | 0.01 | **PASS / Normal** | None |
| 2018-08 | 740 | 181 | 203 | 0.89 | 0.17 | 0.01 | 0.18 | 0.00 | **PASS / Normal** | None |
| 2018-09 | 720 | 186 | 221 | 0.84 | 0.14 | 0.05 | 0.19 | 0.00 | **PASS / Normal** | None |
| SIM sensor_fault | 720 | 1279 | 221 | 5.79 | 0.14 | 0.05 | 0.19 | 0.00 | **ALERT / Requires investigation** | Investigate now: accuracy has degraded |
| SIM temperature_shift | 720 | 215 | 221 | 0.98 | 2.34 | 0.05 | 0.19 | 0.01 | **ALERT / Requires investigation** | Watch: inputs are unusual but accuracy is within limits |

## Alerts and notes

- 2018-01: moderate shift: PSI(temp_c) = 0.13 (below the alert limit)
- 2018-01: moderate shift: PSI(clouds_all) = 0.21 (below the alert limit)
- 2018-01: moderate shift: PSI(weather_main) = 0.21 (below the alert limit)
- **2018-02: ALERT**, feature drift: PSI(temp_c) = 0.78 > 0.25
- 2018-02: moderate shift: PSI(weather_main) = 0.22 (below the alert limit)
- **2018-03: ALERT**, feature drift: PSI(temp_c) = 1.11 > 0.25
- **2018-03: ALERT**, feature drift: PSI(weather_main) = 0.26 > 0.25
- **2018-04: ALERT**, prediction error drift: MAE 324 vs reference 208 (+56%, limit +25%)
- **2018-04: ALERT**, feature drift: PSI(temp_c) = 0.97 > 0.25
- **2018-04: ALERT**, feature drift: PSI(weather_main) = 0.61 > 0.25
- 2018-04: moderate shift: PSI(clouds_all) = 0.12 (below the alert limit)
- **2018-05: ALERT**, feature drift: PSI(temp_c) = 0.50 > 0.25
- 2018-05: moderate shift: PSI(clouds_all) = 0.17 (below the alert limit)
- 2018-05: moderate shift: PSI(weather_main) = 0.14 (below the alert limit)
- **2018-06: ALERT**, feature drift: PSI(clouds_all) = 0.32 > 0.25
- 2018-06: moderate shift: PSI(temp_c) = 0.15 (below the alert limit)
- 2018-06: moderate shift: PSI(weather_main) = 0.14 (below the alert limit)
- 2018-07: moderate shift: PSI(weather_main) = 0.13 (below the alert limit)
- 2018-08: moderate shift: PSI(temp_c) = 0.17 (below the alert limit)
- 2018-08: moderate shift: PSI(weather_main) = 0.18 (below the alert limit)
- 2018-09: moderate shift: PSI(temp_c) = 0.14 (below the alert limit)
- 2018-09: moderate shift: PSI(weather_main) = 0.19 (below the alert limit)
- **SIM sensor_fault: ALERT**, prediction error drift: MAE 1279 vs reference 221 (+479%, limit +25%)
- SIM sensor_fault: moderate shift: PSI(temp_c) = 0.14 (below the alert limit)
- SIM sensor_fault: moderate shift: PSI(weather_main) = 0.19 (below the alert limit)
- **SIM temperature_shift: ALERT**, feature drift: PSI(temp_c) = 2.34 > 0.25
- SIM temperature_shift: moderate shift: PSI(weather_main) = 0.19 (below the alert limit)

## Interpretation

4 of 9 real 2018 monthly batches passed every check.

**2018-02** (Watch: inputs are unusual but accuracy is within limits). Mean temperature -9.4 °C vs a 2016-2017 normal of -2.9 °C. Reasons: feature drift: PSI(temp_c) = 0.78 > 0.25. Days with the largest errors:
- 2018-02-24: Snow, actual 2,401 vs predicted 2,756 vehicles/hour
- 2018-02-02: Clear, actual 3,841 vs predicted 3,458 vehicles/hour
- 2018-02-19: Snow, actual 2,456 vs predicted 2,741 vehicles/hour

**2018-03** (Watch: inputs are unusual but accuracy is within limits). Mean temperature -0.4 °C vs a 2016-2017 normal of 2.0 °C. Reasons: feature drift: PSI(temp_c) = 1.11 > 0.25; feature drift: PSI(weather_main) = 0.26 > 0.25. Days with the largest errors:
- 2018-03-05: Snow, actual 2,373 vs predicted 3,373 vehicles/hour
- 2018-03-31: Snow, actual 2,604 vs predicted 2,994 vehicles/hour
- 2018-03-24: Clear, actual 4,033 vs predicted 3,700 vehicles/hour

**2018-04** (Investigate now: accuracy has degraded). Mean temperature 2.5 °C vs a 2016-2017 normal of 8.9 °C. Reasons: prediction error drift: MAE 324 vs reference 208 (+56%, limit +25%); feature drift: PSI(temp_c) = 0.97 > 0.25; feature drift: PSI(weather_main) = 0.61 > 0.25. Days with the largest errors:
- 2018-04-14: Snow, actual 1,144 vs predicted 3,002 vehicles/hour
- 2018-04-15: Snow, actual 1,238 vs predicted 2,225 vehicles/hour
- 2018-04-03: Snow, actual 2,719 vs predicted 3,529 vehicles/hour

**2018-05** (Watch: inputs are unusual but accuracy is within limits). Mean temperature 18.8 °C vs a 2016-2017 normal of 14.6 °C. Reasons: feature drift: PSI(temp_c) = 0.50 > 0.25. Days with the largest errors:
- 2018-05-28: Clear, actual 2,136 vs predicted 2,418 vehicles/hour
- 2018-05-20: Clear, actual 2,332 vs predicted 2,552 vehicles/hour
- 2018-05-25: Thunderstorm, actual 3,596 vs predicted 3,837 vehicles/hour

**2018-06** (Watch: inputs are unusual but accuracy is within limits). Mean temperature 21.8 °C vs a 2016-2017 normal of 20.6 °C. Reasons: feature drift: PSI(clouds_all) = 0.32 > 0.25. Days with the largest errors:
- 2018-06-09: Thunderstorm, actual 2,668 vs predicted 2,928 vehicles/hour
- 2018-06-10: Clouds, actual 2,272 vs predicted 2,580 vehicles/hour
- 2018-06-22: Clear, actual 3,904 vs predicted 3,823 vehicles/hour

Simulated incidents (to prove the alerting works): sensor_fault -> ALERT; temperature_shift -> ALERT.

**Recommended response to an ALERT:** first check the sensor and data feed (a fault looks like a sudden error spike with normal weather), then check for unusual events such as severe weather, closures or holidays. Retrain only if the change is real and persists for more than one batch; a one-off weather event should be documented, not used to retrain immediately.
