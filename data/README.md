# Data dictionary: `aerofit_treadmill.csv`

180 rows × 9 columns. Each row is one customer who bought an Aerofit treadmill in the prior three months. There are no missing values and no duplicate rows.

| Column | Type | Description |
|---|---|---|
| Product | categorical | Treadmill purchased: KP281 (entry, $1,500), KP481 (mid-level, $1,750), KP781 (advanced, $2,500) |
| Age | integer | Age in years (18 to 50) |
| Gender | categorical | Male / Female |
| Education | integer | Years of education (12 to 21) |
| MaritalStatus | categorical | Single / Partnered |
| Usage | integer | Average number of times the customer plans to use the treadmill each week (2 to 7) |
| Income | integer | Annual income in $ (29,562 to 104,581) |
| Fitness | integer | Self-rated fitness, 1 (poor shape) to 5 (excellent shape) |
| Miles | integer | Average number of miles the customer expects to walk/run each week (21 to 360) |

**Source:** Scaler Data Science & ML program, Aerofit business case
(`https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/125/original/aerofit_treadmill.csv?1639992749`).
MD5 of this copy: `3a2ad7126560cc76914669fd8dcb6775`.
