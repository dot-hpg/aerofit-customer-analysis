# Aerofit: Business Insights & Recommendations

*Submission text for the Scaler "Aerofit - Descriptive Statistics & Probability" case study. Ready to paste into the text editor.*

**Notebook PDF:** (paste your public Google Drive link here)

---

## Problem
Aerofit wants to know who buys each of its three treadmills (KP281 at $1,500, KP481 at $1,750 and KP781 at $2,500) so it can recommend the right model to new customers. We analysed 180 purchases from the last three months.

## Data at a glance
* 180 rows and 9 columns. No missing values and no duplicates. `Product`, `Gender` and `MaritalStatus` were converted to the `category` dtype.
* **Product mix:** KP281 44.4%, KP481 33.3%, KP781 22.2%. Revenue is far more even: 36.9%, 32.3% and 30.8% of $325K.
* **Ranges:** Age 18 to 50 (median 26), Education 12 to 21 years (median 16), Usage 2 to 7 days a week (median 3), Fitness 1 to 5 (median 3), Income $29.6K to $104.6K (median $50.6K), Miles 21 to 360 a week (median 94).
* Age, Income and Miles are right-skewed (mean > median). Their outliers (Income 10.6%, Miles 7.2%) are real premium customers, not errors, so every record was kept and the profiles use medians.

## Customer profiles

| | KP281 "Starter" | KP481 "Progressing runner" | KP781 "Serious athlete" |
|---|---|---|---|
| Buyers | 80 (44%) | 60 (33%) | 40 (22%) |
| Median age | 26 | 26 | 27 |
| Men : women | 50 : 50 | 52 : 48 | 82.5 : 17.5 |
| Median income | $46.6K | $49.5K | $76.6K |
| Median education | 16 yrs | 16 yrs | 18 yrs |
| Use per week | 3 days | 3 days | 5 days |
| Fitness | 3 / 5 | 3 / 5 | 5 / 5 |
| Miles per week | 85 | 85 | 160 |

## Key probabilities
* **Marginal:** P(KP281) = 44.4%, P(KP481) = 33.3%, P(KP781) = 22.2%.
* **Gender:** P(KP781 | Male) = 31.7% vs P(KP781 | Female) = 9.2%. P(KP281 | Female) = 52.6%. P(Male | KP781) = 82.5%.
* **Marital status:** P(KP781 | Partnered) = 21.5% vs P(KP781 | Single) = 23.3%, so there is practically no difference.
* **Income:** P(KP781 | Income > $80K) = 100%. P(KP781 | Income < $40K) = 0%. P(KP281 | Income < $40K) = 71.9%.
* **Fitness:** P(KP781 | Fitness 4-5) = 65.5%. P(KP781 | Fitness ≤ 3) = 3.2%.
* **Usage:** P(KP781 | 5-7 days) = 80.8%. P(KP781 | 2-3 days) = 1%.
* **Miles:** P(KP781 | > 150 miles) = 82.1%.
* **Chi-square tests:** Income, Fitness, Usage, Miles and Gender are significantly linked to the product purchased (p < 0.05). Age group and Marital status are not.

## Insights
1. **Income is the biggest driver** of which treadmill a customer buys. Above $80K every customer bought the KP781, and below $40K nobody did.
2. **Fitness, usage and miles move together** (correlations of 0.67 to 0.79). Together they describe the KP781 buyer: a very fit person who runs 5 days a week and about 160 miles.
3. **KP281 and KP481 buyers are almost identical** in age, income, fitness and usage. The extra $250 is the main thing separating them.
4. **Women are under-served at the premium end.** Only 7 of 76 women bought the KP781, yet those who did are as fit and well-off as the men. This points to a marketing gap, not a budget gap.
5. **Age and marital status hardly change the model choice.** They help decide where to advertise, not what to recommend.
6. **One KP781 sale brings in about 1.7 times the revenue of a KP281 sale**, so moving a few customers up the range has real impact.

## Recommendations
1. **Ask three simple questions at the counter:** "How many days a week will you use it?", "How fit do you feel, 1 to 5?" and "What budget suits you?"
   * 5+ days, fitness 4 to 5, or budget above $80K income: show the **KP781**.
   * 2 to 3 days, fitness 3 or below, or income under $40K: show the **KP281**.
   * Everyone in between: show the **KP281 and KP481 side by side**.
2. **Bring the KP781 to women.** Use female athletes and partner with women's running clubs and gyms.
3. **Make the KP781 easier to afford.** Offer EMI or financing and trade-in deals to fit, frequent runners earning $40K to $80K.
4. **Give the KP481 a clear reason to choose it.** Call it "the treadmill you won't outgrow" and bundle extras (mat, heart-rate strap, training plan) so the $250 step-up feels worth it.
5. **Keep the KP281 the easy first choice.** Lead with price, size and simplicity for students, first-job buyers and couples.
6. **Sell to households.** 59% of buyers are partnered, so run "fitness for two" and family offers.
7. **Upgrade loyal users.** Contact KP281 owners after 6 to 12 months with a trade-in offer for the KP781.
8. **Collect more data** (store, region, reason for buying) to sharpen these profiles.
