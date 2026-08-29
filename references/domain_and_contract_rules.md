# NBA Domain Reference: Rookie Contracts, Longevity & Decision Rules

**Author:** Aryan Goel (Student ID: `26040826`)  
**Project:** Applied Machine Learning (AT1) — Group 38  

---

## 1. The NBA Collective Bargaining Agreement (CBA) Rookie Scale
Under the NBA Collective Bargaining Agreement:
1. **First-Round Draft Picks (Picks 1–30):**
   * Receive mandatory **4-year rookie-scale contracts**.
   * **Years 1 & 2:** 100% fully guaranteed salary.
   * **Year 3:** Team Option (exercised prior to the start of season 2).
   * **Year 4:** Team Option (exercised prior to the start of season 3).
2. **Second-Round Picks & Undrafted Free Agents:**
   * Non-guaranteed or partially guaranteed minimum contracts, two-way contracts, or NBA G-League assignments.

---

## 2. Why $\ge 3$ Seasons is the Optimal Longevity Boundary
* **Meeting Minimum Valuation:** A player who lasts at least 3 seasons in the NBA had their team option picked up (or earned a standard extension / second contract from another team). This proves that the player was economically and competitively viable at the highest level of professional basketball.
* **Separating Busted Investments from Core Assets:** Players who wash out in years 1 or 2 represent wasted draft picks and dead salary cap capital ($5M–$15M).

---

## 3. Machine Learning Metric Justification: Why AUPRC Over AUROC
* **Severe Class Imbalance (~3.74% Positive Rate):**
  * In a dataset where 96.26% of instances are negative, a naive model predicting 0 for all instances achieves an AUROC of 0.50 and accuracy of 96.26%.
  * AUROC incorporates True Negatives ($TN$) in its denominator (False Positive Rate = $FP / (FP + TN)$). Because $TN$ is massive, huge swings in false positives barely affect the AUROC score.
  * **AUPRC (Average Precision)** focuses strictly on Precision ($TP / (TP + FP)$) and Recall ($TP / (TP + FN)$). It heavily penalizes false positives, directly ensuring that prospects ranked in the top percentiles are genuinely high-conviction NBA players.
