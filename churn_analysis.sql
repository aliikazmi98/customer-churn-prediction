-- ── churn_analysis.sql ─────────────────────────────────────────────────────
-- Run against churn.db (SQLite) after churn_model.py has been executed.

-- 1. Overall churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(Churn) AS total_churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS overall_churn_pct
FROM customers;

-- 2. Churn by contract type
SELECT
    Contract,
    COUNT(*) AS customers,
    SUM(Churn) AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Churn by tenure group
SELECT
    tenure_group,
    COUNT(*) AS customers,
    SUM(Churn) AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY tenure_group;

-- 4. Churn by monthly charge tier
SELECT
    charge_bin,
    COUNT(*) AS customers,
    SUM(Churn) AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY charge_bin
ORDER BY charge_bin;

-- 5. Churn by internet service type
SELECT
    InternetService,
    COUNT(*) AS customers,
    SUM(Churn) AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- 6. High-risk micro-segment: Month-to-month + Fiber optic
SELECT
    tenure_group,
    COUNT(*) AS customers,
    SUM(Churn) AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
WHERE Contract = 'Month-to-month'
  AND InternetService = 'Fiber optic'
GROUP BY tenure_group
ORDER BY churn_rate_pct DESC;

-- 7. Revenue at risk (churned customers' monthly charges)
SELECT
    ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_rev_lost,
    ROUND(SUM(CASE WHEN Churn = 0 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_rev_retained,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_rev
FROM customers;

-- 8. Customers recoverable at medium/high risk (predicted by ML — join on customerID)
-- Run after churn_predictions.csv is imported as table "predictions"
-- SELECT p.risk_tier, COUNT(*) AS count, ROUND(AVG(c.MonthlyCharges), 2) AS avg_charges
-- FROM predictions p JOIN customers c USING(customerID)
-- WHERE p.risk_tier IN ('Medium', 'High') AND c.Churn = 0
-- GROUP BY p.risk_tier;
