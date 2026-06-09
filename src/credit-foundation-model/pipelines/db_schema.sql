-- =============================================================================
-- Credit Foundation Model — DuckDB Schema
-- =============================================================================
-- Bronze/Silver/Gold architecture:
--   static_loans       -> Silver: 1:1 static borrower/property attributes per loan
--   dynamic_performance -> Silver: 1:Many monthly performance snapshots per loan
--   gold_features      -> Gold View: ML-ready feature set with temporal labels
-- =============================================================================

-- Drop in reverse dependency order
DROP VIEW IF EXISTS gold_features;
DROP TABLE IF EXISTS dynamic_performance;
DROP TABLE IF EXISTS static_loans;

-- =============================================================================
-- SILVER LAYER: static_loans
-- Holds time-invariant origination and borrower attributes.
-- One row per loan_id (primary key).
-- =============================================================================
CREATE TABLE static_loans (
    loan_id                             VARCHAR         PRIMARY KEY,
    -- Origination info
    origination_year                    INTEGER,
    original_balance                    DOUBLE,
    legal_maturity_months               INTEGER,
    repayment_type                      VARCHAR,
    interest_only_flag                  BOOLEAN,
    rate_type                           VARCHAR,
    interest_payment_frequency          VARCHAR,
    principal_payment_frequency         VARCHAR,
    -- Borrower traits
    borrower_type                       VARCHAR,
    debtor_count                        INTEGER,
    loan_part_count                     INTEGER,
    employment_status                   VARCHAR,
    self_employed_flag                  BOOLEAN,
    loan_purpose                        VARCHAR,
    buy_to_let_flag                     BOOLEAN,
    borrower_annual_income              DOUBLE,
    loan_to_income                      DOUBLE,
    payment_due_to_income_pct           DOUBLE,
    scheduled_monthly_payment           DOUBLE,
    -- Property traits
    property_type                       VARCHAR,
    province                            VARCHAR,
    economic_region_nuts3               VARCHAR,
    country                             VARCHAR,
    construction_year                   INTEGER,
    occupancy                           VARCHAR,
    property_usage                      VARCHAR,
    -- Valuations
    original_market_value_at_origination DOUBLE,
    property_valuation_type             VARCHAR,
    oltomv_original                     DOUBLE,
    -- Guarantee / NHG
    nhg_flag                            BOOLEAN,
    guarantee_type                      VARCHAR,
    -- Energy performance
    epc_label                           VARCHAR,
    epc_issue_year                      INTEGER,
    primary_energy_demand_kwh_m2        DOUBLE,
    -- Construction deposit
    construction_deposit_flag           BOOLEAN,
    construction_deposit_pct            DOUBLE,
    construction_deposit_amount         DOUBLE,
    -- Transaction metadata
    transaction_name                    VARCHAR,
    esma_transaction_identifier         VARCHAR,
    originator_name                     VARCHAR,
    servicer_name                       VARCHAR,
    currency                            VARCHAR,
    closing_date                        DATE,
    maturity_date_proxy                 DATE,

    CHECK (original_balance > 0),
    CHECK (legal_maturity_months > 0),
    CHECK (loan_to_income IS NULL OR loan_to_income >= 0)
);

-- =============================================================================
-- SILVER LAYER: dynamic_performance
-- Holds monthly reporting snapshots for each loan.
-- Composite PK: (loan_id, reporting_date)
-- =============================================================================
CREATE TABLE dynamic_performance (
    loan_id                             VARCHAR         NOT NULL,
    reporting_date                      DATE            NOT NULL,
    -- Balance & rates
    current_balance                     DOUBLE,
    current_interest_rate_pct           DOUBLE,
    remaining_interest_fixed_period_months INTEGER,
    fixed_interest_period_end_in_months INTEGER,
    -- Temporal loan counters
    remaining_term_months               INTEGER,
    seasoning_months                    INTEGER,
    -- LTV metrics
    cltomv_current                      DOUBLE,
    cltimv_current                      DOUBLE,
    current_original_market_value       DOUBLE,
    indexed_market_value                DOUBLE,
    -- Arrears & default signals
    arrears_bucket                      VARCHAR,
    arrears_amount                      DOUBLE,
    days_past_due                       INTEGER,
    default_crr_flag                    BOOLEAN,
    performing_status                   VARCHAR,
    -- Special statuses
    foreclosure_flag                    BOOLEAN,
    forbearance_flag                    BOOLEAN,
    restructuring_flag                  BOOLEAN,
    -- Bucketed features (ordinal categories)
    balance_bucket                      VARCHAR,
    cltomv_current_bucket               VARCHAR,
    cltimv_current_bucket               VARCHAR,
    oltomv_original_bucket              VARCHAR,
    loan_to_income_bucket               VARCHAR,
    payment_due_to_income_pct_bucket    VARCHAR,
    construction_year_bucket            VARCHAR,

    PRIMARY KEY (loan_id, reporting_date),
    FOREIGN KEY (loan_id) REFERENCES static_loans(loan_id),
    CHECK (current_balance >= 0 OR current_balance IS NULL),
    CHECK (days_past_due >= 0 OR days_past_due IS NULL),
    CHECK (current_interest_rate_pct >= 0 OR current_interest_rate_pct IS NULL)
);

-- =============================================================================
-- GOLD LAYER: gold_features (View)
-- ML-ready feature set. Joins static and dynamic tables, computes:
--   - default_in_3m: 1 if loan defaults within next 3 reporting periods
--   - prepay_in_3m:  1 if loan prepays (balance hits 0) within next 3 periods
-- Uses LEAD() window functions partitioned by loan, ordered chronologically.
-- =============================================================================
CREATE VIEW gold_features AS
WITH labeled AS (
    SELECT
        -- Keys
        d.loan_id,
        d.reporting_date,

        -- --─ Static Features ----------------------------------------
        s.origination_year,
        s.original_balance,
        s.legal_maturity_months,
        s.repayment_type,
        s.interest_only_flag,
        s.rate_type,
        s.borrower_type,
        s.debtor_count,
        s.employment_status,
        s.self_employed_flag,
        s.loan_purpose,
        s.buy_to_let_flag,
        s.nhg_flag,
        s.province,
        s.economic_region_nuts3,
        s.construction_year,
        s.occupancy,
        s.property_type,
        s.property_usage,
        s.oltomv_original,
        s.original_market_value_at_origination,
        s.epc_label,
        s.primary_energy_demand_kwh_m2,
        s.loan_to_income,
        s.payment_due_to_income_pct,
        s.borrower_annual_income,

        -- --─ Dynamic Features ----------------------------------------
        d.current_balance,
        d.current_interest_rate_pct,
        d.remaining_term_months,
        d.seasoning_months,
        d.cltomv_current,
        d.cltimv_current,
        d.arrears_bucket,
        d.arrears_amount,
        d.days_past_due,
        d.default_crr_flag,
        d.performing_status,
        d.foreclosure_flag,
        d.forbearance_flag,
        d.restructuring_flag,
        d.balance_bucket,
        d.cltomv_current_bucket,

        -- --─ Temporal Label Engineering (LEAD) ----------------------
        -- Default in next 1-3 months: any CRR default flag fires
        LEAD(d.default_crr_flag, 1) OVER w AS default_t1,
        LEAD(d.default_crr_flag, 2) OVER w AS default_t2,
        LEAD(d.default_crr_flag, 3) OVER w AS default_t3,
        -- Balance goes to 0 (full prepayment) in next 3 months
        LEAD(d.current_balance, 1) OVER w AS balance_t1,
        LEAD(d.current_balance, 2) OVER w AS balance_t2,
        LEAD(d.current_balance, 3) OVER w AS balance_t3,
        -- Balance change (momentum feature)
        LAG(d.current_balance, 1) OVER w AS prev_balance

    FROM dynamic_performance d
    JOIN static_loans s ON d.loan_id = s.loan_id

    WINDOW w AS (PARTITION BY d.loan_id ORDER BY d.reporting_date)
)
SELECT
    loan_id,
    reporting_date,

    -- Static
    origination_year, original_balance, legal_maturity_months,
    repayment_type, interest_only_flag, rate_type, borrower_type,
    debtor_count, employment_status, self_employed_flag, loan_purpose,
    buy_to_let_flag, nhg_flag, province, economic_region_nuts3,
    construction_year, occupancy, property_type, property_usage,
    oltomv_original, original_market_value_at_origination,
    epc_label, primary_energy_demand_kwh_m2,
    loan_to_income, payment_due_to_income_pct, borrower_annual_income,

    -- Dynamic
    current_balance, current_interest_rate_pct, remaining_term_months,
    seasoning_months, cltomv_current, cltimv_current,
    arrears_bucket, arrears_amount, days_past_due,
    default_crr_flag, performing_status, foreclosure_flag,
    forbearance_flag, restructuring_flag,
    balance_bucket, cltomv_current_bucket,

    -- Derived momentum feature
    CASE WHEN prev_balance IS NOT NULL AND prev_balance > 0
         THEN (current_balance - prev_balance) / prev_balance
         ELSE NULL END AS balance_mom_1m,

    -- --─ TARGET LABELS ----------------------------------------------─
    CASE
        WHEN COALESCE(default_t1, FALSE) OR
             COALESCE(default_t2, FALSE) OR
             COALESCE(default_t3, FALSE)
        THEN 1 ELSE 0
    END AS default_in_3m,

    CASE
        WHEN COALESCE(balance_t1, 1) = 0 OR
             COALESCE(balance_t2, 1) = 0 OR
             COALESCE(balance_t3, 1) = 0
        THEN 1 ELSE 0
    END AS prepay_in_3m,

    -- Training split helper
    EXTRACT(YEAR FROM reporting_date)::INTEGER AS obs_year

FROM labeled
-- Exclude last 3 months (no forward labels available)
WHERE default_t1 IS NOT NULL
  AND default_t2 IS NOT NULL
  AND default_t3 IS NOT NULL;
