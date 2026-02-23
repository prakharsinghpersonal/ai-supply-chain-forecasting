-- ============================================================================
-- SupplyGuard — Optimized Analytical SQL Views
-- Complex multi-source aggregation for sub-second dashboard queries
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Carrier Performance Summary (materialised for sub-second response)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ANALYTICS.CARRIER_PERFORMANCE AS
SELECT
    s.carrier,
    COUNT(*)                                                AS total_shipments,
    SUM(CASE WHEN s.status = 'DISRUPTED' THEN 1 ELSE 0 END)
                                                            AS disrupted_count,
    ROUND(disrupted_count / NULLIF(total_shipments, 0), 4)  AS disruption_rate,
    AVG(s.risk_score)                                       AS avg_risk_score,
    MEDIAN(s.weight_kg)                                     AS median_weight_kg,
    AVG(DATEDIFF('day', s.ship_date, s.expected_delivery))  AS avg_transit_days,
    MIN(s.ship_date)                                        AS earliest_shipment,
    MAX(s.ship_date)                                        AS latest_shipment
FROM RAW.SHIPMENTS s
GROUP BY s.carrier;

-- ---------------------------------------------------------------------------
-- 2. Route-Level Risk Aggregation
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ANALYTICS.ROUTE_RISK AS
SELECT
    s.origin,
    s.destination,
    s.carrier,
    COUNT(*)                                                AS volume,
    AVG(s.risk_score)                                       AS avg_risk,
    STDDEV(s.risk_score)                                    AS risk_volatility,
    SUM(CASE WHEN s.status = 'DISRUPTED' THEN 1 ELSE 0 END)
                                                            AS disruptions,
    ROUND(disruptions / NULLIF(volume, 0), 4)               AS disruption_rate
FROM RAW.SHIPMENTS s
GROUP BY s.origin, s.destination, s.carrier
HAVING volume >= 10
ORDER BY avg_risk DESC;

-- ---------------------------------------------------------------------------
-- 3. Daily Operational Dashboard View
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ANALYTICS.DAILY_OPERATIONS AS
SELECT
    DATE_TRUNC('day', s.ship_date)                          AS report_date,
    COUNT(*)                                                AS shipments_today,
    SUM(CASE WHEN s.status = 'DISRUPTED' THEN 1 ELSE 0 END)
                                                            AS disruptions_today,
    AVG(s.risk_score)                                       AS avg_risk_today,
    COUNT(DISTINCT s.carrier)                               AS active_carriers,
    COUNT(DISTINCT s.origin)                                AS active_origins,
    SUM(s.weight_kg)                                        AS total_weight_kg,
    -- 7-day rolling disruption rate
    AVG(SUM(CASE WHEN s.status = 'DISRUPTED' THEN 1 ELSE 0 END))
        OVER (ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
                                                            AS rolling_7d_disruption_avg
FROM RAW.SHIPMENTS s
GROUP BY report_date
ORDER BY report_date DESC;

-- ---------------------------------------------------------------------------
-- 4. Supplier Reliability Scorecard
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ANALYTICS.SUPPLIER_SCORECARD AS
SELECT
    sp.supplier_id,
    sp.supplier_name,
    sp.region,
    AVG(sp.on_time_rate)                                    AS avg_on_time_rate,
    AVG(sp.defect_rate)                                     AS avg_defect_rate,
    AVG(sp.avg_lead_time_days)                              AS avg_lead_time,
    COUNT(DISTINCT sp.report_date)                          AS reporting_periods,
    -- Composite reliability score (higher is better)
    ROUND(
        (AVG(sp.on_time_rate) * 0.5)
        + ((1 - AVG(sp.defect_rate)) * 0.3)
        + ((1 - LEAST(AVG(sp.avg_lead_time_days) / 30.0, 1)) * 0.2),
        4
    ) AS reliability_score
FROM RAW.SUPPLIERS sp
GROUP BY sp.supplier_id, sp.supplier_name, sp.region
ORDER BY reliability_score DESC;
