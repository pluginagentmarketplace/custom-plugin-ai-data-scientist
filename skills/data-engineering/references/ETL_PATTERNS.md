# ETL/ELT Design Patterns Guide

## ETL vs ELT Decision Matrix

```
┌─────────────────┬────────────────────────┬────────────────────────┐
│ Factor          │ ETL                    │ ELT                    │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Best For        │ On-premise, legacy     │ Cloud data warehouses  │
│ Transform       │ During load            │ After load             │
│ Performance     │ Limited by ETL server  │ Leverages DW compute   │
│ Flexibility     │ Schema on write        │ Schema on read         │
│ Cost            │ Infrastructure heavy   │ Compute on demand      │
│ Examples        │ Informatica, SSIS      │ dbt, Snowflake, BigQuery│
└─────────────────┴────────────────────────┴────────────────────────┘
```

## Common Data Pipeline Patterns

### 1. Batch Processing
```
Source → Extract (scheduled) → Transform → Load → Destination
         │                                        │
         └────────── Daily/Hourly ────────────────┘
```
**Use when:** Data freshness <24h acceptable, large volumes

### 2. Streaming (Real-time)
```
Source → Kafka → Spark Streaming → Transform → Load → Destination
         │                                            │
         └────────── Continuous ──────────────────────┘
```
**Use when:** Sub-second latency needed, event-driven

### 3. Lambda Architecture
```
                    ┌─► Batch Layer ──► Serving Layer ─┐
Source ──► Kafka ──┤                                   ├──► Query
                    └─► Speed Layer ──────────────────┘
```
**Use when:** Both real-time and historical analytics needed

### 4. Kappa Architecture
```
Source ──► Kafka ──► Streaming Processing ──► Serving ──► Query
```
**Use when:** Unified processing, simpler architecture preferred

## Data Modeling Patterns

### Star Schema
```
           ┌─── dim_product ───┐
           │                   │
dim_date ──┼── fact_sales ─────┼── dim_customer
           │                   │
           └─── dim_store ─────┘
```

### Snowflake Schema
```
dim_category ── dim_product ──┐
                              │
dim_date ────── fact_sales ───┼── dim_customer ── dim_region
                              │
dim_city ────── dim_store ────┘
```

### Data Vault
```
Hub (business key) ─── Link (relationships) ─── Satellite (attributes)
```

## Data Quality Dimensions

| Dimension | Description | Metric Example |
|-----------|-------------|----------------|
| Completeness | No missing values | Null rate < 1% |
| Accuracy | Correct values | Match rate > 99% |
| Consistency | Same across systems | Cross-system match |
| Timeliness | Data freshness | Latency < 1 hour |
| Uniqueness | No duplicates | Duplicate rate = 0% |
| Validity | Conforms to rules | Format compliance |

## Best Practices Checklist

### Pipeline Design
- [ ] Idempotent operations (re-runnable)
- [ ] Incremental processing where possible
- [ ] Proper error handling and retry logic
- [ ] Data lineage tracking
- [ ] Schema evolution support

### Performance
- [ ] Partition large tables appropriately
- [ ] Use columnar formats (Parquet, ORC)
- [ ] Optimize join order (small to large)
- [ ] Predicate pushdown enabled
- [ ] Appropriate parallelism

### Monitoring
- [ ] Pipeline SLAs defined
- [ ] Data quality metrics tracked
- [ ] Alerting on failures
- [ ] Cost monitoring
- [ ] Performance trending

## Technology Stack Recommendations

| Layer | Open Source | Cloud (AWS) | Cloud (GCP) |
|-------|-------------|-------------|-------------|
| Orchestration | Airflow, Dagster | MWAA, Step Functions | Cloud Composer |
| Streaming | Kafka, Flink | Kinesis, MSK | Pub/Sub, Dataflow |
| Batch | Spark | EMR, Glue | Dataproc |
| Warehouse | Trino, DuckDB | Redshift | BigQuery |
| Transform | dbt | dbt Cloud | dbt Cloud |
| Quality | Great Expectations | Deequ | - |
