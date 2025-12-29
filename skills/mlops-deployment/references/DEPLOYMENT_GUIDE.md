# MLOps Deployment Guide

## Deployment Strategy Selection

```
Model Complexity / Traffic Volume
         │
         ▼
┌────────────────────────────────────────────────────────┐
│                    Low Traffic                          │
│    ┌─────────┐                     ┌─────────┐         │
│    │ Simple  │ ──► Flask/FastAPI  │ Complex │──► Docker │
│    │ Model   │     on VM           │ Model   │   on VM   │
│    └─────────┘                     └─────────┘         │
├────────────────────────────────────────────────────────┤
│                    High Traffic                         │
│    ┌─────────┐                     ┌─────────┐         │
│    │ Simple  │ ──► Kubernetes     │ Complex │──► K8s +  │
│    │ Model   │     with HPA        │ Model   │   Triton  │
│    └─────────┘                     └─────────┘         │
└────────────────────────────────────────────────────────┘
```

## Deployment Options Comparison

| Option | Complexity | Scalability | Cost | Best For |
|--------|------------|-------------|------|----------|
| Flask on VM | Low | Low | Low | Prototypes |
| FastAPI + Docker | Medium | Medium | Medium | Small prod |
| Kubernetes | High | High | High | Production |
| AWS SageMaker | Medium | High | Variable | AWS users |
| GCP Vertex AI | Medium | High | Variable | GCP users |
| Triton Server | High | Very High | High | GPU inference |
| TensorFlow Serving | Medium | High | Medium | TF models |

## Docker Best Practices

```dockerfile
# Multi-stage build for smaller images
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

FROM python:3.10-slim

# Security: run as non-root
RUN useradd -m appuser
USER appuser
WORKDIR /app

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Kubernetes Checklist

```markdown
Pre-deployment:
□ Docker image built and pushed
□ Resource limits defined
□ Health checks configured
□ ConfigMaps/Secrets created
□ Service account configured

Deployment:
□ Deployment manifest applied
□ Service (LoadBalancer/ClusterIP) created
□ HPA configured
□ PodDisruptionBudget set
□ Network policies defined

Post-deployment:
□ Health endpoints verified
□ Metrics being collected
□ Logs flowing to aggregator
□ Alerts configured
□ Rollback procedure tested
```

## Monitoring Metrics

| Metric | Type | Alert Threshold |
|--------|------|-----------------|
| Latency P50 | Histogram | > 100ms |
| Latency P99 | Histogram | > 500ms |
| Error Rate | Counter | > 1% |
| Request Rate | Counter | Anomaly |
| CPU Usage | Gauge | > 80% |
| Memory Usage | Gauge | > 85% |
| Model Staleness | Gauge | > 24h |

## CI/CD Pipeline Stages

```
1. Lint & Test
   └── Code quality checks
   └── Unit tests
   └── Model validation tests

2. Build
   └── Docker image build
   └── Security scan
   └── Push to registry

3. Deploy Staging
   └── Deploy to staging
   └── Integration tests
   └── Performance tests

4. Deploy Production
   └── Canary deployment (10%)
   └── Monitor metrics
   └── Full rollout
   └── Smoke tests
```

## Rollback Strategy

```yaml
# Automatic rollback triggers
rollback:
  triggers:
    - error_rate > 5%
    - latency_p99 > 2s
    - health_check_failures > 3

  procedure:
    1. Detect issue (monitoring)
    2. Stop traffic to new version
    3. Scale up old version
    4. Redirect traffic
    5. Scale down new version
    6. Post-mortem analysis
```

## Security Checklist

```markdown
□ API authentication (API key, JWT, OAuth)
□ Rate limiting configured
□ Input validation/sanitization
□ HTTPS/TLS enabled
□ Secrets in secure vault
□ Container security scan
□ Network policies
□ Audit logging enabled
□ RBAC configured
```
