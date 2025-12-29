# Machine Learning Assets

Configuration templates and reusable assets for ML projects.

## Contents

| File | Type | Purpose |
|------|------|---------|
| `model_config.yaml` | YAML | Model configuration template |

## Usage

```python
import yaml

with open('model_config.yaml') as f:
    config = yaml.safe_load(f)

model_params = config['hyperparameters']
```
