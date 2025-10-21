# Ensemble Strategy Guide

## Overview

The Ensemble feature allows you to automatically combine multiple iteration solutions (pretrained, traditional ML, custom NN) into a single, potentially more powerful ensemble model. This feature is inspired by Google's ADK machine-learning-engineering ensemble approach.

## How It Works

### 1. Multi-Iteration Pipeline

When you run the multi-iteration mode:

```bash
python run_multi_iteration.py -i ./your_dataset
```

The system will:
1. **Run 3 iterations** with different approaches:
   - `iteration_1_pretrained`: HuggingFace transformers
   - `iteration_2_traditional`: XGBoost, LightGBM, CatBoost
   - `iteration_3_custom_nn`: Custom PyTorch models

2. **Compare iterations** using LLM-based intelligent comparison

3. **Create ensemble strategy** (NEW!) combining successful iterations

### 2. Ensemble Agent

The `EnsembleAgent` analyzes all successful iterations and:

- **Analyzes each solution**: Reviews code, metrics, and approach
- **Proposes ensemble method**: Suggests specific techniques like:
  - Weighted averaging
  - Stacking (meta-model)
  - Voting (for classification)
  - Blending
  - Custom hybrid approaches
- **Provides implementation plan**: Detailed steps to combine predictions

### 3. Output Structure

After running multi-iteration mode:

```
runs/run_<timestamp>/
├── iteration_1_pretrained/
│   ├── submission.csv
│   └── states/final_executable_code.py
├── iteration_2_traditional/
├── iteration_3_custom_nn/
├── llm_comparison_results.json      # Comparison analysis
├── ensemble_strategy.json           # 🆕 Ensemble plan
├── ensemble_code.py                 # 🆕 Ensemble implementation (if generated)
├── ensemble/                        # 🆕 Ensemble workspace
│   ├── input/                       # Data files
│   └── ensemble_code.py             # Executable ensemble code
└── final_submission/                # Best single solution
    └── submission.csv
```

## Key Files

### ensemble_strategy.json

Contains the LLM-generated ensemble strategy:

```json
{
  "status": "success",
  "plan": "Detailed natural language plan...",
  "code": "# Optional: Python implementation code",
  "full_response": "Complete LLM response"
}
```

### ensemble_code.py

A ready-to-execute Python script that implements the ensemble strategy. You can:

1. **Review the code** to understand the ensemble approach
2. **Execute it directly**:
   ```bash
   python runs/run_<timestamp>/ensemble_code.py
   ```
3. **Modify it** to fine-tune weights or methods

## Ensemble Strategies

The LLM can suggest various strategies based on your data:

### 1. Weighted Averaging
Best for regression or when models have similar confidence levels.
```python
final_pred = w1 * pred1 + w2 * pred2 + w3 * pred3
```

### 2. Stacking
Uses a meta-model to learn optimal combination.
```python
meta_features = [pred1, pred2, pred3]
final_pred = meta_model.predict(meta_features)
```

### 3. Voting
Best for classification tasks.
```python
final_pred = majority_vote([pred1, pred2, pred3])
```

### 4. Blending
Uses validation set to learn combination weights.

## Configuration

### Enable/Disable Ensemble

Add to your config YAML:

```yaml
# Enable ensemble (default: enabled in multi-iteration mode)
enable_ensemble: true

# Ensemble parameters
ensemble_max_iterations: 3  # For iterative refinement
```

### Minimum Requirements

- At least **2 successful iterations** are required
- Each iteration must have valid predictions
- Iterations should use different approaches for diversity

## Advanced Usage

### Manual Ensemble Implementation

If you want more control, you can:

1. **Get the ensemble plan**:
   - Read `ensemble_strategy.json` for the strategy
   
2. **Load predictions** from each iteration:
   ```python
   pred1 = pd.read_csv('iteration_1_pretrained/submission.csv')
   pred2 = pd.read_csv('iteration_2_traditional/submission.csv')
   pred3 = pd.read_csv('iteration_3_custom_nn/submission.csv')
   ```

3. **Apply your own ensemble logic**:
   ```python
   # Example: Simple weighted average
   ensemble = 0.4 * pred1 + 0.3 * pred2 + 0.3 * pred3
   ```

### Iterative Refinement (Future)

The `EnsembleImplementationAgent` can iteratively refine the ensemble:

```python
# In future versions
from src.iML.agents import EnsembleImplementationAgent

impl_agent = EnsembleImplementationAgent(config, manager, llm_config)
result = impl_agent(
    ensemble_plan=plan,
    iteration_codes=codes
)
```

## Best Practices

### 1. Diversity is Key
- Use iterations with **different algorithms** (e.g., tree-based + neural networks)
- Different feature engineering approaches
- Different validation strategies

### 2. Validate Ensemble Performance
- Always validate on held-out test set
- Compare ensemble vs. best single model
- Check for overfitting

### 3. Weight Selection
- Start with equal weights (1/n for each model)
- Use validation performance to adjust weights
- Consider model confidence/uncertainty

### 4. Production Deployment
- Ensemble increases computational cost
- Consider inference time requirements
- May need to deploy multiple model servers

## Examples

### Example 1: Weighted Average (Regression)

```python
# Ensemble strategy for house price prediction
import pandas as pd
import numpy as np

# Load predictions from each iteration
pred_pretrained = pd.read_csv('iteration_1_pretrained/submission.csv')
pred_traditional = pd.read_csv('iteration_2_traditional/submission.csv')
pred_custom = pd.read_csv('iteration_3_custom_nn/submission.csv')

# Weights based on validation RMSE (lower is better)
# pretrained: RMSE=0.15, traditional: RMSE=0.12, custom: RMSE=0.18
weights = np.array([1/0.15, 1/0.12, 1/0.18])
weights = weights / weights.sum()  # Normalize

# Ensemble predictions
ensemble_pred = (
    weights[0] * pred_pretrained['target'] +
    weights[1] * pred_traditional['target'] +
    weights[2] * pred_custom['target']
)

# Save
submission = pd.DataFrame({
    'id': pred_pretrained['id'],
    'target': ensemble_pred
})
submission.to_csv('ensemble_submission.csv', index=False)
```

### Example 2: Soft Voting (Classification)

```python
# Ensemble strategy for multi-class classification
import pandas as pd

# Load probability predictions from each iteration
pred1 = pd.read_csv('iteration_1_pretrained/submission.csv')
pred2 = pd.read_csv('iteration_2_traditional/submission.csv')
pred3 = pd.read_csv('iteration_3_custom_nn/submission.csv')

# Average probabilities (soft voting)
ensemble_probs = (pred1.iloc[:, 1:] + pred2.iloc[:, 1:] + pred3.iloc[:, 1:]) / 3

# Get final class predictions
final_class = ensemble_probs.idxmax(axis=1)

# Save
submission = pd.DataFrame({
    'id': pred1['id'],
    'class': final_class
})
submission.to_csv('ensemble_submission.csv', index=False)
```

## Troubleshooting

### Issue: "Not enough successful iterations"
**Solution**: Ensure at least 2 iterations complete successfully. Check individual iteration logs.

### Issue: "Code file not found"
**Solution**: Make sure iterations completed and generated `final_executable_code.py`.

### Issue: "Different prediction formats"
**Solution**: Ensemble code needs to handle format differences. The LLM should detect this, but you may need to manually align formats.

### Issue: "Ensemble performs worse than best model"
**Solution**: 
- Check if models are too similar (lack diversity)
- Validate weight selection
- Ensure no data leakage in ensemble validation

## References

- **ADK Machine Learning Engineering**: [Google ADK Samples](https://github.com/google/adk-samples)
- **Kaggle Ensemble Guide**: [Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/)
- **Stacking**: [Wolpert's Stacked Generalization](http://machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf)

## Future Enhancements

- [ ] Automatic ensemble code execution
- [ ] Iterative ensemble refinement with validation feedback
- [ ] Cross-validation based weight optimization
- [ ] Automatic hyperparameter tuning for meta-models
- [ ] Support for heterogeneous output formats
- [ ] Integration with AutoML hyperparameter search

