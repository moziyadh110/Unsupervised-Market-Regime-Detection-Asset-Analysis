# Models Directory

Trained machine learning models are saved here.

## Generated Files

- `kmeans_model.pkl` - Trained K-means model
- `gmm_model.pkl` - Trained Gaussian Mixture Model
- `hmm_model.pkl` - Trained Hidden Markov Model
- `scaler.pkl` - StandardScaler for feature normalization

## Usage

```python
import joblib

# Load saved model
model = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Use for predictions
X_scaled = scaler.transform(new_data)
predictions = model.predict(X_scaled)
```

## Note

Model files are excluded from Git (see `.gitignore`).
Re-train models by running notebook 05.
