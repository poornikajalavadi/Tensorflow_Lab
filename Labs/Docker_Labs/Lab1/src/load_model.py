import joblib
import os
import tempfile
import numpy as np

model_path = os.path.join(tempfile.gettempdir(), 'mnist_model.pkl')

if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.pkl')

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print('MNIST model loaded successfully!')
    print('Model type:', type(model))
    print('Model details:', model)

    # Test with a sample prediction
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    sample = X[0].reshape(1, -1)
    prediction = model.predict(sample)
    print(f'Sample prediction: {prediction[0]} (actual: {y[0]})')
else:
    print('mnist_model.pkl not found in this directory.')