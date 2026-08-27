# Simple Matrix Factorization Recommendation Engine

A foundational collaborative filtering recommendation engine built from scratch using **NumPy**. This implementation uses Gradient Descent with L2 Regularization (Ridge) to predict missing ratings in a sparse user-item matrix.

## Features
* **Masked Training:** Ignores unrated items ($Y = 0$) so the model only optimizes against actual user feedback.
* **L2 Regularization:** Prevents overfitting by penalizing large weight values.
* **Gradient Scaling:** Divides gradients by the total number of active ratings to stabilize learning rates and prevent overflow (`NaN` errors).
* **Output Clipping:** Constrains final predictions to a realistic 1–5 rating scale.

## Requirements
* Python 3.x
* NumPy

## Code Structure
1. **Initialization:** Randomly initializes user matrix $X$ and item matrix $W$.
2. **Forward Pass:** Computes predictions via dot product ($X \cdot W^T$).
3. **Error Calculation:** Computes squared error filtered through the interaction mask.
4. **Backward Pass:** Updates latent features using gradient descent.

## Usage
Run the script directly to see the cost drop across epochs and view the final reconstructed rating matrix:
```bash
python main.py
