# Linear Regression from Scratch in Python

Built only with NumPy libraries this simple linear regression using **gradient descent** showcases the foundational machine learning concept of fitting a line to a model.

This project was developed and tested using **Google Colab**, taking advantage of its fast setup, inline plotting, and interactive environment. Colab allowed me to iterate quickly while visualizing model behavior and debugging in real time.

## How It Works
The goal is to model a relationship between input \( x \) and output \( y \) by finding the optimal parameters \( W \) (slope) and \( B \) (intercept) such that:

$$\[
\hat{y} = Wx + B
\]$$

This is done by minimizing the **Mean Squared Error (MSE)** using **gradient descent**.


### Key Components

- `LinearRegressionGD` class with:
  - `fit(X, y)`: Trains the model using gradient descent to fit on the dataset
  - `predict(X)`: Returns predictions $\( \hat{y} \)$
  - `loss(X, y)`: Computes the MSE loss

- Gradient descent iteratively updates \( W \) and \( B \) using the following rules:
  $\[
  W := W - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W}, \quad
  B := B - \alpha \cdot \frac{\partial \mathcal{L}}{\partial B}
  \]$
  until the loss function converges.

- Tested using a sample dataset:  
  X = [1, 2, 3, 4, 5] and y = [3, 5, 7, 9, 11], Which equates to y = 2x + 1

### Results

After training:
- **Learned Weights**:  
  `W ≈ 2.0078`, `B ≈ 0.9717`
- **Predictions**:  
  `[2.98, 4.99, 7.00, 9.00, 11.01]`
- **Final Loss**:  
  `MSE ≈ 0.00015` — indicating near-perfect convergence

## What I Learned

- How linear regression models work under the hood
- The mechanics of gradient descent
- The role of step size (learning rate) and convergence

## Next Steps

- Support for multiple features (multivariate linear regression)
- Visualization of gradient descent steps and convergence
