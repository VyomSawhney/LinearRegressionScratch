# Multiple Linear Regression from Scratch in Python

Built only with NumPy libraries this simple linear regression using **gradient descent** showcases the foundational machine learning concept of fitting a line to a model.

This project was developed and tested using **Google Colab**, taking advantage of its fast setup, inline plotting, and interactive environment. Colab allowed me to iterate quickly while visualizing model behavior and debugging in real time.

## How It Works
The goal is to model a relationship between input \( x \) and output \( y \) by finding the optimal parameters \( W \) (slope) and \( B \) (intercept) such that:

$$\[
\hat{y} = Wx + B
\]$$

Given:
- $$\( X \in \mathbb{R}^{n \times d} \)$$: input feature matrix
- $$\( W \in \mathbb{R}^{d} \)$$: learned weights
- $$\( B \in \mathbb{R} \)$$: learned scalar bias

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


## What I Learned

- How linear regression models work under the hood
- The mechanics of gradient descent
- The role of step size (learning rate) and convergence

## Next Steps

- Visualization of gradient descent steps and convergence
