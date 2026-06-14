# 📈 Implementing Lasso Regression from Scratch in Python

## 🧮 Complete Implementation of Lasso Regression Algorithm with Gradient Descent

A **from-scratch implementation** of the Lasso (Least Absolute Shrinkage and Selection Operator) Regression algorithm using pure Python and NumPy. This educational project demonstrates the fundamental mathematics behind L1-regularized linear regression, complete with gradient descent optimization and practical salary prediction application.

## 🎯 Project Overview

This project implements Lasso Regression **without relying on machine learning libraries** like scikit-learn. It provides a deep understanding of:
- The mathematical foundations of Lasso Regression
- L1 regularization and its effect on feature selection
- Gradient Descent optimization with regularization
- Handling absolute value in gradient computation
- Model training and prediction mechanics

## 📚 Mathematical Foundations

### 🧠 Lasso Regression Equation
```
Y = wX + b
```
Where:
- **Y**: Dependent Variable (Target)
- **X**: Independent Variable (Feature)
- **w**: Weight parameter
- **b**: Bias parameter

### ⚙️ Gradient Descent for Lasso Regression

Gradient Descent is an optimization algorithm used for minimizing the loss function by iteratively updating parameters:

**Parameter Update Rules:**
```
w = w - α * dw
b = b - α * db
```

**Learning Rate (α)**: Tuning parameter that determines step size at each iteration while moving toward minimum loss.

### 🔍 Gradient Computation with L1 Regularization

The key challenge in Lasso Regression is handling the non-differentiable L1 penalty. The gradient for weights depends on the sign of the weight:

**Gradient for Weights:**
```
if w[i] > 0:
    dw[i] = (-(2 * X[:,i].dot(Y - Y_prediction)) + λ) / m
else:
    dw[i] = (-(2 * X[:,i].dot(Y - Y_prediction)) - λ) / m
```

**Gradient for Bias:**
```
db = -2 * sum(Y - Y_prediction) / m
```

Where:
- **λ (lambda_parameter)**: Regularization parameter controlling shrinkage
- **m**: Number of training examples
- **dw**: Gradient with respect to weights
- **db**: Gradient with respect to bias

## 🛠️ Technical Implementation

### 🏗️ Custom Lasso Regression Class

#### Class Structure:
```python
class Lasso_Regression():
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        # Initialize hyperparameters
    
    def fit(self, X, Y):
        # Training function with gradient descent
    
    def update_weights(self):
        # Gradient descent weight updates with L1 regularization
    
    def predict(self, X):
        # Make predictions using learned parameters
```

#### Core Methods:
1. **`__init__`**: Initialize learning rate, iterations, and regularization parameter
2. **`fit`**: Train model using gradient descent with L1 penalty
3. **`update_weights`**: Compute gradients considering sign of weights and update parameters
4. **`predict`**: Generate predictions using learned weights and bias

### 🔄 Gradient Descent Implementation
```python
def update_weights(self):
    # Calculate predictions
    Y_prediction = self.predict(self.X)
    
    # Initialize gradient array for weights
    dw = np.zeros(self.n)
    
    # Compute gradients for each weight considering sign
    for i in range(self.n):
        if self.w[i] > 0:
            dw[i] = (-(2 * (self.X[:,i]).dot(self.Y - Y_prediction)) + 
                     self.lambda_parameter) / self.m
        else:
            dw[i] = (-(2 * (self.X[:,i]).dot(self.Y - Y_prediction)) - 
                     self.lambda_parameter) / self.m
    
    # Compute gradient for bias
    db = -2 * np.sum(self.Y - Y_prediction) / self.m
    
    # Update parameters
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db
```

## 📊 Dataset & Application

### 💰 Salary Prediction Problem
**Dataset**: Salary vs. Years of Experience
- **Features**: Years of work experience
- **Target**: Salary amount
- **Samples**: 30 employee records

### 🧪 Model Training Results
```
Hyperparameters Used:
- Learning Rate: 0.02
- Number of Iterations: 1000
- Lambda Parameter: 200
```

### 📈 Performance Metrics
| Metric | Custom Lasso | Scikit-learn Lasso |
|--------|--------------|-------------------|
| R² Score | ~0.84 | ~0.84 |
| Mean Absolute Error | ~7600 | ~7600 |

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pip install numpy pandas matplotlib scikit-learn
```

### Installation & Usage

1. **Clone the repository**:
```bash
git clone https://github.com/ManeKarthikeya/Implementing-Lasso-Regression-from-Scratch.git
cd Implementing-Lasso-Regression-from-Scratch
```

2. **Run the salary prediction system**:
```bash
python implementing_lasso_regression_from_scratch.py
```

3. **The model will automatically**:
   - Load and preprocess the salary dataset
   - Train the Lasso Regression model
   - Display predictions and evaluation metrics
   - Compare results with scikit-learn's implementation

### Manual Usage Example
```python
# Import custom Lasso Regression class
from lasso_regression import Lasso_Regression

# Create and train model
model = Lasso_Regression(learning_rate=0.02, 
                         no_of_iterations=1000, 
                         lambda_parameter=200)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
r2 = metrics.r2_score(y_test, predictions)
mae = metrics.mean_absolute_error(y_test, predictions)
```

## 📁 Project Structure

```
Lasso-Regression-from-Scratch/
├── implementing_lasso_regression_from_scratch.py  # Main implementation
├── salary_data.csv                                # Salary dataset
├── requirements.txt                               # Python dependencies (add on you'r own)
└── README.md                                     # Project documentation
```

## 🧪 Model Validation

### 🔍 Comparison with Scikit-learn
The implementation is validated against scikit-learn's Lasso regression:
```python
from sklearn.linear_model import Lasso
sk_model = Lasso()
sk_model.fit(X_train, Y_train)
sk_predictions = sk_model.predict(X_test)

# Compare metrics
print("Custom Lasso R²:", r2_score(y_test, custom_predictions))
print("Sklearn Lasso R²:", r2_score(y_test, sklearn_predictions))
```

### 📊 Evaluation Metrics
- **R² Score (Coefficient of Determination)**: Measures proportion of variance explained by the model
- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and actual values

## 🎯 Educational Value

### 🎓 Learning Objectives
1. **Understanding Lasso Regression Mathematics**
   - L1 regularization formulation
   - Handling non-differentiable points in optimization
   - Feature selection property of Lasso

2. **Implementing Gradient Descent with Regularization**
   - Parameter update mechanics with penalty term
   - Sign-based gradient computation
   - Impact of lambda on weight shrinkage

3. **Custom Algorithm Development**
   - Object-oriented programming for ML
   - Numerical computation with NumPy
   - Model evaluation and validation techniques

### 🔍 Key Insights
- **Regularization Effect**: Higher lambda values lead to more weight shrinkage
- **Feature Selection**: Lasso can drive some weights exactly to zero
- **Bias-Variance Tradeoff**: Lambda controls model complexity
- **Convergence**: Proper learning rate ensures stable optimization


## ✅ Advantages of Lasso Regression

- **Feature Selection**: Automatically selects relevant features
- **Interpretability**: Produces sparse models
- **Regularization**: Prevents overfitting
- **Handles Multicollinearity**: Can select among correlated features
