import numpy as np

class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.random(3)*1e-4 # 10 ^-4 to make the weights smaller
    print(f"Initail weights before Training: {self.weights}")
    self.epochs = epochs # No. of Epochs
    self.eta = eta # Learning Rate

  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights)
    return np.where(z > 0, 1, 0)
  
  def fit(self, X, y):
    self.X = X
    self.y = y

    X_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
    print(f"X with bias: \n{X_bias}")

    for epoch in range(self.epochs):
      print("--"*10)
      print(f"for epoch: {epoch}")
      print("--"*10)

      y_hat = self.activationFunction(X_bias, self.weights) # Foward Propagation

      print(f"Predicted value after forward pass: \n{y_hat}")

      self.error = self.y - y_hat
      print(f"error: \n{self.error}")
      self.weights = self.weights + self.eta * np.dot(X_bias.T, self.error) # Backward Propagation
      print(f"Updated weights after epoch ({epoch}/{self.epochs}): \n{self.weights}")
      print("#####"*10)
  
  def predict(self, X):
    X_bias = np.c_[X, -np.ones((len(X), 1))]
    return self.activationFunction(X_bias, self.weights)
  
  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"Total Loss: {total_loss}")
    return total_loss