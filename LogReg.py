import numpy as np

def sigmoid(x):
  result = 1 / (1 + np.exp(-x))
  return result

class LogisticRegression():
  
  def __init__(self, lr = 0.001 , n_iters = 1000):
    
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.biais = None
    
    
  def fit(self,X,y):
    n_samples , n_features = X.shape
    self.weights = np.zeros(n_features)
    self.biais = 0
    
    #Forward propagation
    
    for _ in range(self.n_iters):
      linear_pred = np.dot(X,self.weights) + self.biais
      predictions = sigmoid(linear_pred)
      
      #Backward propagation
      error = predictions - y
    
      dw = (1/n_samples) * np.dot(X.T, error) #Pour obtenir l'erreur moyenne pour les poids de chaque caractériqtique
      db = (1/n_samples) * np.sum(error)
      
      #Utiliser les gradients calculés pour un réajustement des poids et du biais
      self.weights -=  self.lr * dw
      self.weights -= self.lr * dw
    
  
      
  
  def predict(self,X):
    linear_pred = np.dot(X,self.weights) + self.biais
    y_pred = sigmoid(linear_pred)
    
    class_pred = [0 if y<=0.5 else 1 for y in y_pred]
    return(class_pred)
  
  
  