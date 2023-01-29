# Activation Functions
import math

def sigmoid(x):
  eq = 1/(1+math.exp(-x))
  return eq


def tanh(x):
  eq=(math.exp(x)-math.exp(-x))/ (math.exp(x)+math.exp(-x))
  return eq


def relu(x):
  return max(0,x)


def leaky_relu(x):
  return max(0.1*x,x)
