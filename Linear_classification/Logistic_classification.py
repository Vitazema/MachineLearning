#%%
import numpy as np
import matplotlib.pyplot as plt

n_pts = 100

bias = np.ones(n_pts)

random_x1_values = np.random.normal(12, 2, n_pts)
random_x2_values = np.random.normal(12, 2, n_pts)
top_region = np.array([random_x1_values, random_x2_values, bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T

all_points = np.vstack((top_region, bottom_region))
print(all_points)

def draw(x1, x2):
  line = plt.plot(x1, x2)
  plt.xlim((-5, 20))
  plt.ylim((-5, 20))
  plt.pause(0.0001)
  line[0].remove()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def calculate_error(line_parameters, points, y):
  m = points.shape[0]
  p = sigmoid(points * line_parameters)
  cross_entropy = -(1 / m) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
  return cross_entropy

def gradient_descent(line_parameters, points, y, learning_rate, epochs):
  m = points.shape[0]
  for i in range(epochs):
    p = sigmoid(points * line_parameters)
    gradient = (points.T * (p - y)) * (learning_rate/m)
    line_parameters = line_parameters - gradient
    w1 = line_parameters.item(0)
    w2 = line_parameters.item(1)
    b = line_parameters.item(2)
    # Calculate line parameters by using formula w1 * x1 + w2 * x2 + b = 0
    x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
    x2 = -b / w2 + x1 * (-w1 / w2)
    draw(x1, x2)
    print(calculate_error(line_parameters, points, y))



# Set zero-based line to start with
line_parameters = np.matrix([np.zeros(3)]).T
linear_combinaiton = all_points * line_parameters
probabilities = sigmoid(linear_combinaiton)
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)

print(probabilities)
print(all_points.shape)
print(line_parameters.shape)

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
gradient_descent(line_parameters, all_points, y, 0.13, 10000)
plt.show()

#print(calculate_error(line_parameters, all_points, y))

# %%
