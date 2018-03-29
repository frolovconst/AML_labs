import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# Start session
sess = tf.Session()

# Generate non-linear moon-shape like data
(x_vals, y_vals) = datasets.make_moons(n_samples=350, noise=.20, random_state=0)
y_vals = np.array([1 if y == 1 else -1 for y in y_vals])
class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]

# TODO START YOUR CODE HERE
# Initialize placeholders
x_data = None
y_target = None

# Create variables for svm
b = None # intercept (bias)
A = None # theta

# Build an SVM Model


# Define loss function
loss = None

# Define prediction function
prediction = None

# Define optimizer and train step
my_opt = None
train_step = None

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# To store values of the loss function at each iteration
losses = []

# Save graph to the file to inspect with tensorboard
writer = tf.summary.FileWriter("./logs/", tf.get_default_graph())

# END YOUR CODE HERE

# Training loop
for i in range(500):
    sess.run(train_step, feed_dict={x_data: x_vals, y_target: y_vals[:, np.newaxis]})

    current_loss = sess.run(loss, feed_dict={x_data: x_vals, y_target: y_vals[:, np.newaxis]})
    losses.append(current_loss)

    if i == 0 or (i + 1) % 100 == 0:
        print("Epoch %d, Loss = %.2f" % (i + 1, current_loss))


# Create a mesh to plot points and predictions
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xrange, yrange = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
# Form a grid by taking each point point from x and y range
grid = np.c_[xrange.ravel(), yrange.ravel()]
grid = grid.astype(float)
# Make predictions for each point of the grid
[grid_predictions] = sess.run(prediction, feed_dict={x_data: grid})
grid_predictions = grid_predictions.reshape(xrange.shape)

# Plot initial points and color grid points according to the prediction made for each point
plt.contourf(xrange, yrange, grid_predictions, cmap='copper', alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.title('Linear SVM Results')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

# Plot loss over time
plt.plot(losses, 'k-')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
