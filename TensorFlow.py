import tensorflow as tf

# Constants for True and False values
T, F = 1., -1.

# Bias value
bias = 1.

# Input data with bias
train_in = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]

# Corresponding desired output
train_out = [
    [T],
    [F],
    [F],
    [F],
]

# Create a random weight variable with shape (3, 1)
w = tf.Variable(tf.random.normal([3, 1]))

# Define the step function using the TensorFlow graph mode
def step(x):
    # Check which elements are greater than 0
    is_greater = tf.greater(x, 0)

    # Convert boolean values to float32
    as_float = tf.dtypes.cast(is_greater, tf.float32)

    # Multiply by 2 and subtract 1 to map True to 1 and False to -1
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)

# Set up initial error and target for stopping condition
err = 1
target = 0

# Set up initial epoch and maximum number of epochs
epoch, max_epoch = 0, 10

# Training loop
while err > target and epoch < max_epoch:
    epoch += 1

    # Calculate the output using the step function and input data
    output = step(tf.matmul(train_in, w))

    # Calculate the error by subtracting the desired output from the calculated output
    error = tf.subtract(train_out, output)

    # Calculate the mean squared error (mse) from the error
    mse = tf.reduce_mean(tf.square(error))

    # Calculate the delta for weight update using the input data and error
    delta = tf.matmul(tf.transpose(train_in), error)

    # Update the weight using the calculated delta
    train_op = w.assign_add(delta)

    # Update the error value with the current mse
    err = mse.numpy()

    # Print the current epoch and mse
    print('epoch:', epoch, 'mse:', err)

print("Training completed.")




