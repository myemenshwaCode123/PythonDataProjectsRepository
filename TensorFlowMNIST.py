import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
# One hot encoding mean's only one output is active at a time
# For example, let's say your classifying your digit as 7, you first start from 0 and count to 7,
# then when you get to 7, that means the bit is active, so it's a 1, then keep counting until you get to 9.
# so this would be the representation -------------->  0000000100
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

