import tensorflow as tf
from tensorflow import keras
# from tensorflow import layers, models
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#2. load dataset (MNIST is builtin)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#3. normalize data (scale pixel values to 0-1)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

#4. reshape data (CNN : expect 3D: height,width,channels)
X_train = X_train.reshape(-1, 28, 28 ,1)
X_test = X_test.reshape(-1, 28, 28,1)
# # -1 means figure this dimension out automatically
# # 1 is no. of channels

#5. Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # convolution
    layers.MaxPooling2D((2,2)), # pooling layer
    layers.Flatten(),     # flatten into 1D
    layers.Dense(64, activation='relu'),  # fully connected layer
    layers.Dense(10, activation='softmax')    #output layer(10 classes)
])

#6. compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#7. train the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,   # faster training
    validation_data=(X_test, y_test),  
    verbose=1   # shows progress bar
)

#8. evaluate on the test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", round(test_acc*100, 2), "%")

#9. predict example
prediction = model.predict(X_test[:1])  # get prediction probabilities
predicted_label = prediction.argmax()   # find the most likely class

plt.imshow(X_test[0].reshape(28,28), cmap="gray")
plt.title("prediction: " + str(predicted_label))
plt.axis("off")
plt.show()
