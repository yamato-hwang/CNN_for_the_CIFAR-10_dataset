# CNN_for_the_CIFAR-10_dataset

The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It has 50,000 training images and 10,000 test images. The dataset consists of airplanes, dogs, cats, and other objects.

A Convolutional Neural Network (CNN) is particularly well-suited for image classification tasks like those presented in the CIFAR-10 dataset. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from images. They have been very effective in identifying faces, objects, and traffic signs apart from powering vision in robots and self-driving cars.

A typical CNN for CIFAR-10 classification might consist of several convolutional layers, followed by pooling layers, fully connected layers, and normalization layers. The convolutional layers are responsible for extracting features from the images, the pooling layers reduce the spatial dimensions of the extracted features, and the fully connected layers perform the classification based on these features. Normalization layers, such as Batch Normalization, help in stabilizing and accelerating the training process.

To train a CNN on the CIFAR-10 dataset, one would typically use a framework like TensorFlow or PyTorch, split the dataset into training and validation sets, preprocess the images, define the network architecture, and then train the model using an optimization algorithm like Stochastic Gradient Descent (SGD) or Adam. Once trained, the model can be evaluated on the test set to determine its accuracy.







Test output :

170498071/170498071 [==============================] - 61s 0us/step
Epoch 1/20
782/782 [==============================] - 15s 19ms/step - loss: 1.8733 - accuracy: 0.3265 - val_loss: 2.1211 - val_accuracy: 0.3096
Epoch 2/20
782/782 [==============================] - 15s 20ms/step - loss: 1.4977 - accuracy: 0.4542 - val_loss: 1.5719 - val_accuracy: 0.4585
Epoch 3/20
782/782 [==============================] - 16s 20ms/step - loss: 1.3553 - accuracy: 0.5152 - val_loss: 1.4912 - val_accuracy: 0.5101
Epoch 4/20
782/782 [==============================] - 17s 22ms/step - loss: 1.2548 - accuracy: 0.5555 - val_loss: 1.2463 - val_accuracy: 0.5506
Epoch 5/20
782/782 [==============================] - 18s 23ms/step - loss: 1.1824 - accuracy: 0.5828 - val_loss: 1.1890 - val_accuracy: 0.5949
Epoch 6/20
782/782 [==============================] - 19s 24ms/step - loss: 1.1240 - accuracy: 0.6052 - val_loss: 1.3194 - val_accuracy: 0.5614
Epoch 7/20
782/782 [==============================] - 19s 24ms/step - loss: 1.0890 - accuracy: 0.6190 - val_loss: 1.0106 - val_accuracy: 0.6424
Epoch 8/20
782/782 [==============================] - 18s 24ms/step - loss: 1.0544 - accuracy: 0.6316 - val_loss: 0.9553 - val_accuracy: 0.6647
Epoch 9/20
782/782 [==============================] - 19s 24ms/step - loss: 1.0270 - accuracy: 0.6429 - val_loss: 1.1168 - val_accuracy: 0.6276
Epoch 10/20
782/782 [==============================] - 19s 24ms/step - loss: 0.9964 - accuracy: 0.6519 - val_loss: 0.9132 - val_accuracy: 0.6831
Epoch 11/20
782/782 [==============================] - 19s 24ms/step - loss: 0.9772 - accuracy: 0.6579 - val_loss: 0.8357 - val_accuracy: 0.7080
Epoch 12/20
782/782 [==============================] - 19s 24ms/step - loss: 0.9531 - accuracy: 0.6677 - val_loss: 0.8384 - val_accuracy: 0.7006
Epoch 13/20
782/782 [==============================] - 19s 24ms/step - loss: 0.9316 - accuracy: 0.6744 - val_loss: 0.8189 - val_accuracy: 0.7155
Epoch 14/20
782/782 [==============================] - 19s 24ms/step - loss: 0.9173 - accuracy: 0.6797 - val_loss: 0.8677 - val_accuracy: 0.6951
Epoch 15/20
782/782 [==============================] - 19s 24ms/step - loss: 0.9060 - accuracy: 0.6860 - val_loss: 0.8172 - val_accuracy: 0.7130
Epoch 16/20
782/782 [==============================] - 20s 26ms/step - loss: 0.8925 - accuracy: 0.6934 - val_loss: 0.7743 - val_accuracy: 0.7343
Epoch 17/20
782/782 [==============================] - 20s 25ms/step - loss: 0.8767 - accuracy: 0.6961 - val_loss: 0.7420 - val_accuracy: 0.7410
Epoch 18/20
782/782 [==============================] - 20s 26ms/step - loss: 0.8738 - accuracy: 0.6993 - val_loss: 0.7560 - val_accuracy: 0.7363
Epoch 19/20
782/782 [==============================] - 20s 25ms/step - loss: 0.8598 - accuracy: 0.7017 - val_loss: 0.8396 - val_accuracy: 0.7103
Epoch 20/20
782/782 [==============================] - 20s 26ms/step - loss: 0.8512 - accuracy: 0.7063 - val_loss: 0.8995 - val_accuracy: 0.6892
313/313 [==============================] - 2s 7ms/step - loss: 0.8995 - accuracy: 0.6892
Test Accuracy: 68.92%
