{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DMwqBudjfvSL",
        "outputId": "30cad1f0-93a8-41f2-b313-eacb5260f31b"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: keras in /usr/local/lib/python3.7/dist-packages (2.7.0)\n"
          ]
        }
      ],
      "source": [
        "pip install keras"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TSXTRc37giwE",
        "outputId": "c23c135a-02be-4042-d752-ac189420082e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n",
            "(28, 28)\n",
            "Epoch 1/10\n",
            "1875/1875 [==============================] - 227s 120ms/step - loss: 0.4044 - accuracy: 0.9172\n",
            "Epoch 2/10\n",
            "  34/1875 [..............................] - ETA: 3:43 - loss: 0.1276 - accuracy: 0.9614"
          ]
        }
      ],
      "source": [
        "#reference https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5\n",
        "# reference https://elitedatascience.com/keras-tutorial-deep-learning-in-python\n",
        "\n",
        "from keras.datasets import mnist\n",
        "import matplotlib.pyplot as plt\n",
        "from tensorflow.keras.utils import to_categorical\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout\n",
        "import numpy as np\n",
        "\n",
        "\n",
        "def gen_image(arr):\n",
        "    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)\n",
        "    plt.imshow(two_d, interpolation='nearest')\n",
        "    return plt\n",
        "  \n",
        "\n",
        "  \n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "\n",
        "#download mnist data and split into train and test sets\n",
        "(X_train, y_train), (X_test, y_test) = mnist.load_data()\n",
        "\n",
        "#check image shape\n",
        "print(X_train[0].shape)\n",
        "\n",
        "#normalization values between 0 and 1\n",
        "#X_train = X_train / 255\n",
        "#X_test = X_test / 255\n",
        "\n",
        "#reshape data to fit model\n",
        "X_train = X_train.reshape(60000,28,28,1)\n",
        "X_test = X_test.reshape(10000,28,28,1)\n",
        "\n",
        "#one-hot encode target column which is equal to generate_t in program 5\n",
        "y_train = to_categorical(y_train)\n",
        "y_test = to_categorical(y_test)\n",
        "y_train[0]\n",
        "\n",
        "\n",
        "#create model\n",
        "model = Sequential()\n",
        "#add model layers\n",
        "model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1))) \n",
        "        # 64 are the number of filters, kernel size is the size of the filters example 3*3 here. activation used is relu.finally shape of the image\n",
        "model.add(Conv2D(32, kernel_size=3,strides=(1, 1), activation='relu'))\n",
        "#model.add(MaxPooling2D(pool_size=(2,2)))\n",
        "model.add(Dropout(0.25))\n",
        "model.add(Flatten())\n",
        "model.add(Dense(128, activation='relu'))\n",
        "model.add(Dropout(0.5))\n",
        "model.add(Dense(10, activation='softmax'))\n",
        "\n",
        "# 8. Compile model\n",
        "model.compile(loss='categorical_crossentropy',\n",
        "              optimizer='adam',\n",
        "              metrics=['accuracy'])\n",
        " \n",
        "# 9. Fit model on training data\n",
        "model.fit(X_train, y_train, \n",
        "          batch_size=32, epochs=10, verbose=1) #epochs  = iterations(Nit)\n",
        "\n",
        "# 10. Evaluate model on test data\n",
        "score = model.evaluate(X_test, y_test, verbose=1)\n",
        "\n",
        "print('Testing accuracy - > ',score[1] * 100)\n",
        " \n",
        "ytested = model.predict_classes(X_test)\n",
        "for i in range(4):\n",
        "  gen_image(X_test[i]).show() # printing image vs the predicted image below\n",
        "  print(\"The Predicted Testing image is =%s\" % (ytested[i]))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cF7MH8Brf9E_"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "MNIST_KERAS_programassignment6 (3).ipynb",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.8.2"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
