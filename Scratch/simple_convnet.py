# coding: utf-8
import pickle
import numpy as np
from collections import OrderedDict

from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss


class SimpleConvNet:
    """
    Fashion MNIST 분류를 위한 CNN 모델

    교재의 SimpleConvNet 구조를 기반으로 하되,
    Fashion MNIST 성능 향상을 위해 합성곱 계층을 2개로 확장하였다.

    기존 교재 구조:
    Conv - Relu - Pool - Affine - Relu - Affine - Softmax

    수정 구조:
    Conv1 - Relu1 - Pool1
    Conv2 - Relu2 - Pool2
    Affine1 - Relu3
    Affine2 - Softmax
    """

    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param_1={"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_2={"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
        hidden_size=128,
        output_size=10,
        weight_init_std=0.01
    ):
        input_channel = input_dim[0]
        input_size = input_dim[1]

        filter_num_1 = conv_param_1["filter_num"]
        filter_size_1 = conv_param_1["filter_size"]
        filter_pad_1 = conv_param_1["pad"]
        filter_stride_1 = conv_param_1["stride"]

        filter_num_2 = conv_param_2["filter_num"]
        filter_size_2 = conv_param_2["filter_size"]
        filter_pad_2 = conv_param_2["pad"]
        filter_stride_2 = conv_param_2["stride"]

        conv1_output_size = int((input_size - filter_size_1 + 2 * filter_pad_1) / filter_stride_1 + 1)
        pool1_output_size = int(conv1_output_size / 2)

        conv2_output_size = int((pool1_output_size - filter_size_2 + 2 * filter_pad_2) / filter_stride_2 + 1)
        pool2_output_size = int(conv2_output_size / 2)

        affine_input_size = filter_num_2 * pool2_output_size * pool2_output_size

        self.params = {}

        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num_1,
            input_channel,
            filter_size_1,
            filter_size_1
        )
        self.params["b1"] = np.zeros(filter_num_1)

        self.params["W2"] = weight_init_std * np.random.randn(
            filter_num_2,
            filter_num_1,
            filter_size_2,
            filter_size_2
        )
        self.params["b2"] = np.zeros(filter_num_2)

        self.params["W3"] = weight_init_std * np.random.randn(
            affine_input_size,
            hidden_size
        )
        self.params["b3"] = np.zeros(hidden_size)

        self.params["W4"] = weight_init_std * np.random.randn(
            hidden_size,
            output_size
        )
        self.params["b4"] = np.zeros(output_size)

        self.layers = OrderedDict()

        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            stride=filter_stride_1,
            pad=filter_pad_1
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers["Conv2"] = Convolution(
            self.params["W2"],
            self.params["b2"],
            stride=filter_stride_2,
            pad=filter_pad_2
        )
        self.layers["Relu2"] = Relu()
        self.layers["Pool2"] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers["Affine1"] = Affine(
            self.params["W3"],
            self.params["b3"]
        )
        self.layers["Relu3"] = Relu()

        self.layers["Affine2"] = Affine(
            self.params["W4"],
            self.params["b4"]
        )

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]

            y = self.predict(tx)
            y = np.argmax(y, axis=1)

            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db

        grads["W2"] = self.layers["Conv2"].dW
        grads["b2"] = self.layers["Conv2"].db

        grads["W3"] = self.layers["Affine1"].dW
        grads["b3"] = self.layers["Affine1"].db

        grads["W4"] = self.layers["Affine2"].dW
        grads["b4"] = self.layers["Affine2"].db

        return grads

    def save_params(self, file_name="params.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        layer_names = ["Conv1", "Conv2", "Affine1", "Affine2"]

        for i, layer_name in enumerate(layer_names):
            self.layers[layer_name].W = self.params["W" + str(i + 1)]
            self.layers[layer_name].b = self.params["b" + str(i + 1)]