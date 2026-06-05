# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

from fashion_mnist import load_fashion_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# Fashion MNIST 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_fashion_mnist()

print("x_train:", x_train.shape)
print("t_train:", t_train.shape)
print("x_test:", x_test.shape)
print("t_test:", t_test.shape)


max_epochs = 30


network = SimpleConvNet(
    input_dim=(1, 28, 28),

    conv_param_1={
        "filter_num": 16,
        "filter_size": 3,
        "pad": 1,
        "stride": 1
    },

    conv_param_2={
        "filter_num": 32,
        "filter_size": 3,
        "pad": 1,
        "stride": 1
    },

    hidden_size=128,
    output_size=10,
    weight_init_std=0.01
)

trainer = Trainer(
    network,
    x_train,
    t_train,
    x_test,
    t_test,
    epochs=max_epochs,
    mini_batch_size=100,
    optimizer="Adam",
    optimizer_param={"lr": 0.001},
    evaluate_sample_num_per_epoch=1000,
    verbose=True
)

print("\n===================================")
print("Fashion MNIST CNN Training Start")
print(f"Epochs: {max_epochs}")
print(f"Train Data: {len(x_train)}")
print(f"Test Data: {len(x_test)}")
print("===================================\n")

trainer.train()

network.save_params("params.pkl")
print("Saved Network Parameters!")


# 최종 정확도 출력
final_train_acc = network.accuracy(x_train, t_train)
final_test_acc = network.accuracy(x_test, t_test)

print("=============== Final Accuracy ===============")
print(f"Train Accuracy: {final_train_acc * 100:.2f}%")
print(f"Test Accuracy: {final_test_acc * 100:.2f}%")


# 그래프 저장
x = np.arange(len(trainer.train_acc_list))

plt.plot(x, trainer.train_acc_list, marker="o", label="train", markevery=2)
plt.plot(x, trainer.test_acc_list, marker="s", label="test", markevery=2)

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.title("Scratch CNN Fashion MNIST Accuracy")

plt.savefig("./graph/scratch_accuracy_graph.png")
plt.show()