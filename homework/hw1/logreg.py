from sklearn import datasets 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

'''
Write a python function to randomly split the dataset into 80% samples for training and 20% for
testing. You will train the model on the training subset and once the model is trained, evaluate it
on the testing subset.
'''
def divide_dataset(data_x, data_y, train_percentage):
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(
            data_x, data_y, train_size=train_percentage)
    return train_data_x, train_data_y, test_data_x, test_data_y

'''
The input x should be a Numpy array representing
a batch of data. Each element in the array is an individual input to the sigmoid function. The
output of sigmoid function ˆy should return a Numpy array of the same shape as the input.
'''
def sigmoid(x):
    return 1/(1+(np.exp((-x))))

'''
For calculating accuracy, we will threshold the output, i.e., if ˆy > 0.5, classify as 1; otherwise 0.
Write a function to calculate accuracy and print accuracy after every 1000th training iteration.
'''
def calculating_accuracy(y, y_pred):
    y_pred = np.where(y_pred > 0.5, 1, 0)
    acc = np.sum(y == y_pred) / len(y)
    # y = np.where(y > 0.5, 1, 0)
    # temp = np.where(y == y_pred, 1, 0)
    # acc = sum(temp)/temp.size()
    return acc

def weights_init(x):
    weights = np.random.rand(x.shape[1])
    bias = np.random.rand()
    return weights, bias

def gradient_decent(w, b, alpha, data_x, data_y):

    n = len(data_y)

    # fp
    y_pred = sigmoid(np.dot(data_x, w) + b)

    # comp gradients
    dw = (1/n) * np.dot(data_x.T, (y_pred - data_y))
    db = (1/n) * np.sum(y_pred - data_y)

    # update w and b
    new_w = w - alpha * dw
    new_b = b - alpha * db

    return new_w, new_b

def train(x, y, alpha, num_iterations):

    w, b = weights_init(x)

    # plot vars 
    accur = []
    losses = []
    iterations = []

    for i in range(num_iterations):
        w, b = gradient_decent(w, b, alpha, x, y)

        if i % 1000 == 0:
            y_pred = sigmoid(np.dot(x, w) + b)

            # plot stuff 
            acc = calculating_accuracy(y, y_pred)
            accur.append(acc)
            print(f"Iteration {i}, Accuracy: {calculating_accuracy(y, y_pred)}")

            # BCE = -(y^(i) log(y) + (1-y^(i))log(1-y))
            loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
            losses.append(loss)

            iterations.append(i)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))

    ax1.plot(iterations, accur)
    ax1.set_title("Accuracy vs Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Accuracy")

    ax2.plot(iterations, losses)
    ax2.set_title("Loss vs Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")

    plt.show()

    return w, b

if __name__ == "__main__":
    cancer = datasets.load_breast_cancer()
    X, y = cancer['data'], cancer['target']
    
    # print(X.head())
    # print(y.head())
    
    train_data_x, train_data_y, test_data_x, test_data_y = divide_dataset(X, y, 0.8)

    # print(train_data_x)
    # print(train_data_y)
    # print(calculating_accuracy(train_data_x, train_data_y))

    # weights = weights_init(train_data_x) 
    # print(weights)

    w, b = train(train_data_x, train_data_y, alpha=0.7, num_iterations=5000)

    # Evaluate on test set
    y_pred_test = sigmoid(np.dot(test_data_x, w) + b)
    print(f"Test Accuracy: {calculating_accuracy(test_data_y, y_pred_test)}")



