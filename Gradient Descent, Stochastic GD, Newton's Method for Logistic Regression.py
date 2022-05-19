import numpy as np
import math
import matplotlib.pyplot as plt
import time

# Stochastic Gradient Descent
def sgd(data_num=100, fig_num=99, iter_num=100, lr = 0.003):
    start = time.time()
    x = np.random.rand(fig_num, data_num)
    add_row = np.ones(data_num)
    x_hat = np.vstack([x, add_row])
    y_vec = np.random.randint(2, size=(data_num))
    bet = np.random.rand(1,fig_num +1)
    iter_values = list(range(iter_num))
    obj_values = []
    obj = 0
    for j in range(data_num):
        item = math.log(1 + math.exp(np.dot(bet, x_hat[:, j]))) - y_vec[j] * np.dot(bet, x_hat[:, j])
        obj += item
    obj_values.append(obj)

    for i in range(1,iter_num):
        n = np.random.choice(data_num)
        grad = (1/(1+math.exp(-np.dot(bet, x_hat[:,n]))) - y_vec[n]) * x_hat[:,n]
        bet = bet - lr * grad
        obj=0
        for j in range(data_num):
            item = math.log(1 + math.exp(np.dot(bet,x_hat[:,j]))) - y_vec[j]*np.dot(bet,x_hat[:,j])
            obj += item
        obj_values.append(obj)
    end = time.time()
    execution_time = end-start
    plt.plot(iter_values, obj_values)
    plt.xlabel('Iterations')
    plt.ylabel('Objective values')
    plt.title("SGD (datasize = "+str(data_num) + ', running time ' + str(execution_time)+'s, lr = '+str(lr)+')')
    plt.show()
#sgd(1000,999,1000,0.003)

#Gradient Descent
def gd(data_num=100, fig_num=99, iter_num=100, lr=0.003):
    start = time.time()
    x = np.random.rand(fig_num, data_num)
    add_row = np.ones(data_num)
    x_hat = np.vstack([x, add_row])
    y_vec = np.random.randint(2, size=(data_num))
    bet = np.random.rand(1,fig_num +1)
    iter_values = list(range(iter_num))
    obj_values = []
    obj = 0
    for j in range(data_num):
        item = math.log(1 + math.exp(np.dot(bet, x_hat[:, j]))) - y_vec[j] * np.dot(bet, x_hat[:, j])
        obj += item
    obj_values.append(obj)
    for i in range(1,iter_num):
        grad = np.zeros(fig_num +1)
        for k in range(data_num):
            grad_item = (1/(1+math.exp(-np.dot(bet, x_hat[:,k]))) - y_vec[k]) * x_hat[:,k]
            grad += grad_item
        bet = bet - lr * grad
        obj=0
        for j in range(data_num):
            item = math.log(1 + math.exp(np.dot(bet,x_hat[:,j]))) - y_vec[j]*np.dot(bet,x_hat[:,j])
            obj += item
        obj_values.append(obj)
    end = time.time()
    execution_time = end-start
    plt.plot(iter_values, obj_values)
    plt.xlabel('Iterations')
    plt.ylabel('Objective values')
    plt.title("GD (datasize = "+str(data_num) + ', running time ' + str(execution_time)+'s, lr= '+str(lr)+')')
    plt.show()
#gd(1000,999,1000,0.003)

# Newton's Method
def nm(data_num=100, fig_num=99, iter_num=100):
    start = time.time()
    x = np.random.rand(fig_num, data_num)/100
    add_row = np.ones(data_num)
    x_hat = np.vstack([x, add_row])
    y_vec = np.random.randint(2, size=(data_num))
    bet = np.random.rand(1,fig_num +1)/100
    iter_values = list(range(iter_num))
    obj_values = []
    obj = 0
    for j in range(data_num):
        item = math.log(1 + math.exp(np.dot(bet, x_hat[:, j]))) - y_vec[j] * np.dot(bet, x_hat[:, j])
        obj += item
    obj_values.append(obj)
    for i in range(1,iter_num):
        grad = np.zeros(fig_num +1)
        for k in range(data_num):
            grad_item = (1/(1+math.exp(-np.dot(bet, x_hat[:,k]))) - y_vec[k]) * x_hat[:,k]
            grad += grad_item
        hess = np.zeros((fig_num + 1, fig_num + 1))
        for h in range(data_num):
            e = math.exp(-np.dot(bet, x_hat[:, h]))
            hess_item = (e / ((1 + e) ** 2)) * np.outer(x_hat[:, h], x_hat[:, h])
            hess += hess_item
        hess_inv = np.linalg.inv(hess)
        bet = bet - hess_inv.dot(grad)
        obj=0
        for j in range(data_num):
            item = math.log(1 + math.exp(np.dot(bet,x_hat[:,j]))) - y_vec[j]*np.dot(bet,x_hat[:,j])
            obj += item
        obj_values.append(obj)
    end = time.time()
    execution_time = end-start
    plt.plot(iter_values, obj_values)
    plt.xlabel('Iterations')
    plt.ylabel('Objective values')
    plt.title("NM (datasize = "+str(data_num) + ', running time ' + str(execution_time)+'s)')
    plt.show()
nm(100,99,50)


