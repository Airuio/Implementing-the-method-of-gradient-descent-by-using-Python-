# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 18:30:54 2018

@author: a
"""
#实现梯度下降法：
import numpy as np

class AdalineGD(object):
    def __init__(self,eta=0.01,n_iter=50):  #定义超参数学习率和迭代次数
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self,X,y):        #定义权重系数w和损失函数cost
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):  #更新权重
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def activation(self,X):
        return self.net_input(X) 

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
 