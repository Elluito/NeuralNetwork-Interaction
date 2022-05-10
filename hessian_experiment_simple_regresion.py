import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow_addons as tfa
import tensorflow.keras as keras
from collections import namedtuple
import sympy as sym
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import gym
import re
import matplotlib.pyplot as plt
import scipy as sc
from matplotlib import cm
import os
import time
import multiprocessing
from multiprocessing import Process
import pickle
import math
from scipy.optimize import minimize
from DG2 import ism

BATCH_SIZE = 8
RECORD = False
PATH_TO_FOLDER = "SGD_experiments/"
import functools


class SignomialEstimator_easy():
    def __init__(self):
        self.N = 2
        self.M = 2
        # self.a = np.random.normal(2,1,(self.M,self.N))
        # self.c = np.random.normal(2,1,self.M)
        self.a = np.array([[1,1],[2,1]])
        self.c = np.array([1,-1])

    def __call__(self,x:np.ndarray):


        y = self.just_one_element(x)

        return np.array(y)
    def just_one_element(self,x_i):
        y = 0
        for i in range(self.M):
            temp =self.c[i]
            for j in range(self.N):
                temp *= x_i[j]**(self.a[i,j])
            y += temp

        return y
    def grad(self,x):

        if isinstance(x,float):
            x = np.array([x])

        gradient=[]


        #FIRST THE DERIVATE WITH RESPECT TO x
        # df_dx1 = (self(x)-2)*(2*x[0]*x[1]**0.5+1.5*x[0]**0.5*x[1]**-1.5)
        # gradient.append(df_dx1)
        # df_dx2 = (self(x) - 2) * (0.5* (x[0]**2) * x[1]**(-0.5) - 1.5 * x[0] ** 1.5 * x[1] ** -2.5)
        # gradient.append(df_dx2)

        df_dx1 =(x[1]-2*x[1]*x[0])*self(x)
        gradient.append(df_dx1)
        df_dx2 = (x[0]-x[0]**2)*self(x)
        gradient.append(df_dx2)

        #The all derivatives from the a_ij variables passing j first and i second


        gradient = np.array(gradient)
        # gradient = gradient/np.max(gradient)
        # norm = np.linalg.norm(gradient)
        # if norm > 1:
        #     gradient = gradient/norm



        # gradient = gradient/np.linalg.norm(gradient)
        return gradient

    def hessian(self,args):

        x = args

        H = np.zeros((2,2))
        # H[0,0]  = (2*x[0]*(x[1]**1.5)+1.5*(x[0]**0.5)*(x[1]**-1.5))**2+(self(x)-2)*(2*(x[1]**1.5)+1.5*0.5*(x[
        #                                                                                                        0]**-0.5)*x[
        #     1]**-1.5)
        # H[0, 1] = (0.5*x[0]**2*x[1]**-0.5-1.5*x[0]**1.5*x[1]**-2.5)*(2*x[0]*x[1]**1.5+1.5*x[0]**0.5*x[1]**-1.5)+(
        #     self(x)-2)*(1*x[0]*x[1]**-0.5+1.5*(-1.5)*x[0]**0.5*x[1]-2.5)
        # H[1, 0] = (0.5*x[0]**2*x[1]**-0.5-1.5*x[0]**1.5*x[1]**-2.5)*(2*x[0]*x[1]**1.5+1.5*x[0]**0.5*x[1]**-1.5)+(
        #     self(x)-2)*(1*x[0]*x[1]**-0.5+1.5*(-1.5)*x[0]**0.5*x[1]-2.5)
        # H[1, 1] = (0.5*(x[0]**2)*(x[1]**-0.5)-1.5*(x[0]**1.5)*(x[1]**-2.5))**2+(self(x)-2)*(0.5*(-0.5)*x[0]**2*x[
        #     1]**-1.5-1.5*(-2.5)*(x[0]**1.5)*x[1]**-3.5)
        H[0,0] = (-2*x[1])*self(x)+(x[1]-2*x[1]*x[0])**2
        H[0, 1] = (1-2*x[0])*self(x)+(x[0]-2*x[0]*x[1]**2)*(x[0]-x[0]**2)
        H[1, 0] = (1-2*x[0])*self(x)+(x[0]-2*x[0]*x[1])*(x[0]-x[0]**2)
        H[1, 1] = (x[0]-x[0]**2)**2



        return H


    def random_hessian(self,args):

        x = args
        if isinstance(x, float):
            x = np.array([x])



        random_H =[]

        for i in range(10):
            random_x = np.random.normal(x, 1, x.shape)
            H = self.hessian(random_x)
            random_H.append(H)
        random_H = np.squeeze(np.array(random_H))
        random_H = np.mean(random_H,axis=0)
        return random_H
    def parameters(self):

        return np.hstack((self.c[:],self.a.ravel()))
    def apply_update(self,update):
        actual_point = np.hstack((self.c[:], self.a.ravel()))
        new_point = actual_point+update
        self.c = new_point[:self.M]
        self.a = new_point[self.M:].reshape(self.M,self.N)


class SignomialEstimator():
    def __init__(self,dimension_X,number_of_sums,type=1):
        self.N = dimension_X
        self.M = number_of_sums
        if type==1:
            self.a = np.random.normal(3,1,(self.M,self.N))
            self.c = np.random.normal(3,1,self.M)
        else:
            self.a = np.array([[0.5]])
            self.c = np.array([0.5
                               ])

    def __call__(self,x:np.ndarray):
        shape = x.shape
        # if len(shape) > 1:
        #
        #     assert len(x) == self.N, "X[i] does not has N elements"
        #
        # else:
        #     assert x.shape[0] == self.N, "X[i]:{} does not has {} elements".format(x,self.N)
        #     x = x.reshape(-1, 1)


        y = self.just_one_element(x)

        return np.array(y)
    def just_one_element(self,x_i):
        y = 0
        for i in range(self.M):
            temp =self.c[i]
            for j in range(self.N):
                temp *= x_i[j]**(self.a[i,j])
            y += temp

        return y
    def grad(self,args):
        x,y = args
        if isinstance(x,float):
            x = np.array([x])

        gradient=[]
        multi = []
        for alpha in range(self.M):
            temp = 1
            for j in range(self.N):
                temp *= x[j]**(self.a[alpha,j])
            multi.append(temp)
        #FIRST THE DERIVATE WITH RESPECT TO C
        for alpha in range(self.M):
            df_dc_alpha = multi[alpha]*(self(x)-y)
            gradient.append(df_dc_alpha)
        #The all derivatives from the a_ij variables passing j first and i second

        for alpha in range(self.M):

            for k in range(self.N):

                df_da_alpha_k = (self(x)-y)*self.c[alpha]*multi[alpha]*np.log(x[k])



                gradient.append(df_da_alpha_k)
        gradient = np.array(gradient)
        # gradient = gradient/np.max(gradient)
        # norm = np.linalg.norm(gradient)
        # if norm > 1:
        #     gradient = gradient/norm



        # gradient = gradient/np.linalg.norm(gradient)
        return gradient
    def get_matrix_A(self,x,a,c,multi):
        matrix_return = np.zeros((self.M,self.M))
        for alpha in range(self.M):
            for beta in range(self.M):
                matrix_return[alpha,beta] = multi[alpha]*multi[beta]

        return matrix_return


    def get_matrix_B(self,x,y,a,c,multi):
        return_matrix = np.zeros((self.M,self.M*self.N))

        parameters = np.hstack((c[:],a.ravel()))
        for alpha in range(self.M):
            for beta in range(self.M):
                for k in range(self.N):
                    if alpha != beta:

                        return_matrix[alpha,beta*self.N+k] = c[beta]*multi[beta]*np.log(x[k])*multi[alpha]
                    else:
                        # for i in range(self.M):
                        #     if i == alpha:
                        #         return_matrix[alpha,beta*self.N+k] += 2 * c[alpha] * (multi[alpha]**2) * np.log(
                        #             x[k])
                        #     else:
                        #         return_matrix[alpha,beta*self.N+k] += c[i] * multi[i] * multi[alpha] * np.log(x[k])
                        # return_matrix[alpha, beta * self.N + k] -= y*multi[alpha]*np.log(x[k])
                        return_matrix[alpha, beta * self.N + k] = (forward(x,parameters,self.M,self.N)-y)*multi[
                            alpha]*np.log(x[k])+c[alpha]*(multi[alpha]**2)*np.log(x[k])




        return return_matrix

    def get_matrix_C(self,x,y,a,multi,c):
        return_matrix = np.zeros((self.M * self.N,self.M))
        parameters = np.hstack((c[:], a.ravel()))
        for alpha in range(self.M):
            for k in range(self.N):
                for beta in range(self.M):
                    if alpha != beta:
                        return_matrix[alpha * self.N + k,beta] = c[alpha]*multi[alpha] * multi[beta]* np.log(x[k])
                    else:

                        # for i in range(self.M):
                        #     if i == alpha:
                        #         return_matrix[alpha * self.N + k, beta] += 2*c[alpha]*(multi[alpha]**2)*np.log(x[k])
                        #     else:
                        #         return_matrix[alpha * self.N + k, beta] += c[i]*multi[i]*multi[alpha]*np.log(x[k])
                        #
                        # return_matrix[alpha * self.N + k, beta] += -y*multi[alpha]*np.log(x[k])
                        return_matrix[alpha* self.N+k, beta] = (forward(x,parameters,self.M,self.N) - y) * multi[
                            alpha] * np.log(x[k]) + c[alpha] * (multi[alpha] ** 2) * np.log(x[k])


        return return_matrix

    def get_matrix_D(self,x,y,a,multi,c):
        return_matrix = np.zeros((self.M * self.N, self.M * self.N))
        parameters = np.hstack((c[:], a.ravel()))
        for alpha in range(self.M):
            for beta in range(self.M):
                for k in range(self.N):
                    for h in range(self.N):
                        if alpha != beta:

                            return_matrix[alpha * self.N + k, beta * self.N + h ] = c[alpha] * multi[alpha] * np.log(x[
                                k])*c[beta]*multi[beta]*np.log(x[h])
                        else:
                            # for i in range(self.M):
                            #     if i == alpha:
                            #         return_matrix[alpha * self.N + k, beta * self.N + h] += 2*(c[alpha]**2)*(multi[
                            #                                                                                     alpha]**2)*np.log(x[k])*(np.log(x[h]))
                            #     else:
                            #         return_matrix[alpha * self.N + k, beta * self.N + h] += c[i]*multi[i]*c[
                            #             alpha]*multi[alpha]*np.log(x[k])*np.log(x[h])
                            #
                            #
                            # return_matrix[alpha * self.N + k, beta * self.N + h] += -y*c[alpha]*multi[alpha]*np.log(
                            #     x[k])*np.log(x[h])
                            return_matrix[alpha * self.N + k, beta * self.N + h] = (forward(x,parameters,self.M,self.N)-y)*multi[alpha]*c[
                                alpha]*np.log(x[k])*np.log(x[h])+(c[alpha]**2)*(multi[alpha]**2)*np.log(x[k])*np.log(
                                x[h])


        return return_matrix




    def hessian(self,args):

        x, y = args
        if isinstance(x, float):
            x = np.array([x])
        multi = []
        for alpha in range(self.M):
            temp = 1
            for j in range(self.N):
                temp *= x[j] ** (self.a[alpha, j])
            multi.append(temp)

        multi = np.array(multi)
        A = self.get_matrix_A(x,self.a,self.c,multi)
        B = self.get_matrix_B(x, y,self.a,self.c,multi)
        C = self.get_matrix_C(x,y,self.a,multi,self.c)
        D = self.get_matrix_D(x,y,self.a, multi,self.c)
        temp_1 = np.column_stack((A,B))
        temp_2 = np.column_stack((C,D))

        H = np.row_stack((temp_1,temp_2))
        return H


    def random_hessian(self,args):

        x, y = args
        if isinstance(x, float):
            x = np.array([x])




        random_H =[]

        for i in range(10):

            random_a = np.random.normal(self.a, 1, self.a.shape)
            random_c = np.random.normal(self.c,1,self.c.shape)
            multi = []
            for alpha in range(self.M):
                temp = 1
                for j in range(self.N):
                    temp *= (x[j] ** random_a[alpha, j])
                multi.append(temp)
            multi = np.array(multi)
            A = self.get_matrix_A(x, random_a, random_c, multi)
            B = self.get_matrix_B(x, y, random_a, random_c, multi)
            C = self.get_matrix_C(x, y, random_a, multi, random_c)
            D = self.get_matrix_D(x, y, random_a, multi, random_c)
            temp_1 = np.column_stack((A, B))
            temp_2 = np.column_stack((C, D))

            H = np.row_stack((temp_1, temp_2))
            random_H.append(H)
        random_H = np.squeeze(np.array(random_H))
        random_H = np.mean(random_H,axis=0)
        return random_H
    def universal_random_hessian(self,args):

        x, y = args
        if isinstance(x, float):
            x = np.array([x])


        random_H =[]

        for i in range(100):
            random_a = np.random.uniform(-5, 5, self.a.shape)
            random_c = np.random.uniform(-5,5,self.c.shape)

            multi = []
            for alpha in range(self.M):
                temp = 1
                for j in range(self.N):
                    temp *= (x[j] ** random_a[alpha, j])
                multi.append(temp)

            multi = np.array(multi)
            A = self.get_matrix_A(x, random_a, random_c, multi)
            B = self.get_matrix_B(x, y, random_a, random_c, multi)
            C = self.get_matrix_C(x, y, random_a, multi, random_c)
            D = self.get_matrix_D(x, y, random_a, multi, random_c)
            assert np.sum(B-C.T) < 1e-10,"The matrix is not simetric,C!=B,c.T:{}, b {}, sum:{}".format(C.T,B,
                                                                                                       np.sum(B-C.T))
            temp_1 = np.column_stack((A, B))
            temp_2 = np.column_stack((C, D))

            H = np.row_stack((temp_1, temp_2))
            random_H.append(H)
        random_H = np.squeeze(np.array(random_H))
        random_H = np.mean(random_H,axis=0)
        return random_H

    def det_hessian(self,args):
        x, y = args
        if isinstance(x, float):
            x = np.array([x])
        H = np.zeros((2,2))
        H[0,0]  = x**(2*self.a[0,0])
        H[0, 1] = self.c[0]*(x**(2*self.a[0,0]))*np.log(x)+(x**self.a[0,0])*np.log(x)*(self.c[0]*x**self.a[0,0]-y)
        H[1, 0] = self.c[0]*(x**(2*self.a[0,0]))*np.log(x)+(x**self.a[0,0])*np.log(x)*(self.c[0]*x**self.a[0,0]-y)
        H[1, 1] = (self.c[0]*(x**self.a[0,0])*np.log(x))**2+self.c[0]*(x**self.a[0,0]*(np.log(x)**2)*(self.c[0]*x**self.a[0,0]-y))
        return H

    def parameters(self):

        return np.hstack((self.c[:],self.a.ravel()))
    def apply_update(self,update):
        actual_point = np.hstack((self.c[:], self.a.ravel()))
        new_point = actual_point+update
        self.c = new_point[:self.M]
        self.a = new_point[self.M:].reshape(self.M,self.N)
def ackley_function(x):
      x1,x2= x[0],x[1]
  #returns the point value of the given coordinate
      part_1 = -0.2*math.sqrt(0.5*(x1*x1 + x2*x2))
      part_2 = 0.5*(math.cos(2*math.pi*x1) + math.cos(2*math.pi*x2))
      value = math.exp(1) + 20 -20*math.exp(part_1) - math.exp(part_2)

      return value

def grad_ackley_function(x):
    grad=[]
    df_dx1 = (2.82843*x[0]*np.exp(-0.141421*np.sqrt(x[0]**2+x[1]**2)))/(np.sqrt(x[0]**2+x[1]**2))+np.pi*np.sin(
        2*np.pi*x[0])
    df_dx2 = (2.82843*x[1]*np.exp(-0.141421*np.sqrt(x[0]**2+x[1]**2)))/(np.sqrt(x[0]**2+x[1]**2))+np.pi*np.sin(
        2*np.pi*x[1])
    grad.append(df_dx1)
    grad.append(df_dx2)
    grad = np.array(grad)
    return grad
def hessian_ackley_function(x):
    H = np.zeros((2,2))
    temp1 = np.sqrt(x[0]**2+x[1]**2)
    temp2 = np.exp(-0.141421*temp1)
    H[0,0] = -(0.4*(x[0]**2)*np.exp(-0.141421*temp1))/(temp1**2) - (2.82843*(x[0]**2)*temp2)/(temp1**3) + (
            2.82843*temp2)/(temp1)
    H[0,1] = -(0.4*x[1]*x[0]*temp2)/(temp1**2)-(2.82843*x[1]*x[0]*temp2)/(temp1**3)
    H[1,0] = -(0.4*x[0]*x[1]*temp2)/(temp1**2)-(2.82843*x[1]*x[0]*temp2)/(temp1**3)
    H[1,1] = -(0.4*(x[1]**2)*np.exp(-0.141421*temp1))/(temp1**2) - (2.82843*(x[1]**2)*temp2)/(temp1**3) + (
            2.82843*temp2)/(temp1)
    return H
def random_hessian_ackley_function(x,times,sigma):
    H = []
    for i in range(times):
        random_point = np.random.normal(x,sigma,x.shape)
        H.append(hessian_ackley_function(random_point))
    H = np.mean(H,axis=0)
    return H



def plot_contour(x,y,g):
    fig, ax = plt.subplots()

    X,Y = np.meshgrid(x,y)
    Z = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i,j] = g([x[i],y [j]])
    Z = Z/abs(np.max(Z))
    CS = ax.contour(X, Y, Z,100)

    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Simplest default with labels')
    return fig
def plot_surface(x,y,g,fig=None):
    # fig = plt.figure()
    if not fig:
        fig = plt.figure()

    ax = fig.gca(projection='3d')

    # Make data.
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = g(np.array([x[i], y[j]]))
    # Z = np.clip(Z, -35, 100)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z,cmap="viridis",linewidth=0, antialiased=False,alpha=0.5)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax.set_zlim(-5, 35)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig
def plot_loss_function(x_train,mode="contour"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter


    # x_train = np.array([[1]])

    # y_train = x_train ** (2) + np.random.normal(0, 1, len(x_train))

    y_train = x_train ** (2)


    # Make data.
    alpha = np.arange(0, 3.3, 0.1)
    c = np.arange(0,3.3, 0.1)
    ALPHA, C = np.meshgrid(alpha, c)
    Z = np.zeros((len(c),len(alpha)))
    for i in range(len(alpha)):
        for j in range(len(c)):
            Z[j,i] = np.mean(1/2*(c[j]*x_train**(alpha[i])-y_train)**2)
    # Z = Z/abs(np.max(Z))

    # Plot the surface.
    if mode == "contour":
        fig = plt.figure()
        ax = fig.gca()
        CS = ax.contour(ALPHA, C, Z,100)
        ax.clabel(CS, fontsize=10)
        # plt.contour()
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.scatter(2,1,c="k",marker="x")
        plt.xlabel("alpha")
        plt.ylabel("$C$")


        return fig

    if mode == "surf":
    # Add a color bar which maps values to colors.
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        surf = ax.plot_surface(ALPHA, C, Z,cmap="inferno",
                           linewidth = 0, antialiased=False,alpha=0.3)
        plt.xlabel(r"$\alpha$",size=20)
        plt.ylabel(r"C",size=20)
        ax.plot([2], [1],[0], c="g", marker="x", markersize=20)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # ax.set_zlim(-1.5, 1)





        return fig


def train_Signomial_easy(x0,mode="GD",iterations=30,lr=0.01,noise=True,trainiing_points=1000,plot=True):
    f = SignomialEstimator_easy()
    x_start = x0
    # x_train = np.array([[1]])
    #+ np.random.rand(len(x_train)) * 5
    # f.a = np.array([[0.5], [0.3], [0.1]])
    # f.c = [3, 5, -6]
    # f(x_train)

    x = x_start
    # loss = 1 / 2 * (f(x0)) ** 2
    loss = ackley_function(x)
    trail = []
    trail.append(np.array(x0))

    history = []
    history.append(loss)
    batch_size = 32
    x1 = np.linspace(0.1,5,100)
    x2 = np.linspace(0.1,5,100)
    # x1 =  np.linspace(-10,10,100)
    # x2 =  np.linspace(-10,10,100)
    for i in range(iterations):
        # batch = np.random.choice(range(len(x_train)), batch_size)


        # grad_batch = f.grad(x)
        grad_batch = grad_ackley_function(x)

        # print("Gradient norm: {}".format(np.linalg.norm(grad_batch)))
        if mode == "hessian":
            # H_batch = f.hessian(x)
            H_batch = hessian_ackley_function(x)
            inv_H_batch = np.linalg.inv(H_batch)
            inv_H_batch = inv_H_batch  # +np.eye(2)*0.01
            new_grad_batch = inv_H_batch @ grad_batch



            norm = np.linalg.norm(grad_batch)
            if plot:
                g = lambda x:1/2*(f(x))**2
                fig = plot_contour(x1,x2,g)

                plt.scatter(*x, c="m", marker="o")
                plt.scatter(1,1, c="k", marker="x")
                plt.quiver(*x, -grad_batch[0] / norm, -grad_batch[1] / norm, color=['r'], scale=21)

            # assert new_grad_batch@grad_batch>0,"the degree between gradient and gradient modified is not grater " \
            #                                    "than 0  is {}".format(new_grad_batch@grad_batch)


            if np.linalg.norm(new_grad_batch) > 1:

                new_grad_batch = new_grad_batch / np.linalg.norm(new_grad_batch)
                if plot:
                    plt.quiver(*x, -new_grad_batch[0], -new_grad_batch[1], color=['b'], scale=21)
                    plt.show()
                    plt.close()
                x += -lr*new_grad_batch

            else:
                if plot:
                    plt.quiver(*x, -new_grad_batch[0], -new_grad_batch[1], color=['b'], scale=21)
                    plt.show()
                    plt.close()
                x += -lr*new_grad_batch

        if mode == "random hessian":
            # H_batch = f.random_hessian(x)
            H_batch = random_hessian_ackley_function(x,100,3)

            inv_H_batch = np.linalg.inv(H_batch )

            inv_H_batch = inv_H_batch
            new_grad_batch = inv_H_batch @ grad_batch
            if np.linalg.norm(new_grad_batch) > 1:
                new_grad_batch = new_grad_batch / np.linalg.norm(new_grad_batch)
                x += -lr*new_grad_batch
            else:

                x += -lr*new_grad_batch

        if mode == "GD":

            if np.linalg.norm(grad_batch) > 1:
                new_grad_batch = grad_batch / np.linalg.norm(grad_batch)
                x += -lr * new_grad_batch
            else:
                new_grad_batch = grad_batch
                x += -lr * new_grad_batch

        # loss = (1 / 2 * (f(x)) - 2) ** 2
        # loss = 1/2*(f(x))**2
        loss = ackley_function(x)
        trail.append(np.array(x))
        history.append(loss)


        print("Loss {} iteration {}".format(loss, i))
        print("x1: {}".format(x[0]))
        print("x2: {}".format(x[1]))
    # g = lambda x: 1 / 2 * (f(x)) ** 2
    # fig = plot_contour(x1, x2, g)

    # plt.scatter(*x, c="m", marker="o")
    # plt.scatter(0, 0, c="k", marker="x")
    # plt.quiver(*x, -lr*grad_batch[0] , -lr*grad_batch[1] , color=['r'], scale=21)
    # plt.quiver(*x, -lr*new_grad_batch[0], -lr*new_grad_batch[1], color=['b'], scale=21)
    # plt.show()
    # plt.close()
    return x[0],x[1],history,trail

def train_Signomial(x_train,mode="GD",iterations=30,lr=0.01,noise=True,type="continuos"):
    f = SignomialEstimator(1, 1,type=2)

    trail = []
    # x_train = np.array([[1]])
    if noise:
        y_train = x_train ** (2)+np.random.normal(0,1,len(x_train))
    else:
        y_train = x_train ** (2)

    # f.a = np.array([[0.5], [0.3], [0.1]])
    # f.c = [3, 5, -6]
    # f(x_train)
    history = []
    batch_size = 32
    alpha = np.arange(-5, 3, 0.1)
    c = np.arange(-5, 2, 0.1)
    universal_H_batch = list(map(f.universal_random_hessian, zip(x_train, y_train)))
    universal_H_batch = np.mean(np.squeeze(universal_H_batch), axis=0)
    universal_inv_H_batch = universal_H_batch ** -1
    actual_H_batch = list(map(f.random_hessian, zip(x_train, y_train)))
    actual_H_batch = np.mean(np.squeeze(actual_H_batch), axis=0)
    k = lambda x :np.mean(1/2*(x[0]*x_train**x[1]-x_train**2)**2)

    for i in range(iterations):
        # batch = np.random.choice(range(len(x_train)), batch_size)
        x = x_train
        y = y_train

        grad_batch = np.array(list(map(f.grad, zip(x, y))))
        grad_batch = np.mean(np.squeeze(grad_batch), axis=0)

        # print("Gradient norm: {}".format(np.linalg.norm(grad_batch)))
        if mode == "hessian":
                H_batch = list(map(f.hessian, zip(x, y)))
                H_batch = np.mean(np.squeeze(H_batch), axis=0)
                H_batch = H_batch/np.linalg.norm(H_batch,axis=1)
                inv_H_batch = np.linalg.inv(H_batch)
                inv_H_batch = inv_H_batch      #+np.eye(2)*0.01
                new_grad_batch = grad_batch@inv_H_batch
                # condition_number = np.linalg.cond(inv_H_batch)
                # history.append(condition_number)
                # degree = np.degrees( np.arccos(np.clip(np.dot(new_grad_batch,grad_batch)*(1/(np.linalg.norm(
                #     new_grad_batch)*np.linalg.norm(
                #     grad_batch))),-1,1)))
                # fig = plot_loss_function(x_train,mode="surf")
                # g = lambda x : x.reshape(-1,2)@H_batch@x.reshape(2,-1)
                # plot_surface(alpha,c,g)
                # plt.show()
                # origin = f.parameters()
                # norm =np.linalg.norm(grad_batch)
                # plt.scatter(*origin, c="m", marker="o")
                # plt.quiver(*origin, -grad_batch[0]/norm, -grad_batch[1]/norm, color=['r'], scale=21)


                # assert new_grad_batch@grad_batch>0,"the degree between gradient and gradient modified is not grater " \
                #                                    "than 0  is {}".format(new_grad_batch@grad_batch)
                # print("angle {} rads".format(degree))
                if np.linalg.norm(new_grad_batch) > 1:

                    new_grad_batch = new_grad_batch / np.linalg.norm(new_grad_batch)
                    # plt.quiver(*origin, -new_grad_batch[0], -new_grad_batch[1], color=['b'], scale=21)
                    # plt.show()
                    # plt.close()
                    f.apply_update(-lr * new_grad_batch)

                else:


                    f.apply_update(-lr*new_grad_batch)

        if mode == "random hessian":
                if type == "continuos":

                    H_batch = list(map(f.random_hessian, zip(x, y)))
                    H_batch = np.mean(np.squeeze(H_batch), axis=0)
                    actual_H_batch = H_batch
                    print("ENTRE")
                else:
                    if i%10==0:

                        H_batch = list(map(f.random_hessian, zip(x, y)))
                        H_batch = np.mean(np.squeeze(H_batch), axis=0)
                        actual_H_batch = H_batch
                new_grad_batch = universal_inv_H_batch@ grad_batch
                # history.append(condition_number)
                if np.linalg.norm(new_grad_batch)>1:
                    new_grad_batch = new_grad_batch/np.linalg.norm(new_grad_batch)
                    f.apply_update(-lr*new_grad_batch)

                else:

                    f.apply_update(-lr*new_grad_batch)





        if mode == "GD":
                if np.linalg.norm(grad_batch)>1:
                    new_grad_batch = grad_batch/np.linalg.norm(grad_batch)
                    f.apply_update(-lr * new_grad_batch)
                    loss = np.mean(1 / 2 * (f(x.reshape(-1, 1)) - y) ** 2)



                else:
                    loss = np.mean(1 / 2 * (f(x.reshape(-1, 1)) - y) ** 2)
                    new_grad_batch = grad_batch
                    f.apply_update(-lr * new_grad_batch)




        loss = np.mean(1 / 2 * (f(x.reshape(-1, 1)) - y) ** 2)
        trail.append((f.parameters()[1],f.parameters()[0]))
        history.append(loss)

        if np.linalg.norm(f.parameters()-np.array([1,2]))< 0.1:
            break
        print("Loss {} iteration {}".format(loss, i))
        print("F_a: {}".format(f.a))
        print("F_c: {}".format(f.c))
        # print("Minimum Error: {}".format(np.mean(1/2*(y_train-x_train**2)**2)))
        # print("Norm of update: {}".format(np.linalg.norm(-lr*grad_batch)))
        # print("Norm of c: {}".format(np.linalg.norm(f.c)))
        # print("Norm of a: {}".format(np.linalg.norm(f.a)))
    # prediction = f(x_train.reshape(-1, 1))

    print("F_a: {}".format(f.a))
    print("F_c: {}".format(f.c))
    return f.a,f.c,history,trail



def forward(x,parameters,number_of_sum,number_of_dimensions):
    if len(x.shape)>1:
        y = np.zeros(len(x))
        c = parameters[:number_of_sum]
        a = parameters[number_of_sum:]
        a = a.reshape(number_of_sum,number_of_dimensions)
        for k in range(len(x)):
            for i in range(len(c)):
                temp = c[i]
                for j in range(a.shape[1]):
                    temp *= x[k,j]**a[i,j]
                y[k]+=temp
        return y
    else:
        y = 0
        c = parameters[:number_of_sum]
        a = parameters[number_of_sum:]
        a = a.reshape(number_of_sum, number_of_dimensions)
        for k in range(len(x)):
            for i in range(len(c)):
                temp = c[i]
                for j in range(a.shape[1]):
                    temp *= x[j] ** a[i, j]
                y+= temp
        return y
def matrix_comparison_experiment():






    N = 3
    M = 2
    x_train = np.random.uniform(0.5, 2, (1000, M))
    signomial = SignomialEstimator(M,N)
    param = signomial.parameters()


    y_train = 1.5*x_train[:,0]**2*x_train[:,1]**0.5\
              +3*x_train[:,0]**0.5*x_train[:,1]**2\
              -3*x_train[:,0]**0.75*x_train[:,1]**0.5

    f = lambda parameters: np.mean(1/2*(forward(x_train,parameters,N,M)-y_train)**2)
    lambda_matrix, fhat_archive, f_archive, fp1 = ism(f,N+M*N,5,-5)
    lambda_matrix[np.isnan(lambda_matrix)] = 0
    lambda_matrix = lambda_matrix+lambda_matrix.T
    lambda_matrix[lambda_matrix==0] = 0
    universal_H_batch = list(map(signomial.universal_random_hessian, zip(x_train, y_train)))
    universal_H_batch = np.mean(np.squeeze(universal_H_batch), axis=0)
    universal_H_batch[np.diag_indices_from(universal_H_batch)]=0
    # universal_H_batch = universal_H_batch/np.linalg.norm(universal_H_batch)
    lambda_matrix = lambda_matrix

    plt.matshow(lambda_matrix)
    plt.title(r"$\Lambda$ Matrix")
    plt.colorbar()
    plt.matshow(universal_H_batch)
    plt.title("H")
    plt.colorbar()
    plt.show()
    results_lambda  = np.linalg.eigvals(lambda_matrix)
    resulst_hessian = np.linalg.eigvals(universal_H_batch)
    print("Valores hessian : {}".format(resulst_hessian))
    print("Valores lambda : {}".format(results_lambda))
    print("Rango expectral lambda:{}".format(np.max(results_lambda)-np.min(results_lambda)))
    print("Rango expectral UH:{}".format(np.max(resulst_hessian)-np.min(resulst_hessian)))






def run_experiment_signomial():
    import datetime as date

    number_of_experiments = 1
    results_GD = []
    result_hessian = []
    results_random_hessian = []
    history_GD_s = []
    history_hessian_s = []
    history_random_hessian_s = []
    first = np.random.normal(0, 1, 2)
    x_train = np.random.normal(3, 0.5, 1000)
    g = SignomialEstimator(1, 1)
    l = lambda x: np.mean(1 / 2 * (x[0] * x_train ** x[1] - x_train ** 2) ** 2)
    # solution = minimize(l,g.parameters(),method="L-BFGS-B")
    for exp in range(number_of_experiments):
        # first = np.random.normal(0, 1, 2)
        a, c, history_GD, trail_1 = train_Signomial(np.array(x_train), lr=0.1, iterations=100)
        results_GD.append((a, c))
        history_GD_s.append(history_GD)
    for exp in range(number_of_experiments):
        #     first = np.random.normal(0, 1, 2)
        a, c, history_hessian, trail_2 = train_Signomial(np.array(x_train), lr=0.1, iterations=100, mode="hessian",

                                                         )
        result_hessian.append((a, c))
        history_hessian_s.append(history_hessian)
    for exp in range(number_of_experiments):
        #     first = np.random.normal(0, 1, 2)
        a, c, history_random_hessian, trail_3 = train_Signomial(np.array(x_train), lr=0.1, iterations=100,
                                                                mode="random hessian")

        results_random_hessian.append((a, c))
        history_random_hessian_s.append(history_random_hessian)
    trail_1 = np.array(trail_1)
    trail_2 = np.array(trail_2)
    trail_3 = np.array(trail_3)

    X = np.linspace(-1, 1, 100)
    f = SignomialEstimator_easy()
    # g = lambda y : 1/2*(f(y))**2
    # results_GD = np.squeeze(np.array(results_GD, dtype=np.float64))
    # result_hessian = np.squeeze(np.array(result_hessian, dtype=np.float64))
    # results_random_hessian = np.squeeze(np.array(results_random_hessian, dtype=np.float64))
    # fig = plot_surface(X,X,ackley_function)
    fig = plot_loss_function(x_train, mode="surf")
    ax = fig.gca(projection="3d")
    ax.plot(trail_1[:, 0], trail_1[:, 1], history_GD, c="r", label="SGD", marker="o")
    ax.plot(trail_2[:, 0], trail_2[:, 1], history_hessian, c="k", label="Hessian", marker="o")
    ax.plot(trail_3[:, 0], trail_3[:, 1], history_random_hessian, c="b", label="Random Hessian", marker="o")
    plt.legend()
    plt.figure()
    l1 = np.mean(history_GD, axis=0)
    l2 = np.mean(history_hessian, axis=0)
    l3 = np.mean(history_random_hessian, axis=0)
    plt.plot(history_GD, label="GD")
    plt.plot(history_hessian, label="Hessian")
    # plt.xlabel(r"$Iterations$", size=20)
    # plt.ylabel(r"\rho", size=20)
    # plt.legend()

    # plt.figure()
    plt.plot(history_random_hessian, label="Random Hessian")
    plt.xlabel(r"$Iterations$", size=20)
    plt.ylabel(r"Loss", size=20)
    plt.legend()

    plt.show()

    print(
        "SGD:{} HESSIAN:{} Random Hessian: {}".format(history_GD[-1], history_hessian[-1], history_random_hessian[-1]))
    print("Best point SGD:{} HESSIAN:{} Random Hessian: {}".format(results_GD[-1], result_hessian[-1],
                                                                   results_random_hessian[-1]))
    with open("trail_GD.h5", "w+b") as f:
        pickle.dump(trail_1, f)
    with open("trail_hessian.h5", "w+b") as f:
        pickle.dump(trail_2, f)
    with open("trail_random_hessian.h5", "w+b") as f:
        pickle.dump(trail_3, f)
    with open("history_GD_data_driven.h5", "w+b") as f:
        pickle.dump(history_GD, f)

    with open("history_hessian_data_driven.h5", "w+b") as f:
        pickle.dump(history_hessian, f)
    with open("history_random_hessian_data_driven.h5", "w+b") as f:
        pickle.dump(history_random_hessian, f)

    results_GD = np.squeeze(np.array(results_GD, dtype=np.float64))
    result_hessian = np.squeeze(np.array(result_hessian, dtype=np.float64))
    results_random_hessian = np.squeeze(np.array(results_random_hessian, dtype=np.float64))
    #

    # print("Resultados para gradient descent")
    # print("Media de a {} STD: {}".format(np.mean(results_GD,axis=0)[0],np.std(results_GD,axis=0)[0]))
    # print("Media de c {} STD: {}".format(np.mean(results_GD,axis=0)[1],np.std(results_GD,axis=0)[1]))
    # print("Resultados para Hessian")
    # print("Media de a {} STD: {}".format(np.mean(result_hessian, axis=0)[0], np.std(result_hessian, axis=0)[0]))
    # print("Media de c {} STD: {}".format(np.mean(result_hessian, axis=0)[1], np.std(result_hessian, axis=0)[1]))
    #
    # print("Resultados para Random Hessian")
    # print("Media de a {} STD: {}".format(np.mean(results_random_hessian, axis=0)[0], np.std(results_random_hessian, axis=0)[0]))
    # print("Media de c {} STD: {}".format(np.mean(results_random_hessian, axis=0)[1], np.std(results_random_hessian, axis=0)[1]))
    # plt.plot(history_GD)
    # plt.figure()
    # plt.plot(history_hessian)
    # plt.figure()
    # plt.plot(history_random_hessian)
    # plt.show()

    # model = build_model((1,),1,number_neurons=4,type="critic")
    # PATH_TO_FOLDER = "/content/gdrive/MyDrive/PhD/Hessian Experiments/Random Hessian/"
    # x_train = np.random.uniform(-2*np.pi,2*np.pi,10000)
    # x_test = np.random.uniform(-2*np.pi,2*np.pi,1000)
    # y_train = 3*np.cos(x_train)+np.random.rand(len(x_train))*0.5
    # y_test = 3 *np.cos(x_test) + np.random.rand(len(x_test)) * 0.5
    # # # Batchsize = 32
    # # # Total updates =20*313=6260
    # # history = model.fit(x_train,y_train,epochs=4,batch_size=32)
    # # prediction = model.predict(x_test)
    # # plt.scatter(x_test,y_test,label="true")
    # # plt.scatter(x_test, prediction,label="prediction")
    # # plt.scatter(x_train, y_train, label="Train",alpha=0.1)
    # # plt.legend()
    # # plt.show()
    # model2 = build_model((1,),1,number_neurons=4,type="critic")
    #
    # #
    # train(model2,x_train,y_train,total_iterations=313*10,validation_data=[x_test,y_test],path_to_folder=PATH_TO_FOLDER)
    #
    # prediction = model2.predict(x_test)
    # plt.scatter(x_test, y_test, label="true")
    # plt.scatter(x_test, prediction, label="prediction")
    # plt.scatter(x_train, y_train, label="Train", alpha=0.1)
    # plt.legend()
    # plt.show()


import numpy as np





if __name__ == '__main__':


    #matrix_comparison_experiment()
    run_experiment_signomial()







