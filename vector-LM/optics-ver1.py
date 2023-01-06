#多层膜正问题

#函数引入部分

from __future__ import division
import random
import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import cmath
import _thread

#统一改为以纳米为单位，方便计算

#几何与材料定义部分
n1,n2 = 1.33,1.66       #AB两层介质的折射率
n = 20                  #多层膜的层数

d = 1600            #每层膜大概的厚度
delta_d = 400    #每层膜可变化范围

di=np.array([random.uniform(d-delta_d,d+delta_d) for i in range(1,n+1)])   #随机生成每层膜厚度

lambda_min=400
lambda_max=900 #光谱的最大和最小波长
lambda_step=10    #步长

lambda_num=int((lambda_max-lambda_min)/+lambda_step+1)

Jacobian=[[0 for i in range(0,n)] for j in range(0,lambda_num)] #雅可比矩阵；第(j,i)项对应：第j个波长处的方差对第i层膜厚度的偏导
residual=[0 for j in range(0,lambda_num)] #残差，即目标光强减去当前光强

damp=0.01
v=2

spectrum_out=[0 for j in range(0,lambda_num)]

def input_spectrum(length):#生成目标光谱
    return 0.5*(np.sin(length/20))**2


#光谱求解部分
#定义矩阵


def N(i):
    if i % 2 == 1:
        return n1
    else:
        return n2         #奇数层折射率为n1，偶数层折射率为n2
    
D=[np.mat([[1, 1],[N(i), -N(i)]]) for i in range(0,n)]
D_inv=[np.mat([[0.5,0.5/N(i)],[0.5,-0.5/N(i)]]) for i in range(0,n)]


def P(i,lambda1):
    return np.mat([[cmath.exp(-(0+1j)*2*math.pi*di[i]/lambda1),0], [0, cmath.exp((0+1j)*2*math.pi/lambda1*di[i])]])    #定义矩阵P


#为了方便画图，设为函数
def E_pre(lambda1):
    #把最后一个单独算
    e = np.mat([[1, 1],[1, -1]])*np.mat([[1], [0]])   #空气出射
    
    #中间的用循环算
    for i in range(0,n):
        e=D[i]*P(i,lambda1)*D_inv[i]*e
    #第一个单独算
    e = np.linalg.inv(np.mat([[1, 1], [1, -1]]))* e      #入射为空气
    tmp = abs(e[1,0]/e[0,0])**2
    return tmp

def cost_func():    #估值函数
    f=0
    for i in range(lambda_min,lambda_max,lambda_step):#每隔步长取一点，计算方差
        tr=E_pre(i)
        f=f+(tr-input_spectrum(i))**2
    return f
                   
def get_J(): #计算雅可比矩阵，同时计算残差

    for j in range(0,lambda_num):
        spectrum_out[j]=E_pre(lambda_min+lambda_step*j)
        residual[j]=input_spectrum(lambda_min+lambda_step*j)-spectrum_out[j]

    for i in range(0,n):
        di[i]=di[i]+1

        for j in range(0,lambda_num):
            Jacobian[j][i]=(E_pre(lambda_min+lambda_step*j)-spectrum_out[j])

        di[i]=di[i]-1


    

def plot():#画图
    x = np.arange(lambda_min, lambda_max, 1)
    y1 =[E_pre(a) for a in x]
    y2 =[input_spectrum(a) for a in x]
    plt.xlabel('lambda')
    plt.ylabel('r')
    plt.title("Reflection spectrum")
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
    
def iterate(times):#最速下降迭代函数
    global di
    global damp

    for i in range(0,times):
        cost=cost_func()
        get_J()
        J=np.mat(Jacobian)

        #尝试damp和damp/v，取较好者；如果都不行，就把damp乘上v，再试
        
        movement1=np.linalg.inv((J.T*J)+damp*np.matlib.eye(n))*(J.T)*(np.mat(residual).T)
        movement2=np.linalg.inv((J.T*J)+(damp/v)*np.matlib.eye(n))*(J.T)*(np.mat(residual).T)

        for j in range(0,n):
            di[j]=di[j]+movement1[j,0]

        improve1=cost-cost_func()

        for j in range(0,n):
            di[j]=di[j]+movement2[j,0]-movement1[j,0]

        improve2=cost-cost_func()

        if (improve1<0)and(improve2<0):
            damp=damp*v
            for j in range(0,n):
                di[j]=di[j]-movement2[j,0]
        else:
            if(improve1>improve2):
                for j in range(0,n):
                    di[j]=di[j]-movement2[j,0]+movement1[j,0]
            
        cost=cost_func()
        print("cost="+str(cost))

    plot()
    



