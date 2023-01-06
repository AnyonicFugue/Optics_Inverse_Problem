#多层膜正问题

#函数引入部分

from __future__ import division
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import cmath
import _thread

#统一改为以纳米为单位，方便计算

#几何与材料定义部分
n1,n2 = 1.33,1.66       #AB两层介质的折射率
n = 40                  #多层膜的层数

d = 1600            #每层膜大概的厚度
delta_d = 400    #每层膜可变化范围

di=[random.uniform(d-delta_d,d+delta_d) for i in range(1,n+1)]      #随机生成每层膜厚度

lambda_min=400
lambda_max=900 #光谱的最大和最小波长
lambda_step=2    #步长

lambda_num=int((lambda_max-lambda_min)/+lambda_step+1)

grad=[0 for i in range(0,n)]#梯度
grad_norm=0
part_norm=[0 for i in range(0,4)]
cost=0

def input_spectrum(length):
    return 0.6*(np.sin(length/20))**2


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
                   

def get_part_grad(a,b):#本来想写多线程，用此函数求第a层到第b层的梯度，但是会有Bug
    global cost
    for i in range(a,b):
        di[i]=di[i]+0.1
        tmp=cost_func()
        #print("tmp["+str(i)+"]="+str(tmp))
        grad[i]=10*(tmp-cost)
        #这一步就是对第i层的厚度求导
        di[i]=di[i]-0.1
    return

def get_grad(): #计算估值函数在参数空间的梯度
    global cost
    cost=cost_func()
    print("cost="+str(cost))
    
    global grad_norm
    grad_norm=0
    
    get_part_grad(0,n)
    #for i in range(0,4):
     #   _thread.start_new_thread(get_part_grad,(i*ndiv4,(i+1)*ndiv4))

    for i in range(0,n):
        grad_norm=grad_norm+(grad[i])**2
        
    grad_norm=np.sqrt(grad_norm)    
    

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
    
def iterate(times):#梯度下降迭代函数
    for i in range(0,times):
        get_grad()
        global cost
        cost=cost_func()
        global grad_norm
    
        steplength=cost/grad_norm
    
        for i in range(0,n):
            di[i]=di[i]-steplength*grad[i]

    plot()
    



