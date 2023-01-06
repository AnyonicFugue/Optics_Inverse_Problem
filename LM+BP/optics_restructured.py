#多层膜正问题

#函数引入
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
n = 10                  #多层膜的层数

d = 200            #每层膜大概的厚度
delta_d = 40    #每层膜可变化范围

allowed_cost=0.001#结束迭代时，允许的误差函数值
#di=np.array([random.uniform(d-delta_d,d+delta_d) for i in range(1,n+1)])   #随机生成每层膜厚度

lambda_min=400
lambda_max=1000 #光谱的最大和最小波长
lambda_step=10    #步长
lambda_num=int((lambda_max-lambda_min)/lambda_step)

lambda_list=np.array([lambda_min+i*lambda_step for i in range(lambda_num)])


'''
Jacobian=[[0 for i in range(0,n)] for j in range(0,lambda_num)] #雅可比矩阵；第(j,i)项对应：第j个波长处的方差对第i层膜厚度的偏导
residual=[0 for j in range(0,lambda_num)] #残差，即目标光强减去当前光强

damp=0.01
v=2

spectrum_out=[0 for j in range(0,lambda_num)]

istest=False
test_spectrum=[0 for j in range(0,lambda_num)]



def input_spectrum(length):#生成目标光谱
    if(istest):
        return test_spectrum[int((length-lambda_min)/lambda_step)]
    else:
        return 0.3*(np.cos(length/10))**2

#光谱求解部分
#定义矩阵
'''

def N(i):
    if i % 2 == 1:
        return n1
    else:
        return n2         #奇数层折射率为n1，偶数层折射率为n2
    
D=np.array([np.mat([[1, 1],[N(i), -N(i)]]) for i in range(0,n)])
D_inv=np.array([np.mat([[0.5,0.5/N(i)],[0.5,-0.5/N(i)]]) for i in range(0,n)])


def P(i,lambda1,di):
    return np.mat([[cmath.exp(-(0+1j)*2*math.pi*di[i]/lambda1),0], [0, cmath.exp((0+1j)*2*math.pi/lambda1*di[i])]])    #定义矩阵P


#为了方便画图，设为函数
def E_pre(lambda1,di):
    #把最后一个单独算
    e = np.mat([[1, 1],[1, -1]])*np.mat([[1], [0]])   #空气出射
    
    #中间的用循环算
    for i in range(n-1,-1,-1):
        e=D[i]*P(i,lambda1,di)*D_inv[i]*e
    #第一个单独算
    e = np.linalg.inv(np.mat([[1, 1], [1, -1]]))* e      #入射为空气
    r = abs(e[1,0]/e[0,0])**2
    return r  #反射率

def cost_func(out_spectrum,input_spectrum):    #估值函数
    f=0
    for i in range(lambda_num):#每隔步长取一点，计算方差
        f=f+(out_spectrum[i]-input_spectrum[i])**2
    return f
                   
def get_J(di): #计算雅可比矩阵
    spectrum_out=np.zeros(lambda_num)
    J=np.zeros((lambda_num,n))

    for j in range(lambda_num):
        spectrum_out[j]=E_pre(lambda_list[j],di)

    for i in range(n):
        di[i]=di[i]+0.1

        for j in range(0,lambda_num):
            J[j][i]=10*(E_pre(lambda_list[j],di)-spectrum_out[j])

        di[i]=di[i]-0.1

    return J
    

def plot(input_spectrum,di):#画图
    y1 =[E_pre(a,di) for a in lambda_list]
    y2 =[input_spectrum[a] for a in range(lambda_num)]
    plt.xlabel('lambda')
    plt.ylabel('r')
    plt.title("Reflection spectrum")
    plt.plot(lambda_list, y1)
    plt.plot(lambda_list, y2)
    plt.show()
    
def iterate(times,di,input_spectrum):#最速下降迭代函数；di为厚度初值
    damp=0.00005
    v=2
    cost=0

    output=[E_pre(a,di) for a in lambda_list]
    cost=cost_func(output,input_spectrum)

    for i in range(0,times):

        J=np.asmatrix(get_J(di))#雅可比阵
        #print(J)

        output=np.array([E_pre(a,di) for a in lambda_list])
        residual=input_spectrum-output

        #尝试damp和damp/v，取较好者；如果都不行，就把damp乘上v，再试
        movement1=np.linalg.inv((J.T)*J+damp*np.matlib.eye(n))*(J.T)*(np.mat(residual).T)
        movement2=np.linalg.inv((J.T)*J+(damp/v)*np.matlib.eye(n))*(J.T)*(np.mat(residual).T)

        #计算两次的改进
        di=di+np.asarray(movement1.T)[0]
        output=np.array([E_pre(a,di) for a in lambda_list])#输出光强
        improve1=cost-cost_func(output,input_spectrum)

        di=di-np.asarray(movement1.T)[0]+np.asarray(movement2.T)[0]     
        output=np.array([E_pre(a,di) for a in lambda_list])
        improve2=cost-cost_func(output,input_spectrum)

        if (improve1<0)and(improve2<0):
            print("both too bad!")
            damp=damp*v
            di=di-np.asarray(movement2.T)[0]
        else:
            if(improve1>improve2):
                di=di+np.asarray(movement1.T)[0]-np.asarray(movement2.T)[0]
                print("improve1:")
                #print(improve1)
            else:
                print("improve2:")
                #print(improve2)
                damp=damp/v
        
        

        cost=cost_func(output,input_spectrum)

        if cost<allowed_cost:#足够小了就退出
            plot(input_spectrum,di)
            print("cost small enough!!")
            return

        print("cost="+str(cost))

    plot(input_spectrum,di)
    return di
    
initial_d=np.array([d for i in range(1,n+1)])   
s_in=np.array([0.3*(np.cos(a/40))**2 for a in lambda_list])




