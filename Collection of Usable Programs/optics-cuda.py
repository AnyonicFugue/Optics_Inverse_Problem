# 多层膜正问题

# 函数引入
from __future__ import division
from functools import lru_cache
import random
from numba.cuda.simulator.cudadrv.devicearray import to_device
import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import cmath

from time import time
from numba import cuda
from numba import jit

# 统一改为以纳米为单位，方便计算

# 几何与材料定义部分
n1, n2 = 1.33, 1.67  # AB两层介质的折射率
n = 200  # 多层膜的层数

d = 40  # 每层膜大概的厚度
delta_d = 10  # 每层膜可变化范围

allowed_cost = 0.001  # 结束迭代时，允许的误差函数值
# di=np.array([random.uniform(d-delta_d,d+delta_d) for i in range(1,n+1)])   #随机生成每层膜厚度

lambda_min = 300
#lambda_max = 1600  # 光谱的最大和最小波长
lambda_step = 2  # 步长
lambda_num = 512

lambda_list = np.array([lambda_min+i*lambda_step for i in range(lambda_num)])


@jit
def N(i):
    if i % 2 == 1:
        return n1
    else:
        return n2  # 奇数层折射率为n1，偶数层折射率为n2


D = np.array([[[1, 1], [N(i), -N(i)]] for i in range(n)], dtype=complex)
D_inv = np.array([[[0.5, 0.5/N(i)], [0.5, -0.5/N(i)]]
                 for i in range(n)], dtype=complex)

D_dev = cuda.to_device(D)
D_inv_dev = cuda.to_device(D_inv)
lambda_list_dev = cuda.to_device(lambda_list)

# 为了方便画图，设为函数


@jit
def E_pre(lambda1, di):
    # 把最后一个单独算
    e = np.array([[1+0j, 0+0j], [1+0j, 0+0j]])  # 空气出射
    Pi = np.array([[0+0j, 0+0j], [0+0j, 0+0j]])
    # 中间的用循环算
    for i in range(n-1, -1, -1):
        Pi[0][0] = cmath.exp(-(0+1j)*2*cmath.pi*N(i)*di[i]/lambda1)
        Pi[1][1] = cmath.exp((0+1j)*2*cmath.pi*N(i)*di[i]/lambda1)
        e = D[i]@Pi@D_inv[i]@e

    # 第一个单独算
    Pi = np.array([[0.5+0j, 0.5+0j], [0.5+0j, -0.5+0j]])
    e = Pi @ e  # 入射为空气
    r = abs(e[1][0]/e[0][0])

    return r*r  # 反射率


@cuda.jit
def set_T(di, T, l_list, D_dev, D_inv_dev):  # 都需要是gpu中的数组！
    j = cuda.threadIdx.x
    i = cuda.blockIdx.x

    Ni = 0
    if i % 2 == 1:
        Ni = n1
    else:
        Ni = n2

    lambda1 = l_list[j]

    P0 = cmath.exp(-(0+1j)*2*cmath.pi*Ni*di[i]/lambda1)
    P1 = cmath.exp((0+1j)*2*cmath.pi*Ni*di[i]/lambda1)

    t00 = P0*D_inv_dev[i][0][0]
    t01 = P0*D_inv_dev[i][0][1]
    t10 = P1*D_inv_dev[i][1][0]
    t11 = P1*D_inv_dev[i][1][1]

    #tmp1 = P1*right_T[j][i][1][0]

    T[j][i][0][0] = D_dev[i][0][0]*t00+D_dev[i][0][1]*t10
    T[j][i][0][1] = D_dev[i][0][0]*t01+D_dev[i][0][1]*t11
    T[j][i][1][0] = D_dev[i][1][0]*t00+D_dev[i][1][1]*t10
    T[j][i][1][1] = D_dev[i][1][0]*t01+D_dev[i][1][1]*t11


def set_T_new(T_dev, di_host):
    di_dev = cuda.to_device(di_host)
    set_T[n, lambda_num](di_dev, T_dev, lambda_list_dev, D_dev, D_inv_dev)


@jit
def E_pre_2(T, spec_out):
    fullT = np.array([[0+0j, 0+0j], [0+0j, 0+0j]])

    for j in range(lambda_num):
        fullT = np.array([[0.5+0j, 0.5+0j], [0.5+0j, -0.5+0j]])
        for i in range(n):
            fullT = fullT@T[j][i]

        fullT = fullT@np.array([[1+0j, 0+0j], [1+0j, 0+0j]])
        # print(fullT)
        r = abs(fullT[1][0]/fullT[0][0])
        spec_out[j] = r*r


def E_pre_new(T_dev, di_host, spec_out):  # both on host
    #T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=complex)
    set_T_new(T_dev, di_host)
    E_pre_2(T_dev.copy_to_host(), spec_out)


def cost_func(out_spectrum, input_spectrum):  # 估值函数
    #f = np.linalg.norm(out_spectrum-input_spectrum)
    f = 0
    for j in range(lambda_num):
        f = f+(out_spectrum[j]-input_spectrum[j])**2
    # return f*f
    return f


def plot(input_spectrum, di):  # 画图
    #y1 = [E_pre(a, di) for a in lambda_list]
    l_list=[lambda_min+lambda_step*j for j in range(lambda_num)]
    y1 = [E_pre(l,di) for l in l_list]
    y2 = input_spectrum
    plt.xlabel('lambda')
    plt.ylabel('r')
    plt.title("Reflection spectrum")
    plt.plot(l_list, y1)
    plt.plot(lambda_list, y2)
    plt.show()


def plot_new(input_spectrum, spec_out):  # 画图
    y1 = spec_out
    y2 = input_spectrum
    plt.xlabel('lambda')
    plt.ylabel('r')
    plt.title("Reflection spectrum")
    plt.plot(lambda_list, y1)
    plt.plot(lambda_list, y2)
    plt.show()


@jit
def set_vars(T, di):
    Pji = np.array([[0+0j, 0+0j], [0+0j, 0+0j]])
    for j in range(lambda_num):
        for i in range(n):

            if i % 2 == 1:
                Ni = n1
            else:
                Ni = n2

            Pji[0][0] = cmath.exp(-(0+1j)*2 *
                                  cmath.pi*Ni*di[i]/lambda_list[j])
            Pji[1][1] = cmath.exp(
                (0+1j)*2*cmath.pi*Ni*di[i]/lambda_list[j])
            T[j][i] = D[i]@Pji@D_inv[i]


@jit
def initialize(T, left_T, right_T, di, spec_out, T_set=False):
    # 给定P、T，计算left、right等
    Pji = np.array([[0+0j, 0+0j], [0+0j, 0+0j]])
    if not(T_set):
        # set_T()
        set_vars(T, di)

    for j in range(lambda_num):
        left_T[j][0] = np.array([[0.5+0j, 0.5+0j], [0.5+0j, -0.5+0j]])
        # right_T[j][n-1] = np.array([[1, 1], [1, -1]]
        #                           )@np.array([[1, 0], [0, 0]])
        right_T[j][n-1] = np.array([[1+0j, 0+0j], [1+0j, 0+0j]])

        for i in range(1, n):
            left_T[j][i] = left_T[j][i-1]@T[j][i-1]
            left_T[j][i-1] = left_T[j][i-1]@D[i-1]  # 下一步再补上

        for i in range(n-2, -1, -1):
            right_T[j][i] = T[j][i+1]@right_T[j][i+1]
            right_T[j][i+1] = D_inv[i+1]@right_T[j][i+1]

        # 补上边界情况
        left_T[j][n-1] = left_T[j][n-1]@D[n-1]
        right_T[j][0] = D_inv[0]@right_T[j][0]

        Pji[0][0] = cmath.exp(-(0+1j)*2 *
                              cmath.pi*n2*di[0]/lambda_list[j])
        Pji[1][1] = cmath.exp(
            (0+1j)*2*cmath.pi*n2*di[0]/lambda_list[j])

        fullT = left_T[j][0]@Pji@right_T[j][0]

        spec_out[j] = abs(fullT[1][0]/fullT[0][0])**2

    # print(left_T)
    # print(right_T)


@cuda.jit
def get_J_gpu(spectrum_out, di, J, left_T, right_T, l_list):
    Ni = 0
    stp = 1.0

    j = cuda.threadIdx.x
    i = cuda.blockIdx.x

    if i % 2 == 1:
        Ni = n1
    else:
        Ni = n2

    lambda1 = l_list[j]

    P0 = cmath.exp(-(0+1j)*2*cmath.pi*Ni*(di[i]+stp)/lambda1)
    P1 = cmath.exp((0+1j)*2*cmath.pi*Ni*(di[i]+stp)/lambda1)

    tmp0 = P0*right_T[j][i][0][0]
    tmp1 = P1*right_T[j][i][1][0]

    fT0 = left_T[j][i][0][0]*tmp0+left_T[j][i][0][1]*tmp1
    fT1 = left_T[j][i][1][0]*tmp0+left_T[j][i][1][1]*tmp1

    r = abs(fT1/fT0)
    J[j][i] = (r*r-spectrum_out[j])/stp


def iterate_gpu(times, di, input_spectrum):
    damp = 0.05
    v = 2
    u = 1.05
    cost = 0
    start = 0

    global lambda_num
    global lambda_step

    J = np.zeros((lambda_num, n))
    JTJ = np.empty((n, n))
    #P = np.zeros((lambda_num, n, 2, 2), dtype=complex)
    T = np.zeros((lambda_num, n, 2, 2), dtype=complex)  # T=D*P*inverse(D)
    left_T = np.zeros((lambda_num, n, 2, 2), dtype=complex)
    right_T = np.zeros((lambda_num, n, 2, 2), dtype=complex)
    output = np.zeros(lambda_num)
    residual = np.empty(lambda_num)

    J_dev = cuda.device_array((lambda_num, n))
    #JTJ_dev = cuda.device_array((n, n))
    #P_dev = cuda.to_device(np.array((lambda_num, n, 2, 2), dtype=complex))
    T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=complex)
    left_T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=complex)
    right_T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=complex)

    residual_dev = cuda.device_array(lambda_num)
    output_dev = cuda.to_device(output)
    di_dev = cuda.to_device(di)

    T1 = np.zeros((lambda_num, n, 2, 2), dtype=complex)
    T2 = np.zeros((lambda_num, n, 2, 2), dtype=complex)
    output1 = np.zeros(lambda_num)
    output2 = np.zeros(lambda_num)

    initialize(T, left_T, right_T, di, output)

    for i in range(times):
        start = time()
        initialize(T, left_T, right_T, di, output, T_set=True)
        #plot_new(input_spectrum,output)
        residual = input_spectrum-output
        cost = cost_func(output, input_spectrum)
        print("init time:"+str(time()-start))

        # copy data
        left_T_dev = cuda.to_device(left_T)
        right_T_dev = cuda.to_device(right_T)
        output_dev = cuda.to_device(output)
        di_dev = cuda.to_device(di)

        #get_J_gpu(output_dev, di_dev, J_dev, left_T_dev, right_T_dev)
        start = time()
        get_J_gpu[n, lambda_num](output_dev, di_dev, J_dev, left_T_dev,
                                 right_T_dev, lambda_list_dev)
        print("Calc Jacobian:"+str(time()-start))
        # To be elaborated!!

        start = time()
        J = J_dev.copy_to_host()
        print("data copy time:"+str(time()-start))

        start = time()
        JTJ = J.T@J
        movement1 = np.linalg.inv(JTJ+damp*np.eye(n))@np.dot((J.T), residual)
        movement2 = np.linalg.inv(
            JTJ+(damp/v)*np.eye(n))@np.dot((J.T), residual)

        E_pre_new(T_dev, di+movement1, output1)
        T1 = T_dev.copy_to_host()

        E_pre_new(T_dev, di+movement2, output2)
        T2 = T_dev.copy_to_host()

        improve1 = cost-cost_func(output1, input_spectrum)
        improve2 = cost-cost_func(output2, input_spectrum)

        print("Decide improvement time:"+str(time()-start))

        if (improve1 < 0) and (improve2 < 0):
            print("both too bad!")
            print(improve1)
            print(improve2)
            #cost_func(output,input_spectrum)
            #E_pre_new(T_dev,di,output)
            #plot_new(input_spectrum,output)
            
            damp = damp*v

        else:
            if(improve1 > improve2):
                di = di+movement1
                T = T1
                output = output1
                print("improve1:")
                print(improve1)

                damp = damp*u

            else:
                print("improve2:")
                print(improve2)
                T = T2
                output = output2
                di = di+movement2
                damp = damp/v

        #output = np.array([E_pre(a, di) for a in lambda_list])
        cost = cost_func(output, input_spectrum)

        if cost < allowed_cost:  # 足够小了就退出
            plot(input_spectrum, di)
            print("cost small enough!!")
            return

        print("cost="+str(cost))

    print("damp="+str(damp))
    #plot(input_spectrum, di)

    #lambda_step=0.1
    #lambda_num=10240
    plot(input_spectrum,di)
    #print(di)
    return di
    


def main():

    #spectrum = np.zeros(lambda_num)

    initial_d = np.array([d for i in range(n)])

    #goal_spectrum = np.array([0.4*(math.sin(l/20)**2) for l in lambda_list])
    goal_spectrum = np.array([0.4*(math.sin(l/20)**2)
                             for l in lambda_list])

    # 目标光谱
    # print(goal_spectrum)

    plot(goal_spectrum, initial_d)  # 初始光谱

    iternum = int(input("iternum:"))
    #iterate(iternum, initial_d, goal_spectrum)
    iterate_gpu(iternum, initial_d, goal_spectrum)


if __name__ == "__main__":
    # print(cmath.pi-math.pi)
    main()
