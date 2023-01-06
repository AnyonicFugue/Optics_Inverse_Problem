# 多层膜正问题

# 函数引入
from __future__ import division
from functools import lru_cache
import random
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
n = 2048  # 多层膜的层数

d = 40  # 每层膜大概的厚度
delta_d = 10  # 每层膜可变化范围

allowed_cost = 0.001  # 结束迭代时，允许的误差函数值
# di=np.array([random.uniform(d-delta_d,d+delta_d) for i in range(1,n+1)])   #随机生成每层膜厚度

lambda_min = 300
# lambda_max = 1600  # 光谱的最大和最小波长
lambda_step = 0.25  # 步长
lambda_num = 4096

blocksize = 256
blocknum = math.ceil(lambda_num/blocksize)

T_blocksize = 64
T_blocknum = math.ceil(lambda_num/T_blocksize)

lambda_list = np.array([lambda_min+i*lambda_step for i in range(lambda_num)])


@jit
def N(i):
	if i % 2 == 1:
		return n1
	else:
		return n2  # 奇数层折射率为n1，偶数层折射率为n2


D = np.array([[[1, 1], [N(i), -N(i)]] for i in range(n)], dtype=np.complex64)
D_inv = np.array([[[0.5, 0.5/N(i)], [0.5, -0.5/N(i)]]
				 for i in range(n)], dtype=np.complex64)

D_dev = cuda.to_device(D)
D_inv_dev = cuda.to_device(D_inv)
lambda_list_dev = cuda.to_device(lambda_list)


@cuda.jit
def set_T_kernel(di, T, l_list, D_dev, D_inv_dev):  # 都需要是gpu中的数组！

	j = cuda.blockIdx.y*blocksize+cuda.threadIdx.x
	i = cuda.blockIdx.x

	if(j >= lambda_num):
		return

	tmp = cuda.local.array((2, 2), dtype=np.complex64)

	Ni = 0
	if i % 2 == 1:
		Ni = n1
	else:
		Ni = n2

	lambda1 = l_list[j]

	# T=D*P*Dinv

	# T=P

	T[j][i][0][0] = cmath.exp(-(0+1j)*2*cmath.pi*Ni*di[i]/lambda1)
	T[j][i][1][1] = cmath.exp((0+1j)*2*cmath.pi*Ni*di[i]/lambda1)
	T[j][i][0][1] = 0
	T[j][i][1][0] = 0

	# T=T*D_inv

	for x in range(2):
		for y in range(2):
			tmp[x][y] = 0
			for k in range(2):
				tmp[x][y] = tmp[x][y]+T[j][i][x][k]*D_inv_dev[i][k][y]

	for x in range(2):
		for y in range(2):
			T[j][i][x][y] = tmp[x][y]

	# T=D*T

	for x in range(2):
		for y in range(2):
			tmp[x][y] = 0
			for k in range(2):
				tmp[x][y] = tmp[x][y]+D_dev[i][x][k]*T[j][i][k][y]

	for x in range(2):
		for y in range(2):
			T[j][i][x][y] = tmp[x][y]


def set_T(T_dev, di_host):
	di_dev = cuda.to_device(di_host)
	set_T_kernel[(n, blocknum), blocksize](
		di_dev, T_dev, lambda_list_dev, D_dev, D_inv_dev)


@jit
def E_pre_2(T, spec_out):
	fullT = np.array([[0+0j, 0+0j], [0+0j, 0+0j]], dtype=np.complex64)

	for j in range(lambda_num):
		fullT = np.array(
			[[0.5+0j, 0.5+0j], [0.5+0j, -0.5+0j]], dtype=np.complex64)
		for i in range(n):
			fullT = fullT@T[j][i]

		fullT = fullT@np.array([[1+0j, 0+0j], [1+0j, 0+0j]],
							   dtype=np.complex64)
		# print(fullT)
		r = abs(fullT[1][0]/fullT[0][0])
		spec_out[j] = r*r

# 为了方便画图，设为函数


def E_pre(T_dev, di_host, spec_out):  # both on host
	#T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=np.complex64)
	set_T(T_dev, di_host)
	E_pre_2(T_dev.copy_to_host(), spec_out)


@cuda.jit
def set_left_T_kernel(T, left_T):  # both need to be dev!

	j = cuda.blockIdx.x*T_blocksize+cuda.threadIdx.x

	if(j >= lambda_num):
		return

	left_T[j][0][0][0] = 0.5+0j
	left_T[j][0][0][1] = 0.5+0j
	left_T[j][0][1][0] = 0.5+0j
	left_T[j][0][1][1] = -0.5+0j

	for i in range(1, n):
		#left_T[j][i] = left_T[j][i-1]@T[j][i-1]
		for x in range(2):
			for y in range(2):
				left_T[j][i][x][y] = 0
				for k in range(2):
					left_T[j][i][x][y] = left_T[j][i][x][y] + \
						left_T[j][i-1][x][k]*T[j][i-1][k][y]


@cuda.jit
def set_right_T_kernel(T, right_T, spec_out):

	j = cuda.blockIdx.x*T_blocksize+cuda.threadIdx.x

	if(j >= lambda_num):
		return

	right_T[j][n-1][0][0] = 1+0j
	right_T[j][n-1][0][1] = 0+0j
	right_T[j][n-1][1][0] = 1+0j
	right_T[j][n-1][1][1] = 0+0j

	for i in range(n-2, -1, -1):
		#right_T[j][i] = T[j][i+1]@right_T[j][i+1]
		for x in range(2):
			for y in range(2):
				right_T[j][i][x][y] = 0
				for k in range(2):
					right_T[j][i][x][y] = right_T[j][i][x][y] + \
						T[j][i+1][x][k]*right_T[j][i+1][k][y]

	# fT=D_inv_0*T0*right_T[j][0]
	tmp = cuda.local.array((2, 2), dtype=np.complex64)
	fT = cuda.local.array((2, 2), dtype=np.complex64)

	# fT=D_inv_0
	fT[0][0] = 0.5+0j
	fT[0][1] = 0.5+0j
	fT[1][0] = 0.5+0j
	fT[1][1] = -0.5+0j

	# fT=fT*T0
	for x in range(2):
		for y in range(2):
			tmp[x][y] = 0
			for k in range(2):
				tmp[x][y] = tmp[x][y]+fT[x][k]*T[j][0][k][y]

	for x in range(2):
		for y in range(2):
			fT[x][y] = tmp[x][y]

	# fT=fT*right_T[j][0]
	for x in range(2):
		for y in range(2):
			tmp[x][y] = 0
			for k in range(2):
				tmp[x][y] = tmp[x][y]+fT[x][k]*right_T[j][0][k][y]

	for x in range(2):
		for y in range(2):
			fT[x][y] = tmp[x][y]

	r = abs(fT[1][0]/fT[0][0])
	spec_out[j] = r*r


def cost_func(out_spectrum, input_spectrum):  # 估值函数
	f = np.linalg.norm(out_spectrum-input_spectrum)
	return f


def plot(input_spectrum, spec_out):  # 画图
	y1 = spec_out
	y2 = input_spectrum
	plt.xlabel('lambda')
	plt.ylabel('r')
	plt.title("Reflection spectrum")
	plt.plot(lambda_list, y1)
	plt.plot(lambda_list, y2)
	plt.show()


def initialize_gpu(T_dev, left_T_dev, right_T_dev, di, spec_out, spec_only=False, T_set=False):

	if not(T_set):
		set_T(T_dev, di)

	spec_out_dev = cuda.device_array(lambda_num)

	if not(spec_only):
		set_left_T_kernel[T_blocknum, T_blocksize](T_dev, left_T_dev)

	set_right_T_kernel[T_blocknum, T_blocksize](
		T_dev, right_T_dev, spec_out_dev)

	cuda.synchronize()

	spec_out_dev.copy_to_host(ary=spec_out)


@cuda.jit
def get_J_gpu(spectrum_out, di, J, left_T, right_T, l_list, D_dev, D_inv_dev):
	Ni = 0
	stp = 1.0

	j = cuda.blockIdx.y*blocksize+cuda.threadIdx.x
	i = cuda.blockIdx.x

	if(j >= lambda_num):
		return

	tmp = cuda.local.array((2, 2), dtype=np.complex64)
	fullT = cuda.local.array((2, 2), dtype=np.complex64)

	if i % 2 == 1:
		Ni = n1
	else:
		Ni = n2

	lambda1 = l_list[j]

	# fullT=left_T*D*P*D_inv*right_T

	# T=P

	fullT[0][0] = cmath.exp(-(0+1j)*2*cmath.pi*Ni*(di[i]+stp)/lambda1)
	fullT[1][1] = cmath.exp((0+1j)*2*cmath.pi*Ni*(di[i]+stp)/lambda1)
	fullT[1][0] = 0
	fullT[0][1] = 0

	# T=T*D_inv
	for x in range(2):
		for y in range(2):
			tmp[x][y] = 0
			for k in range(2):
				tmp[x][y] = tmp[x][y]+fullT[x][k]*D_inv_dev[i][k][y]

	for x in range(2):
		for y in range(2):
			fullT[x][y] = tmp[x][y]

	# T=D*T
	for x in range(2):
		for y in range(2):
			tmp[x][y] = 0
			for k in range(2):
				tmp[x][y] = tmp[x][y]+D_dev[i][x][k]*fullT[k][y]

	for x in range(2):
		for y in range(2):
			fullT[x][y] = tmp[x][y]

	# T=T*right_T
	for x in range(2):
		for y in range(2):
			tmp[x][y] = 0
			for k in range(2):
				tmp[x][y] = tmp[x][y]+fullT[x][k]*right_T[j][i][k][y]

	for x in range(2):
		for y in range(2):
			fullT[x][y] = tmp[x][y]

	# T=left_T*T
	for x in range(2):
		for y in range(2):
			tmp[x][y] = 0
			for k in range(2):
				tmp[x][y] = tmp[x][y]+left_T[j][i][x][k]*fullT[k][y]

	for x in range(2):
		for y in range(2):
			fullT[x][y] = tmp[x][y]

	r = abs(fullT[1][0]/fullT[0][0])
	J[j][i] = (r*r-spectrum_out[j])/stp


def iterate_gpu(di, input_spectrum):
	damp = 0.005
	v = 2
	u = 1.05
	cost = 0
	start = 0

	global lambda_num
	global lambda_step

	J = np.zeros((lambda_num, n))
	JTJ = np.empty((n, n))
	output = np.zeros(lambda_num)
	residual = np.empty(lambda_num)

	J_dev = cuda.device_array((lambda_num, n))
	T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=np.complex64)
	left_T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=np.complex64)
	right_T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=np.complex64)

	output_dev = cuda.to_device(output)
	di_dev = cuda.to_device(di)

	output1 = np.zeros(lambda_num)
	output2 = np.zeros(lambda_num)

	initialize_gpu(T_dev, left_T_dev, right_T_dev, di, output)
	plot(input_spectrum, output)

	times = 0
	times = int(input("iterate num:"))

	while(times > 0):

		for i in range(times):
			iter_start = time()
			start = time()
			initialize_gpu(T_dev, left_T_dev, right_T_dev, di, output)

			# plot(input_spectrum,output)
			residual = input_spectrum-output
			cost = cost_func(output, input_spectrum)
			print("init time:"+str(time()-start))

			# copy data
			output_dev = cuda.to_device(output)
			di_dev = cuda.to_device(di)

			#get_J_gpu(output_dev, di_dev, J_dev, left_T_dev, right_T_dev)
			start = time()
			get_J_gpu[(n, blocknum), blocksize](output_dev, di_dev, J_dev, left_T_dev,
												right_T_dev, lambda_list_dev, D_dev, D_inv_dev)
			cuda.synchronize()
			print("Calc Jacobian time:"+str(time()-start))
			# To be elaborated!!

			J = J_dev.copy_to_host()

			start = time()
			JTJ = J.T@J
			movement1 = np.linalg.inv(
				JTJ+damp*np.eye(n))@np.dot((J.T), residual)
			movement2 = np.linalg.inv(
				JTJ+(damp/v)*np.eye(n))@np.dot((J.T), residual)

			#E_pre(T_dev, di+movement1, output1)
			initialize_gpu(T_dev, left_T_dev, right_T_dev, di +
						   movement1, output1, spec_only=True)

			#E_pre(T_dev, di+movement2, output2)
			initialize_gpu(T_dev, left_T_dev, right_T_dev, di +
						   movement2, output2, spec_only=True)
			#T2 = T_dev.copy_to_host()

			improve1 = cost-cost_func(output1, input_spectrum)
			improve2 = cost-cost_func(output2, input_spectrum)

			print("Decide improvement time:"+str(time()-start))

			if (improve1 < 0) and (improve2 < 0):
				print("both too bad!")
				print(improve1)
				print(improve2)
				damp = damp*v

			else:
				if(improve1 > improve2):
					di = di+movement1
					#T = T1
					output = output1
					print("improve1:")
					print(improve1)

					damp = damp*u

				else:
					print("improve2:")
					print(improve2)
					#T = T2
					output = output2
					di = di+movement2
					damp = damp/v

				cost = cost_func(output, input_spectrum)

				if cost < allowed_cost:  # 足够小了就退出
					plot(input_spectrum, output)
					print("cost small enough!!")
					return

				print("\ncost="+str(cost))

			print("iteration time:"+str(time()-iter_start)+'\n')

		print("damp="+str(damp))

		plot(input_spectrum, output)
		times = int(input("iterate num:"))
	# print(di)


def main():

	#spectrum = np.zeros(lambda_num)

	initial_d = np.array([d for i in range(n)])

	#goal_spectrum = np.array([0.4*(math.sin(l/20)**2) for l in lambda_list])
	goal_spectrum = np.array([0.4*(math.sin(l/20)**2)
							 for l in lambda_list])



	iterate_gpu(initial_d, goal_spectrum)


if __name__ == "__main__":
	# print(cmath.pi-math.pi)
	main()
