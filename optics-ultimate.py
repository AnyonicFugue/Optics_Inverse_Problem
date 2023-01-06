# 多层膜正问题

# 函数引入
from __future__ import division
import random
import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import cmath
import xlrd
import xlwt

from time import time
from numba import cuda
from numba import jit


# 统一改为以纳米为单位，方便计算

# 几何与材料定义部分


d = 40  # 每层膜大概的厚度
delta_d = 10  # 每层膜可变化范围

allowed_cost = 0.001  # 结束迭代时，允许的误差函数值
# di=np.array([random.uniform(d-delta_d,d+delta_d) for i in range(1,n+1)])   #随机生成每层膜厚度

lambda_min = 300
lambda_max = 1000  # 光谱的最大和最小波长

n1, n2 = 1.33, 1.67  # AB两层介质的折射率
n = 1024  # 多层膜的层数
lambda_num = 4096 #采样点数量

lambda_list = np.zeros(lambda_num,dtype=np.float32)
weight_list=np.zeros(lambda_num,dtype=np.float32)

l_list = np.linspace(1000/lambda_min, 1000/lambda_max, lambda_num)
for i in range(lambda_num):
	lambda_list[i] = 1000/l_list[i]
	weight_list[i]=1

weight=5



lambda_list_dev = cuda.to_device(lambda_list)
weight_list_dev=cuda.to_device(weight_list)

blocksize = 256
blocknum = math.ceil(lambda_num/blocksize)

T_blocksize = 32
T_blocknum = math.ceil(lambda_num/T_blocksize)


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


@jit
def cost_func(out_spectrum, input_spectrum):  # 估值函数
	f=0
	'''
	for j in range(lambda_num):
		f=f+weight_list[j]*(out_spectrum[j]-input_spectrum[j])**2
	'''
	f = np.linalg.norm(out_spectrum-input_spectrum)
	return f


def plot(input_spectrum, spec_out):  # 画图
	y1 = np.array(spec_out)
	y2 = np.array(input_spectrum)
	remove_spec_weight(y1)
	remove_spec_weight(y2)
	plt.xlabel('lambda')
	plt.ylabel('r')
	plt.title("Reflection spectrum")
	plt.plot(lambda_list, y1)
	plt.plot(lambda_list, y2)
	plt.show()

@jit
def set_spec_weight(res):
	for j in range(lambda_num):
		res[j]=res[j]*weight_list[j]

@jit
def remove_spec_weight(res):
	for j in range(lambda_num):
		res[j]=res[j]/weight_list[j]

def initialize_gpu(T_dev, left_T_dev, right_T_dev, di, spec_out, spec_only=False, T_set=False):

	if not(T_set):
		set_T(T_dev, di)

	spec_out_dev = cuda.device_array(lambda_num, dtype=np.float32)

	if not(spec_only):
		set_left_T_kernel[T_blocknum, T_blocksize](T_dev, left_T_dev)

	set_right_T_kernel[T_blocknum, T_blocksize](
		T_dev, right_T_dev, spec_out_dev)

	cuda.synchronize()

	spec_out_dev.copy_to_host(ary=spec_out)
	set_spec_weight(spec_out)


@cuda.jit
def get_J_gpu(spectrum_out, di, J, left_T, right_T, l_list, D_dev, D_inv_dev,weight_list):

	j = cuda.blockIdx.y*blocksize+cuda.threadIdx.x
	i = cuda.blockIdx.x

	Ni = 0
	stp = 0.5


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
	J[j][i] = (weight_list[j]*(r*r)-spectrum_out[j])/stp
	

def iterate_gpu(di, input_spectrum):
	damp = 0.005
	v = 1.5
	u = 1.05
	cost = 0
	start = 0

	global lambda_num

	J = np.zeros((lambda_num, n), dtype=np.float32)
	JTJ = np.empty((n, n), dtype=np.float32)

	output = np.zeros(lambda_num, dtype=np.float32)
	residual = np.empty(lambda_num, dtype=np.float32)

	J_dev = cuda.device_array((lambda_num, n), dtype=np.float32)
	# JTJ_dev=cuda.device_array((lambda_num,n,2,2),dtype=np.float32)

	T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=np.complex64)
	left_T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=np.complex64)
	right_T_dev = cuda.device_array((lambda_num, n, 2, 2), dtype=np.complex64)

	output_dev = cuda.to_device(output)
	di_dev = cuda.to_device(di)

	output1 = np.zeros(lambda_num, dtype=np.float32)
	output2 = np.zeros(lambda_num, dtype=np.float32)

	initialize_gpu(T_dev, left_T_dev, right_T_dev, di, output)
	plot(input_spectrum, output)

	id_mat = np.identity(n, dtype=np.float32)

	np.setbufsize(2048*4096)

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

			if cost < allowed_cost:  # 足够小了就退出
					plot(input_spectrum, output)
					print("cost small enough!!")
					return

			print("cost="+str(cost)+'\n')

			print("init time:"+str(time()-start))

			# copy data
			output_dev = cuda.to_device(output)
			di_dev = cuda.to_device(di)

			#get_J_gpu(output_dev, di_dev, J_dev, left_T_dev, right_T_dev)
			start = time()
			get_J_gpu[(n, blocknum), blocksize](output_dev, di_dev, J_dev, left_T_dev,
												right_T_dev, lambda_list_dev, D_dev, D_inv_dev,weight_list_dev)
			cuda.synchronize()
			print("Calc Jacobian time:"+str(time()-start))
			# To be elaborated!!

			start = time()
			J = J_dev.copy_to_host()
			JTJ = J.T@J

			#set_spec_weight(residual)

			movement1 = np.linalg.inv(
				JTJ+damp*id_mat)@np.dot((J.T), residual)
			movement2 = np.linalg.inv(
				JTJ+(damp/v)*id_mat)@np.dot((J.T), residual)
			print("matrix calc time:"+str(time()-start))

			start = time()
			initialize_gpu(T_dev, left_T_dev, right_T_dev, di +
						   movement1, output1, spec_only=True)

			initialize_gpu(T_dev, left_T_dev, right_T_dev, di +
						   movement2, output2, spec_only=True)
			#T2 = T_dev.copy_to_host()

			improve1 = cost-cost_func(output1, input_spectrum)
			improve2 = cost-cost_func(output2, input_spectrum)

			print("Decide improvement time:"+str(time()-start))

			if (improve1 < 0) and (improve2 < 0):
				print("both too bad!")
				#print(improve1)
				#print(improve2)
				damp = damp*(v**4)

			else:
				if(improve1 > improve2):
					di = di+movement1
					output = output1
					print("improve1:")
					#print(improve1)
					if(improve1>0.01):
						damp = damp*u
					else:
						damp=damp/(u**2)

				else:
					print("improve2:")
					#print(improve2)
					output = output2
					di = di+movement2
					damp = damp/v

			print("\niteration time:"+str(time()-iter_start))

		print("\ndamp="+str(damp))
		print("cost="+str(cost)+'\n')
		plot(input_spectrum, output)
		times = int(input("iterate num:"))

	# print(di)
	write_excel(di,output)

def write_excel(di,output):
	wb=xlwt.Workbook()
	d_sheet=wb.add_sheet('Widths',cell_overwrite_ok=True)
	spectrum_sheet=wb.add_sheet('spectrum',cell_overwrite_ok=True)

	for i in range(n):
		d_sheet.write(i,0,float(di[i]))

	for j in range(lambda_num):
		spectrum_sheet.write(j,1,float(output[j]))
		spectrum_sheet.write(j,0,float(lambda_list[j]))
	
	wb.save("output.xls")

def set_sine(l, amplitude):
	spec = np.zeros(lambda_num, dtype=np.float32)

	for i in range(lambda_num):
		lambda1 = lambda_list[i]
		spec[i] = amplitude*(math.sin(lambda1/l))**2

	return spec



def set_random(amplitude):
	spec = np.zeros(lambda_num, dtype=np.float32)

	for i in range(lambda_num):
		spec[i] = amplitude*(random.random())

	return spec


def set_sawtooth(l, amplitude):
	spec = np.zeros(lambda_num, dtype=np.float32)

	for i in range(lambda_num):
		lambda1 = lambda_list[i]
		spec[i] = amplitude*((lambda1 % l)/l)

	return spec


def add_peak(center, width, spec):
	for i in range(lambda_num):
		lambda1 = lambda_list[i]
		if(center-width/2 < lambda1) and (center+width/2 > lambda1):
			spec[i] = weight
			weight_list[i]=weight



def add_sawtooth(center,width,spec):
	for i in range(lambda_num):
		lambda1 = lambda_list[i]
		if(center-width/2 < lambda1) and (center+width/2 > lambda1):
			spec[i] = weight*2*abs(lambda1-center)/width
			weight_list[i]=weight
	



def get_excel_spectrum(file_location,sheet_index):
	global lambda_num
	global lambda_list

	data = xlrd.open_workbook(file_location)
	sheet = data.sheet_by_index(sheet_index) #打开sheet，（）内0代表sheet1，1代表sheet2，类推

	lambda_list=np.array([sheet.cell_value(r,0) for r in range(0,sheet.nrows)])
	spec_list=np.array([sheet.cell_value(r,1) for r in range(0,sheet.nrows)])

	lambda_num=lambda_list.size

	return spec_list
	
def get_excel_widths(file_location,sheet_index):
	global n

	data = xlrd.open_workbook(file_location)
	sheet = data.sheet_by_index(sheet_index) #打开sheet，（）内0代表sheet1，1代表sheet2，类推

	d_list=np.array([sheet.cell_value(r,0) for r in range(0,sheet.nrows)])


	n=d_list.size
	return d_list

def flip(spec_list):
	remove_spec_weight(spec_list)
	for i in range(lambda_num):
		spec_list[i]=1-spec_list[i]
	set_spec_weight(spec_list)

def main():

	global lambda_num
	global lambda_list
	global lambda_list_dev
	global weight_list_dev

	goal_spectrum = np.zeros(lambda_num, dtype=np.float32)

	initial_d = np.array([d for i in range(n)],dtype=np.float32)

	#goal_spectrum = set_sine(10, 0.3)
	#goal_spectrum = set_sawtooth(5, 0.4)
	add_peak(500, 60,goal_spectrum)
	#flip(goal_spectrum)
	#add_peak(700, 10,goal_spectrum)

	#goal_spectrum=get_excel_spectrum("building0.xls",1)

	#initial_d=get_excel_widths("output.xls",0)

	lambda_list_dev = cuda.to_device(lambda_list)
	weight_list_dev=cuda.to_device(weight_list)

	iterate_gpu(initial_d, goal_spectrum)



if __name__ == "__main__":
	# print(cmath.pi-math.pi)
	main()
