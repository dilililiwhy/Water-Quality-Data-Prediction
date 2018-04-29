# _*_ coding: utf-8 _*_

"""
使用状态池网络对水质数据进行拟合

"""
#%matplotlib inline 
from . import RC_network
from . import RC_excel
from . import RC_database
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter 
import datetime
import pandas as pd

def load_saved_data(name='',path='./'):
    save_path = path+name
    print ('从[{}]读取数据'.format(save_path))
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
        f.close()

    return data

def display_inputdata(input_data,time_data=None,label=['']):
	colorstyle = ['r','b','g','tan','k','y','m']
	yearsFmt = DateFormatter('%Y-%m') 
	autodates = AutoDateLocator() 

	fig=plt.figure(figsize=(16,12),dpi = 160)
	ax1=fig.add_subplot(211)
	fig.autofmt_xdate()        #设置x轴时间外观  
	ax1.xaxis.set_major_locator(autodates)       #设置时间间隔  
	ax1.xaxis.set_major_formatter(yearsFmt)      #设置时间显示格式   
	

	plt.title('WaterQualite')
	plt.grid(True)
	plt.xlabel("year-month")
	time_begin = datetime.datetime(1990,1,1)
	time_end = datetime.datetime(2007,12,1)
	plt.xlim(time_begin, time_end)

	plt.ylabel("WaterQualite")


	if time_data != None:
		x = time_data
		for i in range(np.shape(input_data)[0]):
			y = np.array(input_data[i,:])[0]
			plt.plot(x,y,colorstyle[i], linewidth=2.0, label=label[i])
		plt.legend(loc="upper left", shadow=True)
	plt.show()


if __name__ == '__main__':
	input_name = ['TN','TP']
	output_name = ["Chla"]
	test_in_data_path_1 = '/Users/zhutq/mystuff/gdals/Reservoir_Computing/new/THL03-总氮'
	test_in_data_path_2 = '/Users/zhutq/mystuff/gdals/Reservoir_Computing/new/THL03-总磷'
	test_out_data_path = '/Users/zhutq/mystuff/gdals/Reservoir_Computing/new/THL03-叶绿素a'

	input_data_path_1 = '/Users/zhutq/mystuff/gdals/Reservoir_Computing/new/THL00-总氮'
	input_data_path_2 = '/Users/zhutq/mystuff/gdals/Reservoir_Computing/new/THL00-总磷'
	output_data_path = '/Users/zhutq/mystuff/gdals/Reservoir_Computing/new/THL00-叶绿素a'
	date_frame_path = '/Users/zhutq/mystuff/gdals/Reservoir_Computing/new/date_frame'

	print ('读取输入数据')
	input_data1 = load_saved_data(path = input_data_path_1)
	input_data2 = load_saved_data(path = input_data_path_2)
	input_data = np.zeros([2,len(input_data1)])
	input_data = np.mat(input_data)
	input_data[0,:] = np.mat(input_data1) 
	input_data[1,:] = np.mat(input_data2)
	print ('读取输出数据')
	output_data = load_saved_data(path = output_data_path)
	output_data = np.mat(output_data)
	print ('读取时间轴')
	date_frame = load_saved_data(path = date_frame_path)
	print ('读取测试输入数据')
	test_in_data1 = load_saved_data(path = test_in_data_path_1)
	test_in_data2 = load_saved_data(path = test_in_data_path_2)
	test_in_data = np.zeros([2,len(test_in_data1)])
	test_in_data = np.mat(test_in_data)
	test_in_data[0,:] = np.mat(test_in_data1) 
	test_in_data[1,:] = np.mat(test_in_data2)
	print ('读取测试输出数据')
	test_out_data = load_saved_data(path = test_out_data_path)
	test_out_data = np.mat(test_out_data)
	

	num_input,num = np.shape(input_data)
	num_output = np.shape(output_data)[0]
	print ('输入数据的维度为{}'.format(np.shape(input_data)))
	print ('其中，每组数据含有{}个数据，共{}组'.format(num_input,num))

	'''	
	df = pd.DataFrame(input_data.T,index=date_frame,columns=input_name)
	df2 = pd.DataFrame(output_data.T,index=date_frame,columns=output_name)
	df3 = df.join(df2)
	df3.plot()
	'''
	print ("--------------------------")
	print ("训练数据如下所示：")
	print ("训练输入数据为{}".format(input_name))
	df_inputdata = pd.DataFrame(input_data).T
	df_inputdata.columns = input_name
	df_inputdata.plot(figsize=(16,12))
	print ("训练输出数据为{}".format(output_name))
	df_outputdata = pd.DataFrame(output_data).T
	df_outputdata.columns = output_name
	df_outputdata.plot(figsize=(16,12))
	print ("--------------------------")
	print ("测试数据如下所示：")
	print ("测试输入数据为{}".format(input_name))
	df_testinput = pd.DataFrame(test_in_data).T
	df_testinput.columns = input_name
	df_testinput.plot(figsize=(16,12))
	print ("测试输出数据为{}".format(output_name))
	df_testoutput = pd.DataFrame(test_out_data).T
	df_testoutput.columns = output_name
	df_testoutput.plot(figsize=(16,12))
	print ("--------------------------")


	dt = 0.1
	t = 10
	g= 1.5

	print ("设定的g值={}".format(g))
	network = RC_network.network(num_input=num_input,num_output=num_output,g=g,dt=dt)
	RC = RC_network.Reservoir_Computing(input_data=input_data,output_data=output_data,network=network,t=t)
	#RC.display(np.array(input_data)[0],np.array(output_data)[0],RC.simtime)



	#RC_network.display(zt[0],ft,simtime)
	RC.updata_testdata(test_in_data,test_out_data)
	RC.training(num_of_train=1000,test_flag=True,time_num=100)
	
	
	#多次训练
	#network_1 =  RC_network.network(num_input=num_input,num_output=num_output,g=g,dt=dt)

	#RC1 = RC_network.Reservoir_Computing(input_data=input_data,output_data=output_data,network=network_1,t=t)

	#RC1.training(......)
	
	#network_2 = RC1.network

	#RC2 = RC_network.Reservoir_Computing(input_data=input_data,output_data=output_data,network=network_2,t=t)
	
	
	
	