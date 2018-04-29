# -*- coding: utf-8 -*-
"""
本程序是用来提取excel中的水质参数列表
并按照不同的观测站，进行时间上的排列绘制。
缺省值省略不画
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
import datetime
import pickle
#%matplotlib inline 


def draw_WQ(input_data,date_frame,label='label',title='plot title',save_flag=False,save_path='',plt_flag = True):
	yearsFmt = DateFormatter('%Y-%m') 
	autodates = AutoDateLocator() 

	fig=plt.figure(figsize=(16,12),dpi = 160)
	ax1=fig.add_subplot(211)
	ax1.xaxis.set_major_formatter(yearsFmt)
	fig.autofmt_xdate()        #设置x轴时间外观  
	ax1.xaxis.set_major_locator(autodates)       #设置时间间隔  
	ax1.xaxis.set_major_formatter(yearsFmt)      #设置时间显示格式   
	

	plt.title(title)
	plt.grid(True)
	plt.xlabel("year-month")
	time_begin = datetime.datetime(1990,1,1)
	time_end = datetime.datetime(2007,12,1)
	plt.xlim(time_begin, time_end)

	plt.ylabel("WaterQualite")

	ax1.plot(date_frame, input_data,"b-", linewidth=2.0, label=label)
	ax1.legend(loc="upper left", shadow=True)


	#x = range(0,len(date_frame))

	if save_flag == True:
		if save_path == '':
			plt.savefig(title)
		else:
			if (not os.path.exists(save_path)):
				print('创建文件夹{}'.format(save_path))
				os.mkdir(save_path)
			path = save_path+'/'+title
			print ('正在保存图片至{}'.format(path))
			plt.savefig(path)
	if plt_flag == True:
		plt.show()
	return

def save_excel(water_df,path=''):
	if path == '':
		print ('正在保存excel至文件运行路径')
	else:
		print ('正在保存excel至 {} 路径'.format(path))
	writer = pd.ExcelWriter(path+'water_df.xlsx', engine='xlsxwriter')
	water_df.to_excel(writer, 'Sheet1')
	writer.save()

def Min_Max_Normalization(array):
	N_array = []
	for x in array:
		x = float(x - np.min(array))/(np.max(array)- np.min(array))
		N_array.append(x)
	return N_array

def Mean_Normalization(array):
	N_array = []
	for x in array:
		x = float(x - array.mean())/array.std()
		N_array.append(x)
	return N_array
    


def screen_excel(excel_path,WQ_name):
	"""
	excel_path  为excel文件路径

	WQ_name   为要提取的水质参数名称，
	可为部分名称，但是不能与其他水质参数重复


	"""
	if os.path.exists(excel_path) and os.path.isfile(excel_path):
		print ('成功打开{}的水质表'.format(excel_path))
		df = pd.read_csv(excel_path)
	else:
		print ('打开错误，请检查水质表的文件路径')
		return
	WaterQualiteId_lists = WQ_name
	WQId_name = [] #由于输入的id为简称，需要识别全称
	col_list = ['time','station']
	print ('水质参数列表为 {}'.format(WaterQualiteId_lists))
	for WaterQualiteId in WaterQualiteId_lists:
		waterId_name = df.filter(regex=WaterQualiteId).columns.values[0]
		WQId_name.append(waterId_name)
		col_list.append(waterId_name)
	print ('col_list is {}'.format(col_list))
	print ('WQId_name is {}'.format(WQId_name))
	water_df = df[col_list].copy()
	water_df = water_df.replace(r'\s+', np.nan, regex=True)
	water_df = water_df.reset_index(drop=True)	
	#water_df = water_df.dropna()  
	water_df = water_df.fillna(0)
	return water_df,WQId_name


def trans_date(date_frame):
	year_month_frame = []
	for date in date_frame:
		year_month = datetime.datetime(int(date[0]/100),int(date[0]%100),15)
		year_month_frame.append(year_month)
	return year_month_frame


def save_image(excel_path,ob_lists,WQ_name,save_path=''):
	water_df,WQId_name = screen_excel(excel_path,WQ_name)
	df = water_df.copy() 
	for ob in ob_lists:
		for name in WQ_name:
			print ('正在读取{}观测站的数据'.format(ob))
			input_data = df[df['station'].isin(([ob]))][[2]].values
			#print ('input_data的类型是{}'.format(type(input_data)))
			#input_data = input_data.astype(np.float)
			date_frame = df[df['station'].isin(([ob]))][[0]].values

			input_data = Mean_Normalization(input_data)
			#input_data = Min_Max_Normalization(input_data)
			date_frame = trans_date(date_frame)
			title = ob+'-'+name

			save_data(input_data,title,save_path)
			#save_data(date_frame,'date_frame',save_path)
			
			draw_WQ(input_data,date_frame,label=name,title=title,save_flag=True,save_path=save_path)
	return 

def save_data(data,name,path='./'):
	global save_path
	if (not os.path.exists(path)):
		print('数据保存路径文件夹不存在，创建文件夹{}'.format(save_path))
		os.mkdir(path)
	save_path = path+'/'+ name
	print ('把数据保存到[{}]'.format(save_path))
	with open(save_path, 'wb') as f:
		pickle.dump(data, f) 



if __name__ == '__main__':
	excel_path = r'E:\Jupyter\201705\water_quality.csv'
	save_path = r'E:\Jupyter\201705\data_filter'
	WQ_name = ['TP']
	title = "TH00"
	ob_lists = ['THL00','THL01','THL03','THL04','THL05','THL06','THL07','THL08']


	test_ob = ['THL01']
	#print ('hello world')
	save_image(excel_path=excel_path,ob_lists=ob_lists,WQ_name=WQ_name,save_path=save_path)

	


	"""
	water_df,WQId_name = screen_excel(excel_path,WQ_name)
	df = water_df.copy()
	#save_excel(water_df,'')
	input_data = df[df['站点'].isin((['THL00']))][[2]].values
	date_frame = df[df['站点'].isin((['THL00']))][[0]].values
	input_data = Mean_Normalization(input_data)
	#input_data = Mean_Normalization(water_df[[2]].values[:170])
	#input_data = Min_Max_Normalization(water_df[[2]].values[:300])
	#date_frame = water_df[[0]].values[:300]
	
	draw_WQ(input_data,date_frame,title)
	"""
	

