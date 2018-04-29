#coding:utf-8
from django.shortcuts import render
from .models import Data, p_Data
import os
import os.path
import numpy as np
import pandas as pd
import time as te
import math
import decimal


# Create your views here.

from django.http import HttpResponse
from training_model import RC_excel
from training_model import RC_network
from training_model import RC_training
from training_model import RC_database


def index(request):
    return  HttpResponse("Training Model")

def training(request):
	return  render(request, 'simple_template.html')

def test(request):

    # 获取网络参数，输入参数，输出参数并从数据库读取数据并进行切割

    np1 = request.POST['np1']
    np2 = request.POST['np2']
    np3 = request.POST['np3']
    iop1 = request.POST['pH']
    iop2 = request.POST['DO']
    iop3 = request.POST['COD']
    iop4 = request.POST['TP']
    iop5 = request.POST['TN']
    iop6 = request.POST['NH4']


    numOfInput = 0
    numOfOutput = 0
    lenOfData = 0

    input_list1 = []
    output_list1 = []
    input_test_list1 = []
    output_test_list1 = []

    #根据用户选择,从水质总表中获取索引，从相应数据库中获取参数以及相应数据，动态生成参数选择栏
    #data1,data2
    #attribute1,attribute2,attribute3,attribute4
    #if dp is 1,2,3：
    #   data1.object, data2.object, data3.object

    if iop1 is not None:
        # 获取pH数据
        pH_list_Training1 = np.array(Data.objects.filter(station='THL00').values_list('pH', flat=True))
        pH_list_Training1_N = RC_database.Mean_Normalization(pH_list_Training1)
        pH_list_Training2 = Data.objects.filter(station='THL01').values_list('pH', flat=True)
        pH_list_Training3 = Data.objects.filter(station='THL03').values_list('pH', flat=True)
        pH_list_Training4 = Data.objects.filter(station='THL04').values_list('pH', flat=True)
        pH_list_Test1 = np.array(Data.objects.filter(station='THL05').values_list('pH', flat=True))
        pH_list_Test1 = RC_database.Mean_Normalization(pH_list_Test1)
        pH_list_Test2 = Data.objects.filter(station='THL06').values_list('pH', flat=True)
        pH_list_Test3 = Data.objects.filter(station='THL07').values_list('pH', flat=True)
        pH_list_Test4 = Data.objects.filter(station='THL08').values_list('pH', flat=True)
        lenOfData = len(pH_list_Training1_N)

        if iop1 == '1':
            numOfInput += 1
            input_list1.append(pH_list_Training1_N)
            input_test_list1.append(pH_list_Test1)
        elif iop1 == '-1':
            numOfOutput += 1
            #output_list1.append(pH_list_Training1)
            #output_test_list1.append(pH_list_Test1)
            output_data = pH_list_Training1_N
            output_test_data = pH_list_Test1
            training_array = np.array(pH_list_Training1)

    if iop2 is not None:
        # 获取DO数据
        DO_list_Training1 = np.array(Data.objects.filter(station='THL00').values_list('DO', flat=True))
        DO_list_Training1_N = RC_database.Mean_Normalization(DO_list_Training1)
        DO_list_Training2 = Data.objects.filter(station='THL01').values_list('DO', flat=True)
        DO_list_Training3 = Data.objects.filter(station='THL03').values_list('DO', flat=True)
        DO_list_Training4 = Data.objects.filter(station='THL04').values_list('DO', flat=True)
        DO_list_Test1 = np.array(Data.objects.filter(station='THL05').values_list('DO', flat=True))
        DO_list_Test1 = RC_database.Mean_Normalization(DO_list_Test1)
        DO_list_Test2 = Data.objects.filter(station='THL06').values_list('DO', flat=True)
        DO_list_Test3 = Data.objects.filter(station='THL07').values_list('DO', flat=True)
        DO_list_Test4 = Data.objects.filter(station='THL08').values_list('DO', flat=True)
        lenOfData = len(DO_list_Training1)


        if iop2 == '1':
            numOfInput += 1
            input_list1.append(DO_list_Training1_N)
            input_test_list1.append(DO_list_Test1)
        elif iop2 == '-1':
            numOfOutput += 1
            #output_list1.append(DO_list_Training1)
            #output_test_list1.append(DO_list_Test1)
            output_data = DO_list_Training1_N
            output_test_data = DO_list_Test1
            training_array = np.array(DO_list_Training1)

    if iop3 is not None:
        # 获取COD数据
        COD_list_Training1 = np.array(Data.objects.filter(station='THL00').values_list('COD', flat=True))
        COD_list_Training1_N = RC_database.Mean_Normalization(COD_list_Training1)
        COD_list_Training2 = Data.objects.filter(station='THL01').values_list('COD', flat=True)
        COD_list_Training3 = Data.objects.filter(station='THL03').values_list('COD', flat=True)
        COD_list_Training4 = Data.objects.filter(station='THL04').values_list('COD', flat=True)
        COD_list_Test1 = np.array(Data.objects.filter(station='THL05').values_list('COD', flat=True))
        COD_list_Test1 = RC_database.Mean_Normalization(COD_list_Test1)
        COD_list_Test2 = Data.objects.filter(station='THL06').values_list('COD', flat=True)
        COD_list_Test3 = Data.objects.filter(station='THL07').values_list('COD', flat=True)
        COD_list_Test4 = Data.objects.filter(station='THL08').values_list('COD', flat=True)
        lenOfData = len(COD_list_Training1)

        if iop3 == '1':
            numOfInput += 1
            input_list1.append(COD_list_Training1_N)
            input_test_list1.append(COD_list_Test1)
        elif iop3 == '-1':
            numOfOutput += 1
            #output_list1.append(COD_list_Training1)
            #output_test_list1.append(COD_list_Test1)
            output_data = COD_list_Training1_N
            output_test_data = COD_list_Test1
            training_array = np.array(COD_list_Training1)

    if iop4 is not None:
        # 获取TP数据
        TP_list_Training1 = np.array(Data.objects.filter(station='THL00').values_list('TP', flat=True))
        TP_list_Training1_N = RC_database.Mean_Normalization(TP_list_Training1)
        TP_list_Training2 = Data.objects.filter(station='THL01').values_list('TP', flat=True)
        TP_list_Training3 = Data.objects.filter(station='THL03').values_list('TP', flat=True)
        TP_list_Training4 = Data.objects.filter(station='THL04').values_list('TP', flat=True)
        TP_list_Test1 = np.array(Data.objects.filter(station='THL05').values_list('TP', flat=True))
        TP_list_Test1 = RC_database.Mean_Normalization(TP_list_Test1)
        TP_list_Test2 = Data.objects.filter(station='THL06').values_list('TP', flat=True)
        TP_list_Test3 = Data.objects.filter(station='THL07').values_list('TP', flat=True)
        TP_list_Test4 = Data.objects.filter(station='THL08').values_list('TP', flat=True)
        lenOfData = len(TP_list_Training1)

        if iop4 == '1':
            numOfInput += 1
            input_list1.append(TP_list_Training1_N)
            input_test_list1.append(TP_list_Test1)
        elif iop4 == '-1':
            numOfOutput += 1
            #output_list1.append(TP_list_Training1)
            #output_test_list1.append(TP_list_Test1)
            output_data = TP_list_Training1_N
            output_test_data = TP_list_Test1
            training_array = np.array(TP_list_Training1)

    if iop5 is not None:
        # 获取TN数据
        TN_list_Training1 = np.array(Data.objects.filter(station='THL00').values_list('TN', flat=True))
        TN_list_Training1_N = RC_database.Mean_Normalization(TN_list_Training1)
        TN_list_Training2 = Data.objects.filter(station='THL01').values_list('TN', flat=True)
        TN_list_Training3 = Data.objects.filter(station='THL03').values_list('TN', flat=True)
        TN_list_Training4 = Data.objects.filter(station='THL04').values_list('TN', flat=True)
        TN_list_Test1 = np.array(Data.objects.filter(station='THL05').values_list('TN', flat=True))
        TN_list_Test1 = RC_database.Mean_Normalization(TN_list_Test1)
        TN_list_Test2 = Data.objects.filter(station='THL06').values_list('TN', flat=True)
        TN_list_Test3 = Data.objects.filter(station='THL07').values_list('TN', flat=True)
        TN_list_Test4 = Data.objects.filter(station='THL08').values_list('TN', flat=True)
        lenOfData = len(TN_list_Training1)

        if iop5 == '1':
            numOfInput += 1
            input_list1.append(TN_list_Training1_N)
            input_test_list1.append(TN_list_Test1)
        elif iop5 == '-1':
            numOfOutput += 1
            #output_list1.append(TN_list_Training1_N)
            #output_test_list1.append(TN_list_Test1)
            output_data = TN_list_Training1
            output_test_data = TN_list_Test1
            training_array = np.array(TN_list_Training1)

    if iop6 is not None:
        # 获取NH4数据
        NH4_list_Training1 = np.array(Data.objects.filter(station='THL00').values_list('NH4', flat=True))
        NH4_list_Training1_N = RC_database.Mean_Normalization(NH4_list_Training1)
        NH4_list_Training2 = Data.objects.filter(station='THL01').values_list('NH4', flat=True)
        NH4_list_Training3 = Data.objects.filter(station='THL03').values_list('NH4', flat=True)
        NH4_list_Training4 = np.array(Data.objects.filter(station='THL04').values_list('NH4', flat=True))
        NH4_list_Training4_N = RC_database.Mean_Normalization(NH4_list_Training4)
        NH4_list_Test1 = np.array(Data.objects.filter(station='THL05').values_list('NH4', flat=True))
        NH4_list_Test1 = RC_database.Mean_Normalization(NH4_list_Test1)
        NH4_list_Test2 = Data.objects.filter(station='THL06').values_list('NH4', flat=True)
        NH4_list_Test3 = Data.objects.filter(station='THL07').values_list('NH4', flat=True)
        NH4_list_Test4 = Data.objects.filter(station='THL08').values_list('NH4', flat=True)
        lenOfData = len(NH4_list_Training1)

        if iop6 == '1':
            numOfInput += 1
            input_list1.append(NH4_list_Training1_N)
            input_test_list1.append(NH4_list_Test1)
        elif iop6 == '-1':
            numOfOutput += 1
            #output_list1.append(NH4_list_Training1)
            #output_test_list1.append(NH4_list_Test1)
            output_data = NH4_list_Training1_N
            output_test_data = NH4_list_Test1
            #training_array = np.array(NH4_list_Training1)
            training_array = np.array(NH4_list_Test1)

    # 归一化
    # RC_database.Mean_Normalization()

    # 生成训练及测试数据集

    input_data = np.zeros([numOfInput, lenOfData])
    #output_data = np.zeros([numOfOutput, lenOfData])
    input_test_data = np.zeros([numOfInput, lenOfData])
    #output_test_data = np.zeros([numOfOutput, lenOfData])

    input_data = np.mat(input_data)
    output_data = np.mat(output_data)
    input_test_data = np.mat(input_test_data)
    output_test_data = np.mat(output_test_data)

    for i in range(0, numOfInput):
        input_data[i, :] = np.mat(input_list1[i])

    #for i in range(0, numOfOutput):
        #output_data[i, :] = np.mat(output_list1[i])

    for i in range(0, numOfInput):
        input_test_data[i, :] = np.mat(input_test_list1[i])

    #for i in range(0, numOfOutput):
        #output_test_data[i, :] = np.mat(output_test_list1[i])

    num_input, num = np.shape(input_data)
    num_output = np.shape(output_data)[0]
    # df_input_data = pd.DataFrame(input_data).T
    # df_inputdata.columns = input_name


    # 生成神经网络

    dt = 0.1
    t = 10
    g = float(np1)

    num_of_train = int(np2)
    time_num = int(np3)

    network = RC_network.network(num_input=num_input, num_output=num_output, g=g, dt=dt)
    RC = RC_network.Reservoir_Computing(input_data=input_data, output_data=output_data, network=network, t=t)
    RC.updata_testdata(input_test_data, output_test_data)

    # 训练

    if num_input == 0 or num_output == 0:
        return HttpResponse('Please choose the output and input parameter!')
    if numOfOutput >= 2:
        return HttpResponse('Only one output parameter is permitted!')
    else:
        currentTime = te.strftime('%Y-%m-%d %H:%M:%S', te.localtime(te.time()))
        f = open('result.txt', 'w')
        f.write(currentTime + "\n")
        f.write('Training Start\n')
        f.close()
        RC.training(num_of_train=2, test_flag=True, time_num=1)

        currentTime = te.strftime('%Y-%m-%d %H:%M:%S', te.localtime(te.time()))
        f = open('result.txt', 'a+')
        f.write(currentTime + "\n")
        f.write('Training Over\n')
        f.close()

        #反归一化
        #predict_data = RC_database.R_Mean_Normalization(np.array(RC.zpt[0]), training_array)
        #print(np.array(Data.objects.filter(station='THL05').values_list('NH4', flat=True)))

        #currentTime = te.strftime('%Y-%m-%d %H:%M:%S', te.localtime(te.time()))
        #f = open('result.txt', 'a+')
        #f.write(currentTime + "\n")
        #f.write('Writing Predict Data\n')
        #f.close()

        #print(output_data)
        #for i in range(0, len(predict_data)):
            #predict_data[i] = float("%.3f" % predict_data[i])
            #if predict_data[i] < 0:
                #predict_data[i] = 0
            #p_Data.objects.filter(station='THL05', id=i+1).update(NH4=predict_data[i])
            #p_Data.objects.filter(station='THL05', id=i + 1).update(COD=predict_data[i])

        #print(len(RC.zpt[0]))
        #print(len(predict_data))
        #print(len(training_array))
        #print(predict_data)


    # return render(request, 'simple_template.html')
    # temp = type(pH_list_Training1)
    # print(temp)
    return HttpResponse('Training Over!')

    # 训练完成后生成的结果图片于img文件夹（命名为showx）以及training_result文件夹（以配置参数命名）各存一份
    # 在训练完毕后生成结果图片之前若有showx图片存在则删除图片，并通过js修改图片src来显示结果
    # curDir = os.getcwd()
    # os.rename(r"training_model/static/img/THL00-NH.png", r"training_model/static/img/show1.png")
    # os.remove(r"training_model/static/img/test.jpg")


def temp(request):
	return  render(request, 'temp.html')

def doTemp(request):
    if request.method == 'POST':
        f = open('result.txt', 'r')
        result = f.read()
        #print("it's a test") # 用于测试
        #print(request.POST['np1'])  # 测试是否能够接收到前端发来的name字段

        return HttpResponse(result)  # 最后返会给前端的数据，如果能在前端弹出框中显示我们就成功了
    else:
        return HttpResponse("<h1>test</h1>")