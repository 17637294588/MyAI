from django.shortcuts import render,HttpResponseRedirect,redirect
from django.http import HttpResponse
import re,json
from django.views import View
from My_Ai import settings
import pandas as pd
import numpy as np
import csv
import time
from sklearn.externals import joblib
# Create your views here.


# 上传文件
class Sub(View):
    def post(self,request):
        mes = {}
        file = request.FILES.get('file',None)
        if file is None:
            mes['code'] = 10010
            return HttpResponse('没传文件')

            
        with open(settings.UPLOAD_ROOT+"/"+file.name,'wb') as f:
            for i in file.readlines():
                f.write(i)

        #读csv
        try: 
            file_list = pd.read_csv(settings.UPLOAD_ROOT+"/"+file.name).head()
        except Exception as e:
            
            print(str(e))

        file_np = np.array(file_list)
        list_csv = file_np.tolist()
        mes['code'] = 200
        mes['mes'] = list_csv


        return HttpResponse(json.dumps(mes),content_type='application/json')



# scv 训练
class Train_scv(View):
    def post(self,request):
        mes = {}
        kernel = request.POST.get('kernel')
        c = request.POST.get('c')
        file_name = request.POST.get('file_name')

        if not all([kernel,c]):
            mes['code'] = 10010
            mes['mes'] = '输入不能为空'
        elif not file_name:
            mes['code'] = 10010
            mes['mes'] = '先上传文件'

        else:

            from sklearn.model_selection import train_test_split

            df = pd.read_csv(settings.UPLOAD_ROOT+"/"+file_name)
            dataset_X,dataset_Y = df.iloc[:,:-1].values,df.iloc[:,-1].values
            trainX,testX,trainY,testY = train_test_split(dataset_X,dataset_Y,test_size=0.2,random_state=37)
            
            # 构建svm 模型
            from sklearn.svm import SVC
            classifier_rbf = SVC(kernel=kernel,C=float(c))  # 构建线性分类器,找到的超平面是一个线性超平面
            classifier_rbf.fit(trainX,trainY)

            # 保存模型
            model_name = 'scv.txt'
            path = settings.UPLOAD_ROOT+'./'+model_name
            joblib.dump(classifier_rbf,path)

            # 打印模型报告
            # from sklearn.metrics import classification_report
            # y_pred=classifier_rbf.predict(testX)
            # print('=============================')
            # report = classification_report(testY, y_pred)

            # print(report)

            # print(type(report))
            # print('================================')

            report = float(classifier_rbf.score(X=testX,y=testY))
            mes['code'] = 200
            mes['mes'] = report
            mes['model_name'] = model_name

        return HttpResponse(json.dumps((mes)))



# 决策树训练
class Train_tree(View):
    def post(self,request):
        mes = {}
        max_depth = request.POST.get('max_depth')
        file_name = request.POST.get('file_name')
        if not all([max_depth]):
            mes['code'] = 10010
            mes['mes'] = '输入不能为空'
        elif not file_name:
            mes['code'] = 10010
            mes['mes'] = '先上传文件'
        else:
            from sklearn.model_selection import train_test_split

            df = pd.read_csv(settings.UPLOAD_ROOT+"/"+file_name)
            dataset_X,dataset_Y = df.iloc[:,:-1].values,df.iloc[:,-1].values
            trainX,testX,trainY,testY = train_test_split(dataset_X,dataset_Y,test_size=0.2,random_state=37)
            
            # 构建决策树分类器
            from sklearn.tree import DecisionTreeClassifier
            dtreeReg = DecisionTreeClassifier(max_depth=int(max_depth))   # max_depth：决策树的层数
            dtreeReg.fit(trainX,trainY)

            # 保存模型
            model_name = 'tree.txt'
            path = settings.UPLOAD_ROOT+'./'+model_name
            joblib.dump(dtreeReg,path)


            # 模型打分
            report = float(dtreeReg.score(X=testX,y=testY))
            mes['code'] = 200
            mes['mes'] = report
            mes['model_name'] = model_name

        return HttpResponse(json.dumps((mes)))



# 模型预测
class Start(View):
    def post(self,request):
        mes = {}
        svc_model_name = request.POST.get('svc_model_name')
        tree_model_name = request.POST.get('tree_model_name')  
        redio = request.POST.get('redio')
        pre = request.POST.get('pre')

        
       
        if not redio:
            mes['code'] = 10010
            mes['mes'] = '请选择模型'

        elif not all([svc_model_name,tree_model_name,pre]):
            mes['code'] = 10010
            mes['mes'] = '请先训练模型'

        else:
            # 处理特征数据
            pre_list = pre.split(',')
            list_all = []
            for i in pre_list:
                list_all.append(float(i))
            pre_np = np.array([list_all])

            if int(redio) == 0:
                path = settings.UPLOAD_ROOT+'./'+svc_model_name
                svc_model = joblib.load(path)
                y_pred=svc_model.predict(pre_np).tolist()
                
            else:
                path = settings.UPLOAD_ROOT+'./'+tree_model_name
                tree_model = joblib.load(path)
                y_pred = tree_model.predict(pre_np).tolist()

            mes['code'] = 200
            mes['mes'] = y_pred
            mes['pre'] = pre
        return HttpResponse(json.dumps((mes)))
        