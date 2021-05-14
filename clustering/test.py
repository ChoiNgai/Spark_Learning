from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

from PyML.FuzzyClustering.Algorithm import fcm
import numpy as np

spark = SparkSession.builder.appName('pyspark - read csv').getOrCreate()
sc = spark.sparkContext 
data = spark.read.csv("E:/Documents/code/PySpark/data/wine.csv")

'''读取数据'''
spark = SparkSession.builder.appName('pyspark - read csv').getOrCreate()
sc = spark.sparkContext 
data = spark.read.csv("E:/Documents/code/PySpark/data/wine.csv")            #数据
label = spark.read.csv("E:/Documents/code/PySpark/data/winelabel.csv")      #标签
data = data.rdd         # dataframe 转换为RDD
cluster_n = int(label.select("*").rdd.max()[0])  #类簇数(此行参数建议不改变)
label = label.rdd

'''参数设置'''

'''预处理'''
# data = ( data - np.min(data,axis=0)) / (np.max(data,axis=0) - np.min(data,axis=0))  #数据标准化


'''估计器'''
# U,center,fcm_obj_fcn = smuc(data,cluster_n,label,max_iter = 100,e = 0.00001,lamda=0.5,printOn = 1)
U,center,fcm_obj_fcn =  fcm(data,cluster_n,m = 2,max_iter = 100,e = 0.00001,printOn = 1)

'''评估器'''
label_pred,abaaba = np.where(U==np.max(U,axis=0)) #最大值索引