from pyspark.sql import SparkSession
from pyspark.sql import functions as f
# from pyspark.sql.types import StructType,StructField, StringType, IntegerType , BooleanType
# from pyspark.ml.clustering import KMeans
# import pyspark.ml.clustering as clust

'''读取数据'''
spark = SparkSession.builder.appName('pyspark - read csv').getOrCreate()
sc = spark.sparkContext 
df = spark.read.csv("E:/Documents/code/PySpark/data/wine.csv")
df.printSchema()
