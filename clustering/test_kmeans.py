from __future__ import print_function

from numpy import array
from math import  sqrt

from pyspark import SparkContext

from pyspark.mllib.clustering import KMeans, KMeansModel

if __name__ == "__main__":
    sc = SparkContext(appName="KmeansExample")

    # Load and parse the data
    data = sc.textFile("kmeans_data.txt")
    parsedData = data.map(lambda line:array([float(x) for x in line.split(' ')]))

    # Build the Model(cluster the data)
    clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")
    print(clusters.clusterCenters)

    print(clusters.predict([0.2, 0.2, 0.2]))

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))


# 原文链接：https://blog.csdn.net/goodstuddayupyyeah/article/details/75020659