from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeansModel
import pyspark.ml.clustering as clust
data = [(Vectors.dense([0.0, 0.0]), 2.0), (Vectors.dense([1.0, 1.0]), 2.0),
        (Vectors.dense([9.0, 8.0]), 2.0), (Vectors.dense([8.0, 9.0]), 2.0)]
df = spark.createDataFrame(data, ["features", "weighCol"])
kmeans = clust.KMeans(k=2)
kmeans.setSeed(1)
kmeans.setWeightCol("weighCol")
kmeans.setMaxIter(10)
kmeans.getMaxIter()
kmeans.clear(kmeans.maxIter)
model = kmeans.fit(df)
model.getDistanceMeasure()
model.setPredictionCol("newPrediction")
model.predict(df.head().features)
centers = model.clusterCenters()
len(centers)
transformed = model.transform(df).select("features", "newPrediction")
rows = transformed.collect()
rows[0].newPrediction == rows[1].newPrediction
rows[2].newPrediction == rows[3].newPrediction
model.hasSummary
summary = model.summary
summary.k
summary.clusterSizes
summary.trainingCost

temp_path = "E:\Documents\code\PySpark\clustering"
kmeans_path = temp_path + "/kmeans"
kmeans.save(kmeans_path)
kmeans2 = KMeans.load(kmeans_path)
kmeans2.getK()

model_path = temp_path + "/kmeans_model"
model.save(model_path)


model2 = KMeansModel.load(model_path)
model2.hasSummary
model.clusterCenters()[0] == model2.clusterCenters()[0]
model.clusterCenters()[1] == model2.clusterCenters()[1]
model.transform(df).take(1) == model2.transform(df).take(1)