from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, split
from pyspark.sql.functions import size
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes
import pyspark.sql.functions as f
from pyspark.sql import SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import array_contains


# sc is an SparkContext.
appName = "testSpark"
master = "local"
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

sqlContext = SQLContext(sc)
df = sqlContext.read.load('/smalldata/*', 
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')
data = df.selectExpr("target as label","topics as features")
data_fl = data.withColumn(
    "features",
    split(col("features"), ",").cast("array<float>").alias("features")
)

data_nan = data_fl.withColumn("first_two", f.array([f.col("features")[i] for i in range(1,19)]))
data_final = data_nan.selectExpr("label","first_two as features")
df_set = data_final.filter(size('features') > 17).filter(size('features') < 19)
df_newset = df_set.withColumn("likes_red",array_contains(col("features"), "0")).filter(col("likes_red") == False)
df_nset = df_newset.select("label","features")
to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
data_vec = df_nset.select("label", to_vector("features").alias("features"))
splits = data_vec.randomSplit([0.75,0.25],seed=24)
layers = [18, 30, 24, 2]

dt =  MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=128, seed=1234)
model=dt.fit(splits[0])

result = model.transform(splits[1])
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
