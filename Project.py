# coding: utf-8

# In[1]:


from __future__ import print_function
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from operator import add
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf, ltrim, rtrim
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import IntegerType, StringType, ArrayType
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import split
from pyspark.sql.functions import size
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
import pyspark.sql.functions as f
from pyspark.sql import SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import array_contains
from pyspark.ml.feature import VectorAssembler

# In[2]:


spark = SparkSession.builder.appName("QuoraInsincere").getOrCreate()
sc = spark.sparkContext

# In[3]:

#reading the data
corpus = spark.read.option("header", "true").option("inferSchema", "true").csv("/train/train.csv")

# In[4]:

# converting the data read into a dataframe
data = corpus.toDF("qid", "question_text", "target")


# In[5]:

# utility function for cleaning the text
def clean_text(x):
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, r' {punct} ')
    for punct in '?!.,"#$%\'()*+:;<=>@[\\]^_`{|}~':
        x = x.replace(punct, '')
    return x


udfClean = udf(lambda x: clean_text(x), StringType())


# In[7]:

# utility function for removing non-ascii characters
def nonasciitoascii(unicodestring):
    return unicodestring.encode("ascii", "ignore")


convertedudf = udf(nonasciitoascii)

# In[8]:


import re

# utility function for cleaning the numerical parts of the text
def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


udfCleanNum = udf(lambda x: clean_numbers(x), StringType())

# In[9]:

# data cleaning
data = data.where(data['question_text'] != "")
data = data.withColumn('question_text', ltrim(data.question_text))
data = data.withColumn('question_text', rtrim(data.question_text))

# In[10]:


data = data.withColumn('question_text', udfClean(data.question_text))
data = data.withColumn('question_text', udfCleanNum(data.question_text))

# In[11]:

#data transformation and removal of stop words
tokenizer = Tokenizer(inputCol="question_text", outputCol="words")
tokenized = tokenizer.transform(data)
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
cleanData = remover.transform(tokenized)
cleanData = cleanData.select("filtered", "target")  # .limit(300000)

# In[12]:

# mapping the words of the text into n-dimensional vectors
word2Vec = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered", outputCol="wordvectors")
model = word2Vec.fit(cleanData)
result = model.transform(cleanData)

# In[13]:


from pyspark.ml.feature import IDF, CountVectorizer

cv = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")
cvmodel = cv.fit(result)
featurizedData = cvmodel.transform(result)

# In[14]:

# calculating tf-idf vectors
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData = rescaledData.select("filtered", "wordvectors", "features", "target")

# In[15]:


from pyspark.ml.clustering import LDA

#topic moddelling using LDA
def train_LDA(dataset):
    num_topics = 8
    max_iterations = 100
    lda = LDA(k=num_topics, maxIter=max_iterations)
    model = lda.fit(dataset.select("filtered", "features", "wordvectors"))
    return model


# In[16]:


modelTopic = train_LDA(rescaledData)

# In[22]:


transformed = modelTopic.transform(rescaledData)

# In[23]:


import numpy as np

# utility function for converting vectors into list
def converttolist(topicDistribution):
    return topicDistribution.tolist()


udfconverttolist = udf(lambda z: converttolist(z), ArrayType(FloatType()))

# In[33]:


trans = transformed.withColumn("topics", udfconverttolist(transformed.topicDistribution))

# In[34]:


traindataset = trans.select("wordvectors", "topics", "target")

# In[35]:


udfToInt = udf(lambda z: int(z), IntegerType())
trainset = traindataset.withColumn("label", udfToInt(traindataset.target))


# In[37]:

#classifier and prediction
data = trainset.select("wordvectors", "topics", "label")
df_set = data.filter(size(data.topics) > 19).filter(size(data.topics) < 21)
df_newset = df_set.withColumn("likes_red", array_contains(col("topics"), "9")).filter(col("likes_red") == False)
to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
data_vec = df_newset.select("label", "wordvectors", to_vector("topics").alias("topics"))

assembler = VectorAssembler(
    inputCols=["wordvectors", "topics"],
    outputCol="features")

output = assembler.transform(data_vec)
splits = output.randomSplit([0.6, 0.4], seed=24)
#layers = [20, 30, 24, 2]

# dt =  MultilayerPerceptronClassifier(maxIter=10, layers=layers, blockSize=128, seed=1234)
dt = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
model = dt.fit(splits[0])

result = model.transform(splits[1])
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
