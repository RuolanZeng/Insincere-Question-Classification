#!/usr/bin/env python
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
from pyspark.sql.functions import col, udf, ltrim , rtrim
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import IntegerType, StringType, ArrayType
import datetime
#from pyspark import SparkConf
#from pyspark import SparkContext
#from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import split
from pyspark.sql.functions import size
from pyspark.ml.linalg import VectorUDT
#from pyspark.sql.functions import udf
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes , RandomForestClassifier
import pyspark.sql.functions as f
from pyspark.sql import SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import array_contains
from pyspark.ml.feature import VectorAssembler


# In[2]:


spark = SparkSession        .builder        .appName("QuoraInsincere")        .getOrCreate()

sc = spark.sparkContext


# In[3]:


corpus = spark.read.option("header","true").option("inferSchema","true").csv( "/train/train.csv")


# In[4]:


data = corpus.toDF("qid","question_text","target")
dataSincere = data.where(data.target==1).limit(10000)
dataInsincere = data.where(data.target==0).limit(10000)
data = dataSincere.union(dataInsincere).sort(col("qid"))
#data = data.limit(2000)


# In[5]:


def clean_text(x):  
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, r' {punct} ')
    for punct in '?!.,"#$%\'()*+:;<=>@[\\]^_`{|}~':
        x = x.replace(punct, '')
    return x
udfClean = udf(lambda x: clean_text(x),StringType())


# In[6]:


def nonasciitoascii(unicodestring):
    return unicodestring.encode("ascii","ignore")
convertedudf = udf(nonasciitoascii)


# In[7]:


import re
def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
udfCleanNum = udf(lambda x: clean_numbers(x),StringType())


# In[8]:


udfToInt = udf(lambda z : int(z), IntegerType())


# In[9]:


data = data.where(data['question_text'] != "")
data = data.withColumn('question_text',convertedudf(data.question_text))
data = data.withColumn('question_text', ltrim(data.question_text))
data = data.withColumn('question_text', rtrim(data.question_text))
data = data.withColumn('question_text', udfClean(data.question_text))
data = data.withColumn('question_text', udfCleanNum(data.question_text))
#data.show(truncate= False)


# In[10]:


tokenizer = Tokenizer(inputCol="question_text", outputCol="words")
tokenized = tokenizer.transform(data)
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
cleanData = remover.transform(tokenized)
cleanData = cleanData.select("filtered","target")


# In[11]:


word2Vec = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered", outputCol="wordvectors")
model = word2Vec.fit(cleanData)
result = model.transform(cleanData)


# In[12]:


from pyspark.ml.feature import IDF, CountVectorizer

cv = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")

cvmodel = cv.fit(result)

featurizedData = cvmodel.transform(result)
#featurizedData.show(truncate=False)


# In[13]:


idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData = rescaledData.select("filtered","wordvectors","features","target")
#rescaledData.select("filtered", "features").show()


# In[22]:


from pyspark.ml.clustering import LDA
def train_LDA(num_topics, dataset):
    max_iterations = 50
    lda = LDA(k=num_topics, maxIter=max_iterations)
    model = lda.fit(dataset.select("features", "wordvectors"))
    return model


# In[25]:


sincere = rescaledData.where(rescaledData["target"]==0)
print("Sincere : ")
for x in range(7,20):
    print("Number of topics : ", x, " start time : ", datetime.datetime.now())
    modelSincere = train_LDA(x,sincere)
    topicSincere = modelSincere.describeTopics(5)
    ll = modelSincere.logLikelihood(sincere)
    print(" loglikelihood : ", str(ll))
    lp = modelSincere.logPerplexity(sincere)
    print("logPerplexity : ", str(lp))
# print("The topics described by their top-weighted terms :")
# topicSincere.show()


# In[17]:


insincere = rescaledData.where(rescaledData["target"]==1)
print("Insincere : ")
for x in range(7,20):
    print("Number of topics : ", x, " start time : ", datetime.datetime.now())
    modelInsincere = train_LDA(x,insincere)
    topicInsincere = modelInsincere.describeTopics(5)
    ll = modelInsincere.logLikelihood(insincere)
    print(" loglikelihood : ", str(ll))
    lp = modelInsincere.logPerplexity(insincere)
    print("logPerplexity : ", str(lp))
# print("The topics described by their top-weighted terms for sincere questions:")
# topicInsincere.show()


# In[18]:





# In[19]:


from pyspark.sql.types import ArrayType, StringType

def indices_to_terms(vocabulary):
    def indices_to_terms(xs):
        return [vocabulary[int(x)] for x in xs]
    return udf(indices_to_terms, ArrayType(StringType()))


# In[20]:


print("topics of sincere questions :")
topicSincere.withColumn("topics_words", indices_to_terms(cvmodel.vocabulary)("termIndices")).show(truncate=False)


# In[21]:


print("topics of insincere questions :")
topicInsincere.withColumn("topics_words", indices_to_terms(cvmodel.vocabulary)("termIndices")).show(truncate=False)


# In[ ]:




