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
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StopWordsRemover
from gensim.models import KeyedVectors
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import IntegerType, StringType


# In[2]:


spark = SparkSession        .builder        .appName("QuoraInsincere")        .getOrCreate()

sc = spark.sparkContext


# In[3]:


corpus = spark.read.option("header","true").option("inferSchema","true").csv( "/home/akash/project/train.csv")


# In[4]:


data = corpus.toDF("qid","question_text","target")
#data = data.limit(200)


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
#clean_text("Is it just me or have you ever been in this phase wherein you became ignorant to the people you once loved, completely disregarding their feelings/lives so you get to have something go your way and feel temporarily at ease. How did things change?")


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


data = data.where(data['question_text'] != "")
data = data.withColumn('question_text',convertedudf(data.question_text))
data = data.withColumn('question_text', ltrim(data.question_text))
data = data.withColumn('question_text', rtrim(data.question_text))

#data=data.withColumn('question_text',commaRep(data.question_text))
#data=data.withColumn('question_text', regexp_replace('question_text', ',', ''))
#data.show(truncate = False)


# In[9]:


data = data.withColumn('question_text', udfClean(data.question_text))
#data = data.withColumn('question_text', udfCleanNum(data.question_text))
#data.show(truncate= False)


# In[10]:


data.show(truncate = False)


# In[11]:


tokenizer = Tokenizer(inputCol="question_text", outputCol="words")
tokenized = tokenizer.transform(data)
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
cleanData = remover.transform(tokenized)
cleanData = cleanData.select("filtered","target")#.limit(300000)


# In[12]:


word2Vec = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered", outputCol="wordvectors")
model = word2Vec.fit(cleanData)

result = model.transform(cleanData)


# In[13]:


from pyspark.ml.feature import IDF, CountVectorizer

cv = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")

cvmodel = cv.fit(result)

featurizedData = cvmodel.transform(result)
#featurizedData.show(truncate=False)


# In[14]:


idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData = rescaledData.select("filtered","wordvectors","features","target")
#rescaledData.select("filtered", "features").show()


# In[15]:


from pyspark.ml.clustering import LDA
def train_LDA(dataset):
    num_topics = 20
    max_iterations = 100
    lda = LDA(k=num_topics, maxIter=max_iterations)
    model = lda.fit(dataset.select("filtered", "features", "wordvectors"))
    return model


# In[16]:


sincere = rescaledData.where(rescaledData["target"]==0)
modelSincere = train_LDA(sincere)
topicSincere = modelSincere.describeTopics(1)
#print("The topics described by their top-weighted terms :")
#topicSincere.show()


# In[17]:


insincere = rescaledData.where(rescaledData["target"]==1)
modelInsincere = train_LDA(insincere)
topicInsincere = modelInsincere.describeTopics(5)
#print("The topics described by their top-weighted terms for sincere questions:")
#topicInsincere.show()


# In[18]:


from pyspark.sql.types import ArrayType, StringType

def indices_to_terms(vocabulary):
    def indices_to_terms(xs):
        return [vocabulary[int(x)] for x in xs]
    return udf(indices_to_terms, ArrayType(StringType()))


# In[ ]:


#topicSincere.withColumn("topics_words", indices_to_terms(cvmodel.vocabulary)("termIndices")).show(truncate=False)


# In[ ]:


#topicInsincere.withColumn("topics_words", indices_to_terms(cvmodel.vocabulary)("termIndices")).show(truncate=False)


# In[19]:


transformedSincere = modelSincere.transform(rescaledData.where(rescaledData["target"]==0))


# In[20]:


transformedInsincere =  modelInsincere.transform(rescaledData.where(rescaledData["target"]==1))


# In[ ]:


#transformedSincere.where(transformedSincere['target']==1).show()


# In[21]:


from pyspark.sql.types import IntegerType, FloatType
import numpy as np
def findTopicSincere(topicDistribution):
    zero = np.zeros(20)
    return np.concatenate((topicDistribution, zero)).tolist()
def findTopicInsincere(topicDistribution):
    zero = np.zeros(20)
    return np.concatenate((zero, topicDistribution)).tolist()
udfFindTopicSincere = udf(lambda z  :findTopicSincere(z), ArrayType(FloatType()))
udfFindTopicInsincere = udf(lambda z  :findTopicInsincere(z), ArrayType(FloatType()))


# In[ ]:


#smallDataset = transformedSincere.select(transformedSincere.topicDistribution, transformedSincere.target).limit(50)
#smallDataset = smallDataset.withColumn("topic", udfFindTopicSincere(smallDataset.topicDistribution))
#smallDataset.show(truncate = False)


# In[ ]:


# smallDataset = smallDataset.withColumn("topic", udfFindTopicSincere(smallDataset.topicDistribution))
# smallDataset.show()


# In[22]:


#smallDataset.show()
topicSincere = transformedSincere.withColumn("topic", udfFindTopicSincere(transformedSincere.topicDistribution))


# In[23]:


topicInsincere = transformedInsincere.withColumn("topic", udfFindTopicInsincere(transformedInsincere.topicDistribution))


# In[24]:


topicSincere = topicSincere.select("wordvectors", "topic","target")
topicInsincere = topicInsincere.select("wordvectors", "topic","target")


# In[ ]:


#topicInsincere.count()


# In[25]:


traindataset = topicSincere.union(topicInsincere)


# In[26]:


traindataset.show()


# In[27]:


#converted = traindataset.withColumn('question_text',convertedudf(traindataset.question_text))
#converted.show()


# In[28]:


traindataset.toPandas().to_csv('traindataset3.csv')




