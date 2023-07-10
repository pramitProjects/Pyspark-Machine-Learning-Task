from pyspark.sql import SparkSession
import matplotlib
matplotlib.use('Agg')
import pyspark.sql.functions as F
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark.sql.functions import col,sum
from pyspark.sql.functions import split,concat,concat_ws,countDistinct,to_date, month,year,max
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import dayofmonth, hour, to_timestamp, regexp_extract, from_utc_timestamp, substring, regexp_replace
from pyspark.sql.functions import first
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 2 Exercise") \
        .config("spark.local.dir","/fastdata/acq21ps") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

rawdata = spark.read.csv("../Data/freMTPL2freq.csv", header=True).cache()
# rawdata.show()
# rawdata.printSchema()

# Converting the columns to DoubleType

rawdata = rawdata.withColumn("IDpol", col("IDpol").cast("double"))
rawdata = rawdata.withColumn("ClaimNb", col("ClaimNb").cast("double"))
rawdata = rawdata.withColumn("Exposure", col("Exposure").cast("double"))
rawdata = rawdata.withColumn("VehPower", col("VehPower").cast("double"))
rawdata = rawdata.withColumn("VehAge", col("VehAge").cast("double"))
rawdata = rawdata.withColumn("DrivAge", col("DrivAge").cast("double"))
rawdata = rawdata.withColumn("BonusMalus", col("BonusMalus").cast("double"))
rawdata = rawdata.withColumn("Density", col("Density").cast("double"))

# rawdata.show()
# rawdata.printSchema()
myseed = 5

print("------------------ TASK A ------------------")
print("\n")
from pyspark.sql.functions import log, when
# Preprocessing the ClaimNb column by replacing the 0 values with small 0.001 as explained in the report
rawdata = rawdata.withColumn('ClaimNb',when(rawdata.ClaimNb==0.0,0.001).otherwise(rawdata.ClaimNb))
# Now forming the LogClaimNb column which will not have any null values
rawdata = rawdata.withColumn("LogClaimNb", log(rawdata.ClaimNb))

# Create NZClaim column as a binary indicator for non-zero ClaimNb values
rawdata = rawdata.withColumn("NZClaim", when(rawdata.ClaimNb > 0.001, 1).otherwise(0))
rawdata.show(20,False)

#------------------------------------ ONE-HOT ENCODING THE CATEGORICAL FEATURES -----------------------------

# Selecting the features and target variables

rawdata = rawdata.select('Exposure','Area','VehPower','VehAge','DrivAge','BonusMalus','VehBrand','VehGas','Density','Region','ClaimNb','LogClaimNb','NZClaim')
categorical_features = ['Area','VehBrand','VehGas','Region']
indexed_features = ['Area_Indexed','VehBrand_Indexed','VehGas_Indexed','Region_Indexed']
AreaIndexer = StringIndexer(inputCol='Area',outputCol='Area_Indexed')
AreaIndexed = AreaIndexer.fit(rawdata).transform(rawdata)


VehBrandIndexer = StringIndexer(inputCol='VehBrand',outputCol='VehBrand_Indexed')
VehBrandIndexed = VehBrandIndexer.fit(AreaIndexed).transform(AreaIndexed)



VehGasIndexer = StringIndexer(inputCol='VehGas',outputCol='VehGas_Indexed')
VehGasIndexed = VehGasIndexer.fit(VehBrandIndexed).transform(VehBrandIndexed)



RegionIndexer = StringIndexer(inputCol='Region',outputCol='Region_Indexed')
RegionIndexed = RegionIndexer.fit(VehGasIndexed).transform(VehGasIndexed)

# Dropping the String categorical features
indexedData = RegionIndexed.drop(*categorical_features)
ohe = OneHotEncoder(inputCols=indexed_features, outputCols=["Area_ohe","VehBrand_ohe","VehGas_ohe","Region_ohe"])
#one hot encoding the indexed categorical features
rawdata_ohe = ohe.fit(indexedData).transform(indexedData).drop(*indexed_features)

#------------------------------------------ SPLITTING THE DATASET INTO TRAIN AND TEST ------------------------------------------
print("The number of zero and non-zero claims in the main dataset are: \n")
rawdata_ohe.groupBy('NZClaim').count().show()

# Converting the whole dataframe into Pandas
pandas_df = rawdata_ohe.select("*").toPandas()
X = pandas_df[["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density","Area_ohe","VehBrand_ohe","VehGas_ohe","Region_ohe","ClaimNb","LogClaimNb"]]
Y = pandas_df["NZClaim"]

#Applying stratiflied split on the NZClaim column to form train and test so that equal proportion of zero and non-zero values (as in the main dataframe) are maintained in them
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=myseed)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
# Converting the train and test sets back to PySpark DataFrames
train_data = spark.createDataFrame(train_data)
test_data = spark.createDataFrame(test_data)
# checking the distribution of ClaimNb in the train and test sets
print("The number of zero and non-zero claims in the train dataset are: \n")
train_data.groupBy('NZClaim').count().show()
print("The number of zero and non-zero claims in the test dataset are: \n")
test_data.groupBy('NZClaim').count().show()

#---------------------------------------------- STANDARDISING NUMERICAL FEATURES --------------------------------------

numeric_features = ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
assembler = VectorAssembler(inputCols=numeric_features,outputCol='Vectored_numeric_features')
output1 = assembler.transform(train_data)
scaler = StandardScaler(inputCol='Vectored_numeric_features',outputCol='Scaled_features', withMean = True, withStd = True)
# Standardising the numerical features in the train set by fitting the scaler on train and transforming on train
scaled_train_data = scaler.fit(output1).transform(output1)

# Selecting the required columns
scaled_train_data = scaled_train_data.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','ClaimNb','LogClaimNb','NZClaim')

output = assembler.transform(test_data)

#Standardising the test set's numerical features by fitting the scaler on train and transforming on test 
scaled_test_data = scaler.fit(output1).transform(output)
scaled_test_data = scaled_test_data.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','ClaimNb','LogClaimNb','NZClaim')

# Forming different train and test sets for the different ML tasks. They only differ in the target column
traindata1 = scaled_train_data.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','ClaimNb')
traindata2 = scaled_train_data.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','LogClaimNb')
traindata3 = scaled_train_data.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','NZClaim')

testdata1 = scaled_test_data.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','ClaimNb')
testdata2 = scaled_test_data.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','LogClaimNb')
testdata3 = scaled_test_data.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','NZClaim')

#----------------------------------------- GETTING THE VALIDATION SET OUT OF THE TRAINING SET -----------------------------------------------------

pandas_df = train_data.select("*").toPandas()
X = pandas_df[["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density","Area_ohe","VehBrand_ohe","VehGas_ohe","Region_ohe","ClaimNb","LogClaimNb"]]
Y = pandas_df["NZClaim"]
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=myseed)
trainingData = pd.concat([X_train, y_train], axis=1)
validationData = pd.concat([X_val, y_val], axis=1)
trainingData = spark.createDataFrame(trainingData)
validationData = spark.createDataFrame(validationData)

#---------------------------------------- STANDARDIDING THE NUMERICAL FEATURES OF THE NEW TRAINING AND VALIDATION SET ------------------------------------
output1 = assembler.transform(trainingData)
scaler = StandardScaler(inputCol='Vectored_numeric_features',outputCol='Scaled_features', withMean = True, withStd = True)
trainingData = scaler.fit(output1).transform(output1)

trainingData = trainingData.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','ClaimNb','LogClaimNb','NZClaim')

output = assembler.transform(validationData)
validationData = scaler.fit(output1).transform(output)
validationData = validationData.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','ClaimNb','LogClaimNb','NZClaim')

#--------------------------------------- MACHINE LEARNING TASKS ------------------------------------------

print (" --------------- Question B Subpart b, c i Poisson Regression ------------------" )
print("\n")

#Getting the respective new training and validation for this task
trainingData1 = trainingData.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','ClaimNb')
validationData1 = validationData.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','ClaimNb')


from pyspark.ml.regression import GeneralizedLinearRegression


featureAssembler = VectorAssembler(inputCols=['Scaled_features', 'Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe'], outputCol='features')

#Defining the GLM model
glm = GeneralizedLinearRegression(family="poisson", labelCol="ClaimNb", featuresCol="features")

# forming the pipeline with the feature vector assembler and the GLM model
pipeline = Pipeline(stages=[featureAssembler, glm])

# regParam values to iterate over
regParams = [0.001, 0.01, 0.1, 1, 10]

# Fit the pipeline with each regParam and evaluate on the validation set
rmse_values = []
rmse_values1 = []
start  = time.time()
for regParam in regParams:
    glm.setRegParam(regParam)
    model = pipeline.fit(trainingData1)
    predictions = model.transform(validationData1)
    evaluator = RegressionEvaluator(labelCol="ClaimNb", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    rmse_values.append(rmse)
    print("RegParam={}, Validation RMSE={}".format(regParam, rmse))
    predictions = model.transform(trainingData1)
    rmse = evaluator.evaluate(predictions)
    rmse_values1.append(rmse)
    print("RegParam={}, Training RMSE={}".format(regParam, rmse))
end = time.time()
print("\n Time Taken for Tuning Poisson Regression is {} units".format(end-start) )
#Plotting the validation curve
plt.clf()
plt.plot(regParams, rmse_values, 'ro-', label = 'Validation score', color = 'r')
plt.plot(regParams, rmse_values1, 'ro-', label = 'Training score', color = 'b')
plt.legend()
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('RMSE')
plt.title('Validation Curve')
plt.savefig('../Output/PoissonCurve.png')

# Obtaining the best regParam
best_regParam = regParams[np.argmin(rmse_values)]
glm.setRegParam(best_regParam)
# Training the whole trainset with the best regParam
best_model = pipeline.fit(traindata1)

# Evaluating the best model on the test set
test_predictions = best_model.transform(testdata1)
evaluator = RegressionEvaluator(labelCol="ClaimNb", predictionCol="prediction", metricName="rmse")
test_rmse = evaluator.evaluate(test_predictions)
print("Best RegParam={}, Test RMSE={}".format(best_regParam, test_rmse))

#Printing the best model coefficients
print("Model Coefficients: ", best_model.stages[-1].coefficients.values)

print(" ------------------ Question B subpart b, c, ii L1-Regularized Linear Regression ------------------------------")
print("\n")

trainingData2 = trainingData.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','LogClaimNb')
validationData2 = validationData.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','LogClaimNb')

from pyspark.ml.regression import LinearRegression

featureAssembler = VectorAssembler(inputCols=['Scaled_features', 'Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe'], outputCol='features')


lr = LinearRegression(labelCol="LogClaimNb", featuresCol="features", regParam=0.1, elasticNetParam=1.0, solver="normal")
pipeline = Pipeline(stages=[featureAssembler, lr])

regParams = [0.001, 0.01, 0.1, 1, 10]
rmse_values = []
rmse_values1 = []
start = time.time()
for regParam in regParams:
    lr.setRegParam(regParam)
    model = pipeline.fit(trainingData2)
    predictions = model.transform(validationData2)
    evaluator = RegressionEvaluator(labelCol="LogClaimNb", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    rmse_values.append(rmse)
    print("RegParam={}, Validation RMSE={}".format(regParam, rmse))
    predictions = model.transform(trainingData2)
    rmse = evaluator.evaluate(predictions)
    rmse_values1.append(rmse)
    print("RegParam={}, Training RMSE={}".format(regParam, rmse))
end = time.time()
print("\n Time Taken for Tuning L1-Linear Regression is {} units".format(end-start) )

#Plotting the validation curve
plt.clf()
plt.plot(regParams, rmse_values, 'ro-', label = 'Validation score', color = 'r')
plt.plot(regParams, rmse_values1, 'ro-', label = 'Training score', color = 'b')
plt.legend()
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('RMSE')
plt.title('Validation Curve')
plt.savefig('../Output/L1_Linear_Curve.png')



best_regParam = regParams[np.argmin(rmse_values)]
lr.setRegParam(best_regParam)
best_model = pipeline.fit(traindata2)

#Evaluating the best model on the test set
test_predictions = best_model.transform(testdata2)
evaluator = RegressionEvaluator(labelCol="LogClaimNb", predictionCol="prediction", metricName="rmse")
test_rmse = evaluator.evaluate(test_predictions)
print("Best RegParam={}, Test RMSE={}".format(best_regParam, test_rmse))

# Print the model coefficients
print("Model Coefficients: ", best_model.stages[-1].coefficients.values)

print(" ------------------ Question B subpart b, c, ii L2-Regularized Linear Regression ------------------------------")
print("\n")

lr = LinearRegression(labelCol="LogClaimNb", featuresCol="features", regParam=0.1, elasticNetParam=0.0, solver="normal")
pipeline = Pipeline(stages=[featureAssembler, lr])

regParams = [0.001, 0.01, 0.1, 1, 10]
rmse_values = []
rmse_values1 = []
start = time.time()
for regParam in regParams:
    lr.setRegParam(regParam)
    model = pipeline.fit(trainingData2)
    predictions = model.transform(validationData2)
    evaluator = RegressionEvaluator(labelCol="LogClaimNb", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    rmse_values.append(rmse)
    print("RegParam={}, Validation RMSE={}".format(regParam, rmse))
    predictions = model.transform(trainingData2)
    rmse = evaluator.evaluate(predictions)
    rmse_values1.append(rmse)
    print("RegParam={}, Training RMSE={}".format(regParam, rmse))
end = time.time()
print("\n Time Taken for Tuning L2-Linear Regression is {} units".format(end-start) )


plt.clf()
plt.plot(regParams, rmse_values, 'ro-', label = 'Validation score', color = 'r')
plt.plot(regParams, rmse_values1, 'ro-', label = 'Training score', color = 'b')
plt.legend()
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('RMSE')
plt.title('Validation Curve')
plt.savefig('../Output/L2_Linear_Curve.png')



best_regParam = regParams[np.argmin(rmse_values)]
lr.setRegParam(best_regParam)
best_model = pipeline.fit(traindata2)


test_predictions = best_model.transform(testdata2)
evaluator = RegressionEvaluator(labelCol="LogClaimNb", predictionCol="prediction", metricName="rmse")
test_rmse = evaluator.evaluate(test_predictions)
print("Best RegParam={}, Test RMSE={}".format(best_regParam, test_rmse))

#Printing the model coefficients
print("Model Coefficients: ", best_model.stages[-1].coefficients.values)


print(" ------------------ Question B subpart b, c, iii L1-Regularized Logistic Regression ------------------------------")
print("\n")

trainingData3 = trainingData.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','NZClaim')
validationData3 = validationData.select('Scaled_features','Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe','NZClaim')

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

featureAssembler = VectorAssembler(inputCols=['Scaled_features', 'Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe'], outputCol='features')
lr = LogisticRegression(labelCol="NZClaim", featuresCol="features", regParam=0.1, elasticNetParam=1, family="binomial")


pipeline = Pipeline(stages=[featureAssembler, lr])

regParams = [0.001, 0.01, 0.1, 1, 10]
accuracy_values = []
accuracy_values1 = []
start = time.time()
for regParam in regParams:
    lr.setRegParam(regParam)
    model = pipeline.fit(trainingData3)
    predictions = model.transform(validationData3)
    evaluator = MulticlassClassificationEvaluator(labelCol="NZClaim", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    accuracy_values.append(accuracy)
    print("RegParam={}, Validation Accuracy={}".format(regParam, accuracy))
    predictions = model.transform(trainingData3)
    accuracy = evaluator.evaluate(predictions)
    accuracy_values1.append(accuracy)
    print("RegParam={}, Training Accuracy={}".format(regParam, accuracy))
end = time.time()
print("\n Time Taken for Tuning L1-Logistic Regression is {} units".format(end-start) )

plt.clf()
fig, ax = plt.subplots()
ax.plot(regParams, accuracy_values, 'ro-', label = 'Validation accuracy', color = 'r')
ax.plot(regParams, accuracy_values1, 'ro-', label = 'Training accuracy', color = 'b')
ax.legend()
ax.set_xscale('log')
ax.set_xlabel('regParam')
ax.set_ylabel('Accuracy')
ax.set_title('Validation Curve')

# Setting the y-axis limit to the range of accuracy values
accuracy_values = np.array(accuracy)
minimum = np.min(accuracy_values) - 0.0001
maximum = np.max(accuracy_values) + 0.0001
ax.set_ylim(minimum,maximum)
#yticks = np.round(np.linspace(minimum, maximum, num=5), 6)
#ax.set_yticks(yticks)
plt.savefig('../Output/L1_Logistic_Curve.png')

best_regParam = regParams[np.argmax(accuracy_values)]
lr.setRegParam(best_regParam)
best_model = pipeline.fit(traindata3)

# Evaluate the best model on the test set
test_predictions = best_model.transform(testdata3)
evaluator = MulticlassClassificationEvaluator(labelCol="NZClaim", predictionCol="prediction", metricName="accuracy")
test_accuracy = evaluator.evaluate(test_predictions)
print("Best RegParam={}, Test Accuracy={}".format(best_regParam, test_accuracy))
# Print the model coefficients
print("Model Coefficients: ", best_model.stages[-1].coefficients.values)

print(" ------------------ Question B subpart b, c, iii L2-Regularized Logistic Regression ------------------------------")
print("\n")
featureAssembler = VectorAssembler(inputCols=['Scaled_features', 'Area_ohe','VehBrand_ohe','VehGas_ohe','Region_ohe'], outputCol='features')
lr = LogisticRegression(labelCol="NZClaim", featuresCol="features", regParam=0.1, elasticNetParam=0, family="binomial")


pipeline = Pipeline(stages=[featureAssembler, lr])
accuracy_values = []
accuracy_values1 = []
start = time.time()
for regParam in regParams:
    lr.setRegParam(regParam)
    model = pipeline.fit(trainingData3)
    predictions = model.transform(validationData3)
    evaluator = evaluator = MulticlassClassificationEvaluator(labelCol="NZClaim", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    accuracy_values.append(accuracy)
    print("RegParam={}, Validation Accuracy={}".format(regParam, accuracy))
    predictions = model.transform(trainingData3)
    accuracy = evaluator.evaluate(predictions)
    accuracy_values1.append(accuracy)
    print("RegParam={}, Training Accuracy={}".format(regParam, accuracy))
end = time.time()
print("\n Time Taken for Tuning L2-Logistic Regression is {} units".format(end-start) )

plt.clf()
fig, ax = plt.subplots()
ax.plot(regParams, accuracy_values, 'ro-', label = 'Validation accuracy', color = 'r')
ax.plot(regParams, accuracy_values1, 'ro-', label = 'Training accuracy', color = 'b')
ax.legend()
ax.set_xscale('log')
ax.set_xlabel('regParam')
ax.set_ylabel('Accuracy')
ax.set_title('Validation Curve')

#Setting the y-axis limit to the range of accuracy values
accuracy_values = np.array(accuracy)
minimum = np.min(accuracy_values) - 0.0001
maximum = np.max(accuracy_values) + 0.0001
ax.set_ylim(minimum,maximum)
plt.savefig('../Output/L2_Logistic_Curve.png')

best_regParam = regParams[np.argmax(accuracy_values)]
lr.setRegParam(best_regParam)
best_model = pipeline.fit(traindata3)


test_predictions = best_model.transform(testdata3)
evaluator = evaluator = MulticlassClassificationEvaluator(labelCol="NZClaim", predictionCol="prediction", metricName="accuracy")
test_accuracy = evaluator.evaluate(test_predictions)
print("Best RegParam={}, Test Accuracy={}".format(best_regParam, test_accuracy))
# Displaying the model coefficients
print("Model Coefficients: ", best_model.stages[-1].coefficients.values)
