
# Machine Learning with MLLib

* Machine Learning
  * Subclass of Artificial Intelligence (AI)
  * Automatically learn and improve from training without being explicitly programmed
  * Where programmers see $f(x)=y$ as
     * $f$ is a function that a programmer can call
     * $x$ is a set of arguments it can give $f$
     * $y$ is the evaluated answer
  * Data Scientists see $f(x)=y$ as
     * $x$ is the multidimensional data that represents all the features or columns
     * $y$ is a series that represents the "answer" via training
     * $f$ is the function that the model comes up with that after learning from $x$ and $y$

## Data Science Pipeline

<img src="../images/datascience_pipeline.png" alt="data_science_pipeline" width="50%"/>

To perform data science, we must do a lot before we even plug into a model. 

* **Raw Data is Collected** 
  * We must find the data, that's hard already, seek information from :
    * Databases
    * Datapipelines (Kafka, Kafka Streaming, Spark Streaming)
    * Webservices
    * RDF/JSON-LD/SPARQL Sources
* **Data is Processed**
  * Create numerical values for `String` elements (e.g. "Male" to `0`, "Female" to `1`)
  * Create scaled values for numbers that don't start at 0 (e.g. Blood Pressure)
  * Engineer columns to create new other columns (e.g. weekends, holidays, financial quarters from dates)
* **Data is Cleaned**
  * Remove `null` or `NaN`
  * Remove Duplicates
  * Remove irrelevant columns (e.g. `id`)
  * Remove columns that are too correlated (e.g. Fahrenheit and Celsius)
* **Exploratory Data Analysis**
  * See what you can find
  * Ask questions
  * Ask domain experts
  * Ask data science specialist
  * Seek Relationships
* **Visualizations**
  * Combined with above. In a graph can we see:
    * Blobs
    * Lines
    * A way we can drive a line through the data?
* **Machine Learning Statistical Models**
  * Split the data into training and testing
  * Now that we know what we are looking for, apply one model or a few
  * See how well they perform with scoring 

## Data Terms

<img src="../images/processing_data_1.png" alt="processing_data_1" width="50%"/>

## What answers is the ML model looking for?

* Classification
  * _survive_, _did not survive_ (binary classification)
  * _rain_, _sunny, _cloudy_, _overcast_
  * _buy_, _dontbuy_
* Regression
  * Given _handsize_ what is the _height_
  * Given _age_ what is the _choelesterol_
  * Given _salary_ what is their credit limit?

## Detect a target column

* For supervised learning, one of the columns is the _target_ or the answer
* Determine which column is the target and note it

<img src="../images/processing_data_2.png" alt="processing_data_2" width="50%"/>

## Split the data into training and testing sets

* Select a 70% - 30% split of the data
  * 70% will be used for training
  * 30% will be used for testing
* You can choose 60% - 40% or any combination 

<img src="../images/processing_data_3.png" alt="processing_data_3" width="50%"/>

## Splits should be random 

* There is a danger of doing the above, since the first part of may not have targets in the testing, and we would end up with terrible scores
* It is wise to mix up the training and testing
* In Spark we can do so with a _random seed_


<img src="../images/processing_data_4.png" alt="processing_data_4" width="50%"/>

## Select a model and split training and testing

* Select a model with the characteristics that you are looking for
* Split your training and testing

<img src="../images/processing_data_5.png" alt="processing_data_5" width="50%"/>

## Undergo the training phase

* Provide the model by calling the `fit` method
* This will start the model by determining which function best maps the data for data it has not seen before including testing data and real life data
<img src="../images/processing_data_6.png" alt="processing_data_6" width="50%"/>

# Reserve the actual results

* Depending which Machine Learning product you choose, you may remove the actual data, or just understand what the "actual" testing results should be.
* In Spark, the API doesn't mind if you leave the actual results in the test since it will likely give a new `DataFrame` with the new predicted results as columns

<img src="../images/processing_data_7.png" alt="processing_data_7" width="50%"/>

## Giving the model data it has not seen before

* We will challenge the model it has not seen before
* This will come up with predicted values

<img src="../images/processing_data_8.png" alt="processing_data_8" width="50%"/>

# Match the results

* Given what was predicted and the actual values that were reserved earlier, we can take them and score how well our model did

<img src="../images/processing_data_9.png" alt="processing_data_9" width="50%"/>

# Selecting the perfect score

* Select a score depends on what we were doing a classification or regression.
* For Regression, some popular metrics include:
  * Mean Absolute Error
  * Mean Squared Error / Root Mean Squared Error
* For Classification, some popular metrics include:
  * Confusion Metrics
  * Area Under Curve / Receiver Operating Characteristics

<img src="../images/processing_data_10.png" alt="processing_data_10" width="50%"/>


```scala
import org.apache.spark.sql.Row

val frame = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("../data/heart.csv")

frame.show()
```

    +---+---+---+--------+----+---+-------+-------+-----+-------+-----+---+----+------+
    |age|sex| cp|trestbps|chol|fbs|restecg|thalach|exang|oldpeak|slope| ca|thal|target|
    +---+---+---+--------+----+---+-------+-------+-----+-------+-----+---+----+------+
    | 63|  1|  3|     145| 233|  1|      0|    150|    0|    2.3|    0|  0|   1|     1|
    | 37|  1|  2|     130| 250|  0|      1|    187|    0|    3.5|    0|  0|   2|     1|
    | 41|  0|  1|     130| 204|  0|      0|    172|    0|    1.4|    2|  0|   2|     1|
    | 56|  1|  1|     120| 236|  0|      1|    178|    0|    0.8|    2|  0|   2|     1|
    | 57|  0|  0|     120| 354|  0|      1|    163|    1|    0.6|    2|  0|   2|     1|
    | 57|  1|  0|     140| 192|  0|      1|    148|    0|    0.4|    1|  0|   1|     1|
    | 56|  0|  1|     140| 294|  0|      0|    153|    0|    1.3|    1|  0|   2|     1|
    | 44|  1|  1|     120| 263|  0|      1|    173|    0|    0.0|    2|  0|   3|     1|
    | 52|  1|  2|     172| 199|  1|      1|    162|    0|    0.5|    2|  0|   3|     1|
    | 57|  1|  2|     150| 168|  0|      1|    174|    0|    1.6|    2|  0|   2|     1|
    | 54|  1|  0|     140| 239|  0|      1|    160|    0|    1.2|    2|  0|   2|     1|
    | 48|  0|  2|     130| 275|  0|      1|    139|    0|    0.2|    2|  0|   2|     1|
    | 49|  1|  1|     130| 266|  0|      1|    171|    0|    0.6|    2|  0|   2|     1|
    | 64|  1|  3|     110| 211|  0|      0|    144|    1|    1.8|    1|  0|   2|     1|
    | 58|  0|  3|     150| 283|  1|      0|    162|    0|    1.0|    2|  0|   2|     1|
    | 50|  0|  2|     120| 219|  0|      1|    158|    0|    1.6|    1|  0|   2|     1|
    | 58|  0|  2|     120| 340|  0|      1|    172|    0|    0.0|    2|  0|   2|     1|
    | 66|  0|  3|     150| 226|  0|      1|    114|    0|    2.6|    0|  0|   2|     1|
    | 43|  1|  0|     150| 247|  0|      1|    171|    0|    1.5|    2|  0|   2|     1|
    | 69|  0|  3|     140| 239|  0|      1|    151|    0|    1.8|    2|  2|   2|     1|
    +---+---+---+--------+----+---+-------+-------+-----+-------+-----+---+----+------+
    only showing top 20 rows
    
    




    import org.apache.spark.sql.Row
    frame: org.apache.spark.sql.DataFrame = [age: int, sex: int ... 12 more fields]
    



## Showing the schema


```scala
frame.printSchema()
```

    root
     |-- age: integer (nullable = true)
     |-- sex: integer (nullable = true)
     |-- cp: integer (nullable = true)
     |-- trestbps: integer (nullable = true)
     |-- chol: integer (nullable = true)
     |-- fbs: integer (nullable = true)
     |-- restecg: integer (nullable = true)
     |-- thalach: integer (nullable = true)
     |-- exang: integer (nullable = true)
     |-- oldpeak: double (nullable = true)
     |-- slope: integer (nullable = true)
     |-- ca: integer (nullable = true)
     |-- thal: integer (nullable = true)
     |-- target: integer (nullable = true)
    
    

## Isolate the features into a feature column

* Data Scientists call features columns
* We need a column with all the features for each model


```scala
import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler()
    .setInputCols(Array("age"))
    .setOutputCol("features")
```




    import org.apache.spark.ml.feature.VectorAssembler
    assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_df0132477920
    



## The `target` column of our supervised learning needs to be renamed into a `label` 

* The model requires that features are isolated into it's onw column
* Notice the `features` is a list of elements we require to plugin
* We are going to calculate the regression between `chol` and `age`, so age will be our label


```scala
val newFrame = assembler.transform(frame).withColumnRenamed("chol", "label")
newFrame.show()
```

    +---+---+---+--------+-----+---+-------+-------+-----+-------+-----+---+----+------+--------+
    |age|sex| cp|trestbps|label|fbs|restecg|thalach|exang|oldpeak|slope| ca|thal|target|features|
    +---+---+---+--------+-----+---+-------+-------+-----+-------+-----+---+----+------+--------+
    | 63|  1|  3|     145|  233|  1|      0|    150|    0|    2.3|    0|  0|   1|     1|  [63.0]|
    | 37|  1|  2|     130|  250|  0|      1|    187|    0|    3.5|    0|  0|   2|     1|  [37.0]|
    | 41|  0|  1|     130|  204|  0|      0|    172|    0|    1.4|    2|  0|   2|     1|  [41.0]|
    | 56|  1|  1|     120|  236|  0|      1|    178|    0|    0.8|    2|  0|   2|     1|  [56.0]|
    | 57|  0|  0|     120|  354|  0|      1|    163|    1|    0.6|    2|  0|   2|     1|  [57.0]|
    | 57|  1|  0|     140|  192|  0|      1|    148|    0|    0.4|    1|  0|   1|     1|  [57.0]|
    | 56|  0|  1|     140|  294|  0|      0|    153|    0|    1.3|    1|  0|   2|     1|  [56.0]|
    | 44|  1|  1|     120|  263|  0|      1|    173|    0|    0.0|    2|  0|   3|     1|  [44.0]|
    | 52|  1|  2|     172|  199|  1|      1|    162|    0|    0.5|    2|  0|   3|     1|  [52.0]|
    | 57|  1|  2|     150|  168|  0|      1|    174|    0|    1.6|    2|  0|   2|     1|  [57.0]|
    | 54|  1|  0|     140|  239|  0|      1|    160|    0|    1.2|    2|  0|   2|     1|  [54.0]|
    | 48|  0|  2|     130|  275|  0|      1|    139|    0|    0.2|    2|  0|   2|     1|  [48.0]|
    | 49|  1|  1|     130|  266|  0|      1|    171|    0|    0.6|    2|  0|   2|     1|  [49.0]|
    | 64|  1|  3|     110|  211|  0|      0|    144|    1|    1.8|    1|  0|   2|     1|  [64.0]|
    | 58|  0|  3|     150|  283|  1|      0|    162|    0|    1.0|    2|  0|   2|     1|  [58.0]|
    | 50|  0|  2|     120|  219|  0|      1|    158|    0|    1.6|    1|  0|   2|     1|  [50.0]|
    | 58|  0|  2|     120|  340|  0|      1|    172|    0|    0.0|    2|  0|   2|     1|  [58.0]|
    | 66|  0|  3|     150|  226|  0|      1|    114|    0|    2.6|    0|  0|   2|     1|  [66.0]|
    | 43|  1|  0|     150|  247|  0|      1|    171|    0|    1.5|    2|  0|   2|     1|  [43.0]|
    | 69|  0|  3|     140|  239|  0|      1|    151|    0|    1.8|    2|  2|   2|     1|  [69.0]|
    +---+---+---+--------+-----+---+-------+-------+-----+-------+-----+---+----+------+--------+
    only showing top 20 rows
    
    




    newFrame: org.apache.spark.sql.DataFrame = [age: int, sex: int ... 13 more fields]
    




```scala
val focusedFrame = newFrame.select("label", "features")
```




    focusedFrame: org.apache.spark.sql.DataFrame = [label: int, features: vector]
    



## Split the data

* We need to split the data training and testing
* We are going to split 70% training - 30% testing
* It will be essential that we put a random seed to randomly select the rows (observations)


```scala
import org.apache.spark.sql.Dataset
val splitData: Array[Dataset[Row]] = focusedFrame.randomSplit(Array(0.7, 0.3), seed = 1234L)
val trainingData = splitData(0)
trainingData.show()
```

    +-----+--------+
    |label|features|
    +-----+--------+
    |  149|  [49.0]|
    |  149|  [71.0]|
    |  157|  [41.0]|
    |  160|  [45.0]|
    |  164|  [62.0]|
    |  166|  [61.0]|
    |  167|  [40.0]|
    |  168|  [57.0]|
    |  169|  [44.0]|
    |  172|  [41.0]|
    |  175|  [38.0]|
    |  175|  [38.0]|
    |  175|  [51.0]|
    |  176|  [59.0]|
    |  177|  [43.0]|
    |  177|  [46.0]|
    |  177|  [59.0]|
    |  177|  [65.0]|
    |  178|  [60.0]|
    |  180|  [42.0]|
    +-----+--------+
    only showing top 20 rows
    
    




    import org.apache.spark.sql.Dataset
    splitData: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: int, features: vector], [label: int, features: vector])
    trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
    




```scala
val testingData = splitData(1)
testingData.show()
```

    +-----+--------+
    |label|features|
    +-----+--------+
    |  126|  [57.0]|
    |  131|  [57.0]|
    |  141|  [44.0]|
    |  174|  [70.0]|
    |  184|  [56.0]|
    |  185|  [60.0]|
    |  188|  [54.0]|
    |  193|  [56.0]|
    |  196|  [52.0]|
    |  197|  [53.0]|
    |  197|  [76.0]|
    |  198|  [35.0]|
    |  199|  [39.0]|
    |  201|  [54.0]|
    |  204|  [59.0]|
    |  206|  [54.0]|
    |  211|  [43.0]|
    |  212|  [52.0]|
    |  212|  [59.0]|
    |  212|  [66.0]|
    +-----+--------+
    only showing top 20 rows
    
    




    testingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
    



## Linear Regression

* Linear Regression is a model that draws a line through the data points
* After training it provides a coefficient (line slope) and intercept e.g. $mx + b$
* Here we will use some standard parameters (called hyperparameters by data scientists)
* For a visual understanding of linear regression [enjoy this visualization](http://setosa.io/ev/ordinary-least-squares-regression/)


```scala
import org.apache.spark.ml.regression.LinearRegression
val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
```




    import org.apache.spark.ml.regression.LinearRegression
    lr: org.apache.spark.ml.regression.LinearRegression = linReg_d3bf2c407085
    



## Training 


```scala
val lrModel = lr.fit(trainingData)
```




    lrModel: org.apache.spark.ml.regression.LinearRegressionModel = linReg_d3bf2c407085
    



## Print the coefficient (slope) and intercept


```scala
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
```

    Coefficients: [1.5602976407592346] Intercept: 163.61093163371797
    

## Summarize the model over the training set and print out some metrics


```scala
val summary = lrModel.evaluate(testingData)
println(f"Mean Squared Error: ${summary.meanSquaredError}%1.2f")
println(f"Mean Absolute Error: ${summary.meanAbsoluteError}%1.2f")
```

    Mean Squared Error: 2155.67
    Mean Absolute Error: 35.70
    




    summary: org.apache.spark.ml.regression.LinearRegressionSummary = org.apache.spark.ml.regression.LinearRegressionSummary@5428536f
    



## Decision Tree

* Decision Trees will find the information required to split the data with a series of `if` statements internally
* How it does so is with a recursive split and determining a purity score
* Decision Trees take multiple feature (column) data


### Use `VectorAssembler` to arrange all the features 

* A Decision Tree can use all features so we will include that
* The column with all the features will be called `features`


```scala
import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler()
      .setInputCols(Array("age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"))
      .setOutputCol("features")
```




    import org.apache.spark.ml.feature.VectorAssembler
    assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_5d5b5aa3e820
    



### Perform the transformation

* Notice the `features` column and the elements that it contains
* We will plug in the data along with the `target` on whether or not they will have heart disease


```scala
val transformed = assembler.transform(frame)
transformed.show()
```

    +---+---+---+--------+----+---+-------+-------+-----+-------+-----+---+----+------+--------------------+
    |age|sex| cp|trestbps|chol|fbs|restecg|thalach|exang|oldpeak|slope| ca|thal|target|            features|
    +---+---+---+--------+----+---+-------+-------+-----+-------+-----+---+----+------+--------------------+
    | 63|  1|  3|     145| 233|  1|      0|    150|    0|    2.3|    0|  0|   1|     1|[63.0,1.0,3.0,145...|
    | 37|  1|  2|     130| 250|  0|      1|    187|    0|    3.5|    0|  0|   2|     1|[37.0,1.0,2.0,130...|
    | 41|  0|  1|     130| 204|  0|      0|    172|    0|    1.4|    2|  0|   2|     1|[41.0,0.0,1.0,130...|
    | 56|  1|  1|     120| 236|  0|      1|    178|    0|    0.8|    2|  0|   2|     1|[56.0,1.0,1.0,120...|
    | 57|  0|  0|     120| 354|  0|      1|    163|    1|    0.6|    2|  0|   2|     1|[57.0,0.0,0.0,120...|
    | 57|  1|  0|     140| 192|  0|      1|    148|    0|    0.4|    1|  0|   1|     1|[57.0,1.0,0.0,140...|
    | 56|  0|  1|     140| 294|  0|      0|    153|    0|    1.3|    1|  0|   2|     1|[56.0,0.0,1.0,140...|
    | 44|  1|  1|     120| 263|  0|      1|    173|    0|    0.0|    2|  0|   3|     1|[44.0,1.0,1.0,120...|
    | 52|  1|  2|     172| 199|  1|      1|    162|    0|    0.5|    2|  0|   3|     1|[52.0,1.0,2.0,172...|
    | 57|  1|  2|     150| 168|  0|      1|    174|    0|    1.6|    2|  0|   2|     1|[57.0,1.0,2.0,150...|
    | 54|  1|  0|     140| 239|  0|      1|    160|    0|    1.2|    2|  0|   2|     1|[54.0,1.0,0.0,140...|
    | 48|  0|  2|     130| 275|  0|      1|    139|    0|    0.2|    2|  0|   2|     1|[48.0,0.0,2.0,130...|
    | 49|  1|  1|     130| 266|  0|      1|    171|    0|    0.6|    2|  0|   2|     1|[49.0,1.0,1.0,130...|
    | 64|  1|  3|     110| 211|  0|      0|    144|    1|    1.8|    1|  0|   2|     1|[64.0,1.0,3.0,110...|
    | 58|  0|  3|     150| 283|  1|      0|    162|    0|    1.0|    2|  0|   2|     1|[58.0,0.0,3.0,150...|
    | 50|  0|  2|     120| 219|  0|      1|    158|    0|    1.6|    1|  0|   2|     1|[50.0,0.0,2.0,120...|
    | 58|  0|  2|     120| 340|  0|      1|    172|    0|    0.0|    2|  0|   2|     1|[58.0,0.0,2.0,120...|
    | 66|  0|  3|     150| 226|  0|      1|    114|    0|    2.6|    0|  0|   2|     1|[66.0,0.0,3.0,150...|
    | 43|  1|  0|     150| 247|  0|      1|    171|    0|    1.5|    2|  0|   2|     1|[43.0,1.0,0.0,150...|
    | 69|  0|  3|     140| 239|  0|      1|    151|    0|    1.8|    2|  2|   2|     1|[69.0,0.0,3.0,140...|
    +---+---+---+--------+----+---+-------+-------+-----+-------+-----+---+----+------+--------------------+
    only showing top 20 rows
    
    




    transformed: org.apache.spark.sql.DataFrame = [age: int, sex: int ... 13 more fields]
    



### Applying the Decision Tree Model

* Plugging in the model, we will direct it to the `feature` column, and the `target`


```scala
import org.apache.spark.ml.classification.DecisionTreeClassifier
val decisionTreeClassifier = new DecisionTreeClassifier()
      .setFeaturesCol("features")
      .setLabelCol("target")
```




    import org.apache.spark.ml.classification.DecisionTreeClassifier
    decisionTreeClassifier: org.apache.spark.ml.classification.DecisionTreeClassifier = dtc_60c1e553e8e0
    



### Splitting the data for training and testing


```scala
val splitData = newFrame.randomSplit(Array(0.7, 0.3), seed = 1234L)
val trainingData = splitData(0)
val testingData = splitData(1)
```




    splitData: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([age: int, sex: int ... 13 more fields], [age: int, sex: int ... 13 more fields])
    trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [age: int, sex: int ... 13 more fields]
    testingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [age: int, sex: int ... 13 more fields]
    



### Training the model


```scala
val model = decisionTreeClassifier.fit(trainingData)
```




    model: org.apache.spark.ml.classification.DecisionTreeClassificationModel = DecisionTreeClassificationModel (uid=dtc_60c1e553e8e0) of depth 4 with 15 nodes
    



### Calling `transform` to view the data


```scala
val result = model.transform(testingData)
result.show(10)
```

    +---+---+---+--------+-----+---+-------+-------+-----+-------+-----+---+----+------+--------+-------------+--------------------+----------+
    |age|sex| cp|trestbps|label|fbs|restecg|thalach|exang|oldpeak|slope| ca|thal|target|features|rawPrediction|         probability|prediction|
    +---+---+---+--------+-----+---+-------+-------+-----+-------+-----+---+----+------+--------+-------------+--------------------+----------+
    | 29|  1|  1|     130|  204|  0|      0|    202|    0|    0.0|    2|  0|   2|     1|  [29.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    | 34|  0|  1|     118|  210|  0|      1|    192|    0|    0.7|    2|  0|   2|     1|  [34.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    | 34|  1|  3|     118|  182|  0|      0|    174|    0|    0.0|    2|  0|   2|     1|  [34.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    | 39|  0|  2|     138|  220|  0|      1|    152|    0|    0.0|    1|  0|   2|     1|  [39.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    | 41|  1|  1|     135|  203|  0|      1|    132|    0|    0.0|    1|  0|   1|     1|  [41.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    | 41|  1|  2|     112|  250|  0|      1|    179|    0|    0.0|    2|  0|   2|     1|  [41.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    | 42|  1|  0|     136|  315|  0|      1|    125|    1|    1.8|    1|  0|   1|     0|  [42.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    | 42|  1|  2|     120|  240|  1|      1|    194|    0|    0.8|    0|  0|   3|     1|  [42.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    | 43|  0|  2|     122|  213|  0|      1|    165|    0|    0.2|    1|  0|   2|     1|  [43.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    | 43|  1|  0|     120|  177|  0|      0|    120|    1|    2.5|    1|  0|   3|     0|  [43.0]|  [13.0,36.0]|[0.26530612244897...|       1.0|
    +---+---+---+--------+-----+---+-------+-------+-----+-------+-----+---+----+------+--------+-------------+--------------------+----------+
    only showing top 10 rows
    
    




    result: org.apache.spark.sql.DataFrame = [age: int, sex: int ... 16 more fields]
    



### Determining the score and our performance

* We will procure the `org.apache.spark.ml.evaluation.BinaryClassificationEvaluator` for this decision tree
* This is a binary response: Has heart disease, Does not have heart disease
* The default score for the `BinaryClassificationEvaluator` is the AUC (Area Under the Curve) / ROC (Receiving Operating Characteristic) Score which determines the area of the false positive rate against the true positive rate.
* The best AUC, is 1.0


```scala
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluator = new BinaryClassificationEvaluator()
                    .setLabelCol("target") 
                    .setRawPredictionCol("rawPrediction") 
```




    import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
    evaluator: org.apache.spark.ml.evaluation.BinaryClassificationEvaluator = binEval_77f550ecaecd
    



### Displaying the final score


```scala
val aucScore = evaluator.evaluate(result)
println(s"AUC Score = $aucScore")
```

    AUC Score = 0.5243055555555556
    




    aucScore: Double = 0.5243055555555556
    



## What does a random forest do?

* Random Forest takes multiple trees and determines a score based on average or voting
* This is the wisdom of the crowd
* Each tree can either institute a (WR) with replacement, or (WOR) without replacement
* With Replacement is like measuring fish and throwing the fish back in the water. You may get the same one again


```scala
import org.apache.spark.ml.classification.RandomForestClassifier
val rf = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("target")
      .setNumTrees(100)
```




    import org.apache.spark.ml.classification.RandomForestClassifier
    rf: org.apache.spark.ml.classification.RandomForestClassifier = rfc_8156aff317f3
    




```scala
val model = rf.fit(trainingData)
```




    model: org.apache.spark.ml.classification.RandomForestClassificationModel = RandomForestClassificationModel (uid=rfc_8156aff317f3) with 100 trees
    




```scala
val result = model.transform(testingData)
result.show(10)
```

    +---+---+---+--------+-----+---+-------+-------+-----+-------+-----+---+----+------+--------+--------------------+--------------------+----------+
    |age|sex| cp|trestbps|label|fbs|restecg|thalach|exang|oldpeak|slope| ca|thal|target|features|       rawPrediction|         probability|prediction|
    +---+---+---+--------+-----+---+-------+-------+-----+-------+-----+---+----+------+--------+--------------------+--------------------+----------+
    | 29|  1|  1|     130|  204|  0|      0|    202|    0|    0.0|    2|  0|   2|     1|  [29.0]|[33.7344011536839...|[0.33734401153683...|       1.0|
    | 34|  0|  1|     118|  210|  0|      1|    192|    0|    0.7|    2|  0|   2|     1|  [34.0]|[33.7344011536839...|[0.33734401153683...|       1.0|
    | 34|  1|  3|     118|  182|  0|      0|    174|    0|    0.0|    2|  0|   2|     1|  [34.0]|[33.7344011536839...|[0.33734401153683...|       1.0|
    | 39|  0|  2|     138|  220|  0|      1|    152|    0|    0.0|    1|  0|   2|     1|  [39.0]|[41.6632611947362...|[0.41663261194736...|       1.0|
    | 41|  1|  1|     135|  203|  0|      1|    132|    0|    0.0|    1|  0|   1|     1|  [41.0]|[18.1589430852963...|[0.18158943085296...|       1.0|
    | 41|  1|  2|     112|  250|  0|      1|    179|    0|    0.0|    2|  0|   2|     1|  [41.0]|[18.1589430852963...|[0.18158943085296...|       1.0|
    | 42|  1|  0|     136|  315|  0|      1|    125|    1|    1.8|    1|  0|   1|     0|  [42.0]|[15.6839430852963...|[0.15683943085296...|       1.0|
    | 42|  1|  2|     120|  240|  1|      1|    194|    0|    0.8|    0|  0|   3|     1|  [42.0]|[15.6839430852963...|[0.15683943085296...|       1.0|
    | 43|  0|  2|     122|  213|  0|      1|    165|    0|    0.2|    1|  0|   2|     1|  [43.0]|[36.6461551756008...|[0.36646155175600...|       1.0|
    | 43|  1|  0|     120|  177|  0|      0|    120|    1|    2.5|    1|  0|   3|     0|  [43.0]|[36.6461551756008...|[0.36646155175600...|       1.0|
    +---+---+---+--------+-----+---+-------+-------+-----+-------+-----+---+----+------+--------+--------------------+--------------------+----------+
    only showing top 10 rows
    
    




    result: org.apache.spark.sql.DataFrame = [age: int, sex: int ... 16 more fields]
    




```scala
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluator = new BinaryClassificationEvaluator()
                    .setLabelCol("target") 
                    .setRawPredictionCol("rawPrediction") 
```




    import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
    evaluator: org.apache.spark.ml.evaluation.BinaryClassificationEvaluator = binEval_cc223e13d819
    




```scala
val aucScore = evaluator.evaluate(result)
println(s"AUC Score = $aucScore")
```

    AUC Score = 0.5779671717171717
    




    aucScore: Double = 0.5779671717171717
    



## Lab: Machine Learning with the Titanic Dataset

**Step 1** Read in two datasources from the `../data` folder, one is meant for training the other for testing, you do not need to split or clean the data. 

  * titanic_train_clean.csv
  * titanic_test_clean.csv 

**Step 2:** Do the necessary preparation

**Step 3:** Prepare the dataset to be used with a `VectorAssembler`

**Step 4:** Create a `DecisionTreeClassifier`

**Step 5:** `fit` the training data, and use `trasnform` with the testing data

**Step 6:** Determine how well it did with a `BinaryClassificationEvaluator`

**Step 7:** If time remaining, try using a `RandomForestClassifier`

**Step 8:** `fit` the training data, and use `transform` with the testing data

**Step 9:** Determine how well it did with a `BinaryClassificationEvaluator`
