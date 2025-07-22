# ACM Summer School 2025 – Day 2 - 22 July 2025 - Morning Session

**Distributed Data Analytics in Supercomputing Systems**  
*Josep Lluís Berral García (CROMAI / Data Centric Computing – UPC/BSC)*

This document contains all hands-on practical exercises from the course, organized by topic with explanations and references to the original slides.
https://bit.ly/40tZpvJ
---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Basic Spark Operations](#basic-spark-operations)
3. [Spark SQL and DataFrames](#spark-sql-and-dataframes)
4. [Spark Machine Learning (MLlib)](#spark-machine-learning-mllib)
5. [Spark Streaming](#spark-streaming)
6. [MareNostrum Execution](#marenostrum-execution)

---

## Environment Setup

### Step 0: Setting Up the Spark Environment
*Reference: Slides 8-12*

This section sets up a distributed Spark environment using Singularity containers, creating one master node and two worker nodes on a single machine.

**Set environment variables:**
```bash
IMAGE=dcc-spark02.simg
HOST_MASTER=localhost
MASTER_OUT=spark_master
WORKER_OUT_1=spark_worker_1
WORKER_OUT_2=spark_worker_2
SPARK_TEMP=/tmp/spark-work
```
*Defines the container image and output file names for logging.*

**Create Spark temporary folder:**
```bash
mkdir -p $SPARK_TEMP
```
*Creates a directory for Spark to store temporary work files.*

**Set worker resource configuration:**
```bash
SPARK_WORKER_ARGS='-c 1 -m 1g'
```
*Allocates 1 CPU core and 1GB memory per worker.*

**Start Spark master:**
```bash
singularity run $IMAGE "org.apache.spark.deploy.master.Master" \
    --host ${HOST_MASTER} > ${MASTER_OUT}.out 2> ${MASTER_OUT}.err &
```
*Launches the Spark master service that coordinates job execution.*

**Start first Spark worker:**
```bash
singularity run $IMAGE 'org.apache.spark.deploy.worker.Worker' \
    -d ${SPARK_TEMP} ${SPARK_WORKER_ARGS} \
    spark://${HOST_MASTER}:7077 > ${WORKER_OUT_1}.out 2> ${WORKER_OUT_1}.err &
```
*Starts a worker node that will execute tasks assigned by the master.*

**Start second Spark worker:**
```bash
singularity run $IMAGE 'org.apache.spark.deploy.worker.Worker' \
    -d ${SPARK_TEMP} ${SPARK_WORKER_ARGS} \
    spark://${HOST_MASTER}:7077 > ${WORKER_OUT_2}.out 2> ${WORKER_OUT_2}.err &
```
*Adds a second worker for parallel processing.*

### Checking Container Status
*Reference: Slide 11*

**Check that containers are running:**
```bash
ps a | grep spark | grep -v grep
```
*Verifies that master and worker processes are active.*

### Connect as a Client
*Reference: Slide 12*

**Start Spark shell (Scala):**
```bash
singularity exec $IMAGE spark-shell --master spark://${HOST_MASTER}:7077
```
*Opens an interactive Scala shell connected to the Spark cluster.*

**Alternative - Start PySpark:**
```bash
singularity exec $IMAGE pyspark --master spark://${HOST_MASTER}:7077
```
*For Python users: opens a Python shell with Spark context.*

**Alternative - Start SparkR:**
```bash
singularity exec $IMAGE sparkR --master spark://${HOST_MASTER}:7077
```
*For R users: opens an R session with Spark capabilities.*

### Environment Cleanup
*Reference: Slides 11, 51*

**Kill all Spark processes when done:**
```bash
kill $(ps aux | grep spark | grep -v grep | awk '{ print $2 }')
```
*Terminates all Spark-related processes to free resources.*

**Exit Scala shell:**
```scala
:q
```
*Use ':q' (colon-q) to exit the Spark shell properly.*

---

## Basic Spark Operations

### Practical 1: Basic Word Count with RDD
*Reference: Slides 14-16*

The classic MapReduce example demonstrating distributed text processing.

**Load text file:**
```scala
val textFile = sc.textFile("/nord3/spark/README.md")
```
*Creates an RDD from a text file, automatically parallelizing the data.*

**Check that we can read the file:**
```scala
textFile.first()
```
*Returns the first line to verify successful file loading.*

**Split lines into words:**
```scala
val textFlatMap = textFile.flatMap(line => line.split(" "))
```
*Transforms each line into individual words using flatMap.*

**Map words to (word, 1) pairs:**
```scala
val words = textFlatMap.map(word => (word, 1))
```
*Creates key-value pairs for counting.*

**Reduce by key to count occurrences:**
```scala
val counts = words.reduceByKey((x, y) => x + y)
```
*Aggregates counts for each unique word across the cluster.*

**Show results (triggers lazy execution):**
```scala
counts.take(5).foreach(println)
```
*Spark uses lazy evaluation - computation happens only when results are needed.*

### Practical 2: Enhanced Word Count with Filtering and Sorting
*Reference: Slide 17*

Demonstrates data filtering and sorting operations on distributed datasets.

**Order the results by count (descending):**
```scala
val ranking = counts.sortBy(x => x._2, false)
```
*Sorts words by frequency, with most common first (false = descending).*

**Collect all data locally:**
```scala
val local_result = ranking.collect()
```
*Brings all distributed data to the driver node - use carefully with large datasets!*

**Show top 5 results:**
```scala
ranking.take(5).foreach(println)
```
*Efficiently retrieves only the top 5 results without collecting all data.*

**Filter out empty words:**
```scala
val cleancount = counts.filter(x => {!"".equals(x._1)})
```
*Removes empty strings from the word count.*

**Sort filtered results:**
```scala
val cleanrank = cleancount.sortBy(x => x._2, false)
```
*Sorts the cleaned data by count.*

**Collect and display cleaned results:**
```scala
val local_cleanrank = cleanrank.collect()
local_cleanrank.take(5).foreach(println)
```
*Shows top 5 words after filtering.*

---

## Spark SQL and DataFrames

### Practical 3: Loading CSV Data
*Reference: Slides 19-20*

DataFrames provide a higher-level abstraction for structured data processing.

**Prepare the dataset (if needed):**
```bash
wget http://bit.ly/2jZgeZY -O csv_hus.zip
mkdir hus && mv csv_hus.zip hus/ && cd hus && unzip csv_hus.zip && cd ..
```
*Downloads and extracts the US Census housing data.*

**Load CSV as DataFrame:**
```scala
val df = spark.read.format("csv")
  .option("inferSchema", true)
  .option("header", "true")
  .load("./hus/ss13husa.csv")
```
*Reads CSV with automatic schema inference and header row.*

**Explore the data structure:**
```scala
df.show()
```
*Displays first 20 rows in tabular format.*

**View schema information:**
```scala
df.printSchema()
```
*Shows column names and data types.*

**Count total rows:**
```scala
df.count()
```
*Returns the total number of records in the DataFrame.*

### Practical 4: SQL Operations - Selection
*Reference: Slides 22-23*

Demonstrates column selection using both DataFrame API and SQL syntax.

**Register DataFrame as SQL table:**
```scala
df.createGlobalTempView("husa")
```
*Makes DataFrame accessible via SQL queries.*

**Column selection using DataFrame API:**
```scala
df.select("SERIALNO", "RT", "DIVISION", "REGION").show()
```
*Selects specific columns using programmatic API.*

**Column selection using SQL:**
```scala
spark.sql("""
  SELECT SERIALNO, RT, DIVISION, REGION 
  FROM global_temp.husa
""").show()
```
*Same operation using familiar SQL syntax.*

### Practical 5: SQL Operations - Filtering
*Reference: Slide 24*

Shows row filtering (WHERE clause) operations.

**Filtering with DataFrame API:**
```scala
df.select("SERIALNO", "RT", "DIVISION", "REGION")
  .filter("PUMA > 2600")
  .show()
```
*Filters rows where PUMA value exceeds 2600.*

**Filtering with SQL WHERE clause:**
```scala
spark.sql("""
  SELECT SERIALNO, RT, DIVISION, REGION 
  FROM global_temp.husa 
  WHERE PUMA < 2100
""").show()
```
*SQL equivalent showing records with PUMA less than 2100.*

### Practical 6: SQL Operations - Aggregation
*Reference: Slide 25*

Demonstrates grouping and aggregation operations.

**GroupBy with DataFrame API:**
```scala
df.groupBy("DIVISION").count().show()
```
*Counts records per division using DataFrame syntax.*

**GroupBy with SQL:**
```scala
spark.sql("""
  SELECT DIVISION, COUNT(*) as count 
  FROM global_temp.husa 
  GROUP BY DIVISION
""").show()
```
*Same aggregation using SQL GROUP BY.*

**Multiple aggregations:**
```scala
df.groupBy("DIVISION").agg(
  count("*").as("count"),
  avg("SERIALNO").as("avg_serial"),
  max("PUMA").as("max_puma"),
  min("PUMA").as("min_puma")
).show()
```
*Computes multiple statistics per group: count, average, max, and min.*

### Practical 7: Saving and Loading Parquet Files
*Reference: Slide 26*

Parquet is a columnar storage format optimized for analytics.

**Save DataFrame as Parquet:**
```scala
df.write.parquet("./hus/husa.parquet")
```
*Writes DataFrame in efficient Parquet format.*

**Load Parquet file:**
```scala
val pqFileDF = spark.read.parquet("./hus/husa.parquet")
```
*Reads Parquet files back into a DataFrame.*

**Display loaded data:**
```scala
pqFileDF.limit(5).show()
```
*Shows first 5 rows of the loaded Parquet data.*

**Use Parquet file in SQL:**
```scala
pqFileDF.createOrReplaceTempView("parquetFile")
val namesDF = spark.sql("""
  SELECT SERIALNO 
  FROM parquetFile 
  WHERE PUMA < 2100
""")
namesDF.show()
```
*Demonstrates SQL queries on Parquet-backed tables.*

---

## Spark Machine Learning (MLlib)

### Practical 8: Data Preparation for ML
*Reference: Slides 29-30*

Preparing data in the format required for Spark ML algorithms.

**Import ML libraries:**
```scala
import org.apache.spark.ml.linalg.Vectors
```
*Imports vector types used by ML algorithms.*

**Create sample data manually:**
```scala
val data = Seq(
  (Vectors.dense(2.0, 3.0, 5.0), 1.0),
  (Vectors.dense(4.0, 6.0, 7.0), 2.0),
  (Vectors.sparse(3, Seq((1, 1.0), (2, 7.0))), 1.5)
)
```
*Creates feature vectors with labels.*

**Convert to DataFrame:**
```scala
val df = data.toDF("features", "weight")
df.show()
```
*Transforms sequence into ML-ready DataFrame.*

### Practical 9: Linear Regression
*Reference: Slides 31-33*

Complete linear regression workflow from training to evaluation.

**Import regression libraries:**
```scala
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
```
*Loads necessary ML components.*

**Load sample data:**
```scala
var datafile = "./sample_linear_regression_data.txt"
val dataset = spark.read.format("libsvm").load(datafile)
```
*Reads data in LIBSVM format (label feature:value format).*

**Split data for training and testing:**
```scala
val splits = dataset.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)
```
*60% for training, 40% for testing, with caching for performance.*

**Configure and train model:**
```scala
val lr = new LinearRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)
```
*Sets maximum iterations, regularization, and elastic net parameters.*

**Fit the model:**
```scala
val model = lr.fit(training)
```
*Trains the model on training data.*

**View model parameters:**
```scala
println(s"Coefficients: ${model.coefficients}")
println(s"Intercept: ${model.intercept}")
```
*Displays learned weights and bias.*

**Make predictions:**
```scala
val predictions = model.transform(test)
predictions.show()
```
*Applies model to test data.*

**Evaluate model performance:**
```scala
val evaluator = new RegressionEvaluator().setMetricName("mse")
val mse = evaluator.evaluate(predictions.select("prediction", "label"))
println(s"Mean Squared Error = $mse")
```
*Computes mean squared error on test set.*

**Save and load model:**
```scala
model.write.overwrite().save("LR_Model")
val sameModel = LinearRegressionModel.load("LR_Model")
```
*Persists trained model for future use.*

### Practical 10: Support Vector Machine Classification
*Reference: Slides 34-36*

Binary classification using linear SVM.

**Import classification libraries:**
```scala
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
```
*Loads SVM and evaluation components.*

**Load classification data:**
```scala
var datafile = "./sample_libsvm_data.txt"
val dataset = spark.read.format("libsvm").load(datafile)
```
*Reads binary classification dataset.*

**Prepare train/test split:**
```scala
val splits = dataset.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)
```
*Splits data maintaining class distribution.*

**Configure and train SVM:**
```scala
val lsvc = new LinearSVC()
  .setMaxIter(10)
  .setRegParam(0.1)
val model = lsvc.fit(training)
```
*Trains linear SVM with regularization.*

**Generate predictions:**
```scala
val predictions = model.transform(test)
predictions.show()
```
*Classifies test instances.*

**Evaluate accuracy:**
```scala
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions.select("prediction", "label"))
println(s"Test set accuracy = $accuracy")
```
*Measures classification accuracy.*

**Save the model:**
```scala
model.write.overwrite().save("SVM_Model")
val sameModel = LinearSVCModel.load("SVM_Model")
```
*Persists SVM model.*

### Practical 11: K-Means Clustering
*Reference: Slides 37-38*

Unsupervised clustering to find data groupings.

**Import clustering libraries:**
```scala
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
```
*Loads k-means and evaluation tools.*

**Load clustering data:**
```scala
var datafile = "./sample_kmeans_data.txt"
val dataset = spark.read.format("libsvm").load(datafile)
```
*Reads unlabeled data for clustering.*

**Configure and train k-means:**
```scala
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(dataset)
```
*Finds 2 clusters with fixed random seed.*

**View cluster centers:**
```scala
model.clusterCenters.foreach(println)
```
*Displays the centroid of each cluster.*

**Assign clusters:**
```scala
val predictions = model.transform(dataset)
predictions.show()
```
*Shows which cluster each point belongs to.*

**Evaluate clustering quality:**
```scala
val silhouette = new ClusteringEvaluator().evaluate(predictions)
println(s"Silhouette score = $silhouette")
```
*Silhouette score measures cluster cohesion (closer to 1 is better).*

**Compute within-cluster sum of squares:**
```scala
val WSSSE = model.computeCost(dataset)
println(s"Within Set Sum of Squared Errors = $WSSSE")
```
*Lower WSSSE indicates tighter clusters.*

**Save clustering model:**
```scala
model.write.overwrite().save("KMeansModel")
val sameModel = KMeansModel.load("KMeansModel")
```
*Saves cluster definitions for later use.*

### Practical 12: Vector Assembly for ML
*Reference: Slides 39-40*

Converting DataFrame columns into feature vectors for ML.

**Import vector assembler:**
```scala
import org.apache.spark.ml.feature.VectorAssembler
```
*Tool for combining multiple columns into a single vector column.*

**Create sample DataFrame:**
```scala
val dataset = spark.createDataFrame(Seq(
  (0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0),
  (1, 19, 2.0, Vectors.dense(1.0, 15.0, 0.5), 0.0)
)).toDF("id", "hour", "mobile", "userFeatures", "clicked")
```
*Mixed data with scalars and vectors.*

**Configure vector assembler:**
```scala
val assembler = new VectorAssembler()
  .setInputCols(Array("hour", "mobile", "userFeatures"))
  .setOutputCol("features")
```
*Combines selected columns into single feature vector.*

**Apply transformation:**
```scala
val output = assembler.transform(dataset)
```
*Creates new DataFrame with assembled features.*

**View results:**
```scala
val vecdf = output.select("features", "clicked")
vecdf.show(truncate = false)
```
*Shows combined feature vectors ready for ML.*

---

## Spark Streaming

### Practical 13: Basic Streaming Word Count
*Reference: Slides 42-46*

Real-time word counting from socket stream.

**Terminal 1 - Set up Spark streaming application:**

**Import streaming libraries:**
```scala
import org.apache.spark.sql.functions._
import spark.implicits._
```
*Required imports for streaming operations.*

**Create input stream from socket:**
```scala
val lines = spark.readStream
  .format("socket")
  .option("host", "localhost")
  .option("port", 9999)
  .load()
```
*Connects to TCP socket for incoming data.*

**Process streaming data:**
```scala
val words = lines.as[String].flatMap(_.split(" "))
val wordCounts = words.groupBy("value").count()
```
*Splits lines into words and counts occurrences.*

**Configure output to console:**
```scala
val query = wordCounts.writeStream
  .outputMode("complete")
  .format("console")
```
*Shows complete word count table after each batch.*

**Start the streaming query:**
```scala
query.start()
```
*Begins processing incoming data.*

**Terminal 2 - Send data to Spark:**

**Start netcat server:**
```bash
nc -lk 9999
```
*Opens socket on port 9999 for sending text.*

**Type words to send:**
```
hello world
hello spark streaming
world of big data
```
*Each line is processed as a micro-batch.*

### Practical 14: Streaming with Triggers and Output Modes
*Reference: Slides 48-49*

Advanced streaming configurations for controlled processing.

**Import trigger types:**
```scala
import org.apache.spark.sql.streaming.Trigger
```
*Enables processing time controls.*

**Create stream with processing time trigger:**
```scala
val query = wordCounts.writeStream
  .outputMode("complete")
  .trigger(Trigger.ProcessingTime("5 seconds"))
  .format("console")
  .start()
```
*Processes accumulated data every 5 seconds.*

**Use update output mode:**
```scala
val query = wordCounts.writeStream
  .outputMode("update")
  .trigger(Trigger.ProcessingTime("5 seconds"))
  .format("console")
  .start()
```
*Shows only changed counts, not complete table.*

**Stop streaming query:**
```scala
query.stop()
```
*Terminates the streaming job gracefully.*

---

## MareNostrum Execution

### Practical 15: Running Spark on MareNostrum Supercomputer
*Reference: Slides 54-57*

Complete workflow for executing Spark jobs on BSC's MareNostrum cluster.

**Connect to MareNostrum:**
```bash
ssh [USER]@glogin1.bsc.es
```
*Replace [USER] with your MareNostrum username.*

**Copy Spark container:**
```bash
cp /home/nct/nct00012/spark-mn5.tar.gz ./
```
*Copies pre-built Spark container from shared location.*

**Extract container:**
```bash
tar xzvf spark-mn5.tar.gz
```
*Unpacks container and associated scripts.*

**Set up passwordless SSH between nodes:**
```bash
ssh-keygen
```
*Generates SSH key pair (press Enter for defaults).*

**Add key to authorized hosts:**
```bash
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```
*Enables nodes to communicate without passwords.*

**Edit submission script:**
```bash
nano start-mnv.sh
```
*Or use vim/vi to edit the script.*

**Update user directories in script:**
```bash
HOME_DIR=/home/nct/nctXXXXX/spark_test
WORK_DIR=/gpfs/scratch/nct_324/nctXXXXX/spark_test
```
*Replace nctXXXXX with your actual user ID.*

**Check reservation (if applicable):**
```bash
#SBATCH --reservation=RT396165_Jul21
```
*Verify or update the reservation name for the training.*

**Submit job to queue:**
```bash
sbatch -A nct_324 -q gp_training start-mnv.sh
```
*Submits Spark job to the training queue.*

**Monitor job status:**
```bash
squeue
```
*Shows your jobs in the queue.*

**Cancel job if needed:**
```bash
scancel [JOB_ID]
```
*Stops a running or queued job.*

**Check results after completion:**
```bash
cd experiments_XXXXXXX/wc-result.data
ls -la
```
*Navigate to results directory to find output files.*

---

## Additional Tips and Troubleshooting

### Common Issues and Solutions

**Port already in use:**
- Kill existing Spark processes before starting new ones
- Use different port numbers in configuration

**Out of memory errors:**
- Increase worker memory allocation in SPARK_WORKER_ARGS
- Use `.cache()` judiciously on frequently accessed RDDs
- Consider increasing number of partitions

**Container not found:**
- Verify IMAGE path is correct
- Check Singularity module is loaded: `module load singularity`

**Network connectivity issues:**
- Ensure all nodes can communicate
- Check firewall settings
- Verify hostname resolution

### Best Practices

1. **Always clean up resources** after experiments
2. **Use appropriate data formats** (Parquet for analytics, CSV for data exchange)
3. **Cache intermediate results** when data is reused multiple times
4. **Monitor Spark UI** at http://[master-host]:8080 for job progress
5. **Start small** - test with subset of data before full-scale processing

---

**End of Complete Practical Guide**
