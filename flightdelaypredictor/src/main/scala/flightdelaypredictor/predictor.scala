package FlightDelayPredictor

import org.apache.spark._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.regression.{LinearRegression}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._


object Flight {

  val schema = StructType(Array(
    StructField("year", DoubleType, true),
    StructField("month", DoubleType, true),
    StructField("dayofMonth", DoubleType, true),
    StructField("dayOfWeeK", DoubleType, true),
    StructField("depTime", DoubleType, true),
    StructField("crsDepTime", DoubleType, true),
    StructField("arrTime", DoubleType, true),
    StructField("cRSArrTime", DoubleType, true),
    StructField("uniqueCarrier", StringType, true),
    StructField("flightNum", DoubleType, true),
    StructField("tailNum", StringType, true),
    StructField("actualElapsedTime", DoubleType, true),
    StructField("cRSElapsedTime", DoubleType, true),
    StructField("airTime", DoubleType, true),
    StructField("arrDelay", DoubleType, true),
    StructField("depDelay", DoubleType, true),
    StructField("origin", StringType, true),
    StructField("dest", StringType, true),
    StructField("distance", DoubleType, true),
    StructField("taxiIn", DoubleType, true),
    StructField("taxiOut", DoubleType, true),
    StructField("cancelled", DoubleType, true),
    StructField("cancellationCode", DoubleType, true),
    StructField("diverted", StringType, true),
    StructField("carrierDelay", StringType, true),
    StructField("weatherDelay", StringType, true),
    StructField("nASDelay", StringType, true),
    StructField("securityDelay", StringType, true),
    StructField("lateAircraftDelay", StringType, true)
  ))

  def main(args: Array[String]) {

    val dataPath = "/home/proton/Documents/UPM-BigData-Spark/flightdelaypredictor/data/2008short.csv"
    
    val conf = new SparkConf().setAppName("predictor").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val rawData = sqlContext.read.format("com.databricks.spark.csv")
                .option("header", "true")
                .option("inferSchema", "false")
                .schema(schema)
                .load(dataPath)
                .withColumn("delayOutputVar", col("ArrDelay").cast("double"))
                .cache()

    // remove forbidden variables
    val data2 = rawData
                .drop("actualElapsedTime")
                .drop("arrTime")
                .drop("airTime")
                .drop("taxiIn")
                .drop("diverted")
                .drop("weatherDelay")
                .drop("nASDelay")
                .drop("securityDelay")
                .drop("lateAircraftDelay")
                .drop("uniqueCarrier")
    
    // remove cancelled flights
    val data = data2.filter(col("Cancelled") > 0)

    val categoricalVariables = Array("tailNum", "origin", "dest")
    val categoricalIndexers = categoricalVariables
      .map(i => new StringIndexer().setInputCol(i).setOutputCol(i+"Index"))
    val categoricalEncoders = categoricalVariables
      .map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec"))

    // assemble all of our features into one vector which we will call "features". 
    // This will house all variables that will be input into our model.
    val assembler = new VectorAssembler()
                    .setInputCols(Array("tailNumVec", "originVec", "destVec", "year", "month", "dayofMonth", "dayOfWeeK", "depTime", "crsDepTime", "cRSArrTime", "flightNum", "cRSElapsedTime", "depDelay", "distance", "taxiOut"))
                    .setOutputCol("features")


    val lr = new LinearRegression()
      .setLabelCol("delayOutputVar")
      .setFeaturesCol("features")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 1.0))
      .build()

    val steps: Array[org.apache.spark.ml.PipelineStage] = categoricalIndexers ++ categoricalEncoders ++ Array(assembler, lr)
    val pipeline = new Pipeline().setStages(steps)

    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline) // the estimator can also just be an individual model rather than a pipeline
      .setEvaluator(new RegressionEvaluator().setLabelCol("delayOutputVar"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val Array(training, test) = data.randomSplit(Array(0.70, 0.30), seed = 12345)

    val model = tvs.fit(training)


    val holdout = model.transform(test).select("prediction", "delayOutputVar")

    // have to do a type conversion for RegressionMetrics
    val rm = new RegressionMetrics(holdout.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")


    sc.stop()

  }
}
