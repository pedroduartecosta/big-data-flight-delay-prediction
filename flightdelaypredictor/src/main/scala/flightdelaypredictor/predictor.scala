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

import scala.io._

import breeze.linalg._
import breeze.numerics._
import breeze.plot._

object Flight {

  def main(args: Array[String]) {
    print("UPM Big Data Spark project by\n")
    print("Carolina Neves\nPedro Costa\n\n")
    print("Where is your dataset located? (provide full path on disk) \n")

    //val dataPath = readLine()

    val dataPath = "/home/proton/Documents/UPM-BigData-Spark/flightdelaypredictor/data/2008short.csv"
    
    val conf = new SparkConf().setAppName("predictor").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val rawData = sqlContext.read.format("com.databricks.spark.csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(dataPath)
                .withColumn("DelayOutputVar", col("ArrDelay").cast("double"))
                .withColumn("DepTimeDouble", col("DepTime").cast("double"))
                .withColumn("CRSElapsedTimeDouble", col("CRSElapsedTime").cast("double"))
                .withColumn("DepDelayDouble", col("DepDelay").cast("double"))
                .withColumn("TaxiOutDouble", col("TaxiOut").cast("double"))
                .cache()

    // remove forbidden variables
    val data2 = rawData
                .drop("ActualElapsedTime")
                .drop("ArrTime")
                .drop("AirTime")
                .drop("TaxiIn")
                .drop("Diverted")
                .drop("CarrierDelay")
                .drop("WeatherDelay")
                .drop("NASDelay")
                .drop("SecurityDelay")
                .drop("LateAircraftDelay")
                .drop("UniqueCarrier")
                .drop("CancellationCode")
                .drop("DepTime")
                .drop("CRSElapsedTime")
                .drop("DepDelay")
                .drop("TaxiOut")

    // remove cancelled flights
    val data = data2.filter("DelayOutputVar is not null")

    val fligh_delays = DenseVector(Source.fromFile(dataPath).getLines.drop(1).filter(x => x == "NA").map(_.split(",")(14).toDouble).toSeq :_ * )
    val number_of_flights = DenseVector.range(0,fligh_delays.length, 1).map( _.toDouble)


    println(":::::::::::::::::::::::::::")
    println(fligh_delays)

    val fig = Figure()
    val plt = fig.subplot(0)
    plt += plot(number_of_flights, fligh_delays)
    fig.refresh()

    
    val categoricalVariables = Array("TailNum", "Origin", "Dest")
    val categoricalIndexers = categoricalVariables
      .map(i => new StringIndexer().setInputCol(i).setOutputCol(i+"Index").setHandleInvalid("skip"))
    val categoricalEncoders = categoricalVariables
      .map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec").setDropLast(false))

    // assemble all of our features into one vector which we will call "features". 
    // This will house all variables that will be input into our model.
    val assembler = new VectorAssembler()
                    .setInputCols(Array("TailNumVec", "OriginVec", "DestVec", "Year", "Month", "DayofMonth", "DayOfWeek", "DepTimeDouble", "CRSDepTime", "CRSArrTime", "FlightNum", "CRSElapsedTimeDouble", "DepDelayDouble", "Distance", "TaxiOutDouble"))
                    .setOutputCol("features")
                    .setHandleInvalid("skip")


    val lr = new LinearRegression()
      .setLabelCol("DelayOutputVar")
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
      .setEvaluator(new RegressionEvaluator().setLabelCol("DelayOutputVar"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val Array(training, test) = data.randomSplit(Array(0.70, 0.30), seed = 12345)

    val model = tvs.fit(training)

    val holdout = model.transform(test).select("prediction", "DelayOutputVar")

    // have to do a type conversion for RegressionMetrics
    val rm = new RegressionMetrics(holdout.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError)) // 9.27208480408947
    println("mean absolute error: " + 	rm.meanAbsoluteError)
    println("R Squared: " + rm.r2)                          // 0.9418762043860976
    println("Explained Variance: " + rm.explainedVariance + "\n") //1391.913765231291


    sc.stop()

  }
}
