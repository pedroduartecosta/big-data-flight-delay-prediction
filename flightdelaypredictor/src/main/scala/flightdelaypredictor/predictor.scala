package FlightDelayPredictor

import org.apache.spark._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._

import scala.io._

object Flight {

  def main(args: Array[String]) {
    print("UPM Big Data Spark project by\n")
    print("Carolina Neves\nPedro Costa\n\n")
    print("Where is your dataset located? (provide full path on disk) \n")

    val dataPath = readLine()

    var mlTechnique: Int = 0

    while(mlTechnique != 1 && mlTechnique != 2){
      print("Which machine learning technique do you want to use? \n")
      print("[1] Linear Regression \n")
      print("[2] Random Forest Trees \n")
      mlTechnique = readInt()
    }

    //val dataPath = "/home/proton/Documents/UPM-BigData-Spark/flightdelaypredictor/data/2008.csv"
    
    val conf = new SparkConf().setAppName("predictor").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

        val rawData = sqlContext.read.format("com.databricks.spark.csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(dataPath)
                .withColumn("DelayOutputVar", col("ArrDelay").cast("double"))
                .withColumn("DepDelayDouble", col("DepDelay").cast("double"))
                .withColumn("TaxiOutDouble", col("TaxiOut").cast("double"))
                .cache()

    // remove forbidden variables
    val data2 = rawData
                .drop("ActualElapsedTime") // Forbidden
                .drop("ArrTime") // Forbidden
                .drop("AirTime") // Forbidden
                .drop("TaxiIn") // Forbidden
                .drop("Diverted") // Forbidden
                .drop("CarrierDelay") // Forbidden
                .drop("WeatherDelay") // Forbidden
                .drop("NASDelay") // Forbidden
                .drop("SecurityDelay") // Forbidden
                .drop("LateAircraftDelay") // Forbidden
                .drop("DepDelay") // Casted to double in a new variable called DepDelayDouble
                .drop("TaxiOut") // Casted to double in a new variable called TaxiOutDouble

    // remove cancelled flights
    val data3 = data2.filter("DelayOutputVar is not null")

    // Remove correlated variables
    val data4 = data3
                .drop("UniqueCarrier") // Always the same value
                .drop("CancellationCode") // Cancelled flights don't count
                .drop("DepTime") // Highly correlated to CRSDeptime
                .drop("CRSArrTime") // Highly correlated to CRSDeptime
                .drop("CRSElapsedTime") // Highly correlated to Distance

    // Remove uncorrelated variables to the lable arrDelay
    val data = data4
                .drop("Distance")
                .drop("FlightNum")
                .drop("CRSDepTime")
                .drop("Year")
                .drop("Month")
                .drop("DayofMonth")
                .drop("DayOfWeek")
    
    val categoricalVariables = Array("TailNum", "Origin", "Dest")
    val categoricalIndexers = categoricalVariables
      .map(i => new StringIndexer().setInputCol(i).setOutputCol(i+"Index").setHandleInvalid("skip"))
    val categoricalEncoders = categoricalVariables
      .map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec").setDropLast(false))

    // assemble all of our features into one vector which we will call "features". 
    // This will house all variables that will be input into our model.
    val assembler = new VectorAssembler()
                    .setInputCols(Array("TailNumVec", "OriginVec", "DestVec", "DepDelayDouble", "TaxiOutDouble"))
                    .setOutputCol("features")
                    .setHandleInvalid("skip")

    mlTechnique match {
      case 0 => 
        val lr = new LinearRegression()
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")
        val paramGrid = new ParamGridBuilder()
          .addGrid(lr.regParam, Array(0.1, 0.01))
          .addGrid(lr.fitIntercept)
          .addGrid(lr.elasticNetParam, Array(0.0, 1.0))
          .build()
        val steps:Array[org.apache.spark.ml.PipelineStage] = categoricalIndexers ++ categoricalEncoders ++ Array(assembler, lr)

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

        println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
        println("mean absolute error: " + 	rm.meanAbsoluteError)
        println("R Squared: " + rm.r2)                         
        println("Explained Variance: " + rm.explainedVariance + "\n")

      case _ =>
        val rf = new RandomForestRegressor()
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")

        val paramGrid = new ParamGridBuilder().build()

        val steps:Array[org.apache.spark.ml.PipelineStage] = categoricalIndexers ++ categoricalEncoders ++ Array(assembler, rf)

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

        println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
        println("mean absolute error: " + 	rm.meanAbsoluteError)
        println("R Squared: " + rm.r2)                         
        println("Explained Variance: " + rm.explainedVariance + "\n")

    }

    // First Run
    // println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError)) // 9.796740796107883
    // println("mean absolute error: " + 	rm.meanAbsoluteError) // 6.887088798296243
    // println("R Squared: " + rm.r2)                          // 0.9350795166082608
    // println("Explained Variance: " + rm.explainedVariance + "\n") // 1382.225087766879

    // Second Run
    // println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError)) // 9.85055218341565
    // println("mean absolute error: " + 	rm.meanAbsoluteError) // 6.891695864162034
    // println("R Squared: " + rm.r2)                          // 0.934662584301822
    // println("Explained Variance: " + rm.explainedVariance + "\n") // 1386.6828761967377


    //Third Run
    // println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError)) // 9.858289809687346
    // println("mean absolute error: " + 	rm.meanAbsoluteError) // 6.930385044621098
    // println("R Squared: " + rm.r2)                          // 0.9343839060145654
    // println("Explained Variance: " + rm.explainedVariance + "\n") // 1383.7301688723805


    sc.stop()
  }

}
