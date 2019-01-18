package FlightDelayPredictor

import org.apache.spark._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor, GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._

import scala.io._

object Flight {

  def main(args: Array[String]) {
    print("\n")
    print("UPM Big Data Spark project by\n")
    print("Carolina Neves\nPedro Costa\n")
    print("{carolina.mneves,p.duarte}@alumnos.upm.es\n\n")
    print("Where is your dataset located? (provide full path on disk) \n")

    val dataPath = readLine()

    var mlTechnique: Int = 0

    while(mlTechnique != 1 && mlTechnique != 2 && mlTechnique != 3){
      print("\n")
      print("Which machine learning technique do you want to use? \n")
      print("[1] Linear Regression \n")
      print("[2] Random Forest Trees \n")
      print("[3] Gradient-Boosted Trees \n")
      mlTechnique = readInt()
    }

    print("\n")
    print("Use categorical features? (yes/no) \n")
    val useCategorical = readBoolean()

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
                .drop("UniqueCarrier") // Always the same value // Remove correlated variables
                .drop("CancellationCode") // Cancelled flights don't count
                .drop("DepTime") // Highly correlated to CRSDeptime
                .drop("CRSArrTime") // Highly correlated to CRSDeptime
                .drop("CRSElapsedTime") // Highly correlated to Distance
                .drop("Distance") // Remove uncorrelated variables to the arrDelay
                .drop("FlightNum") // Remove uncorrelated variables to the arrDelay
                .drop("CRSDepTime") // Remove uncorrelated variables to the arrDelay
                .drop("Year") // Remove uncorrelated variables to the arrDelay
                .drop("Month") // Remove uncorrelated variables to the arrDelay
                .drop("DayofMonth") // Remove uncorrelated variables to the arrDelay
                .drop("DayOfWeek") // Remove uncorrelated variables to the arrDelay
                .drop("TailNum")

    // remove cancelled flights
    val data = data2.filter("DelayOutputVar is not null")
                
    // assemble all of our features into one vector which we will call "features". 
    // This will house all variables that will be input into our model.
    val assembler = if(useCategorical){
      new VectorAssembler()
        .setInputCols(Array("OriginVec", "DestVec", "DepDelayDouble", "TaxiOutDouble"))
        .setOutputCol("features")
        .setHandleInvalid("skip")
    }else{
      new VectorAssembler()
        .setInputCols(Array("DepDelayDouble", "TaxiOutDouble"))
        .setOutputCol("features")
        .setHandleInvalid("skip")
      
    }

    val categoricalVariables = if(useCategorical){
                                Array("Origin", "Dest")
                              }else{
                                null
                              }
   
    val categoricalIndexers = if(useCategorical){
                                categoricalVariables.map(i => new StringIndexer().setInputCol(i).setOutputCol(i+"Index").setHandleInvalid("skip"))
                              }else{
                                null
                              }
    val categoricalEncoders = if(useCategorical){
                                categoricalVariables.map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec").setDropLast(false))
                              }else{
                                null
                              }
    
    

    mlTechnique match {
      case 1 => 
        val lr = new LinearRegression()
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")
        val paramGrid = new ParamGridBuilder()
          .addGrid(lr.regParam, Array(0.1, 0.01))
          .addGrid(lr.fitIntercept)
          .addGrid(lr.elasticNetParam, Array(0.0, 1.0))
          .build()

        val steps:Array[org.apache.spark.ml.PipelineStage] = if(useCategorical){
                                                                categoricalIndexers ++ categoricalEncoders ++ Array(assembler, lr)
                                                              }else{
                                                                Array(assembler, lr)
                                                              }

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

      case 2 =>
        val rf = new RandomForestRegressor()
          .setNumTrees(10)
          .setMaxDepth(10)
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")


        val steps:Array[org.apache.spark.ml.PipelineStage] = if(useCategorical){
                                                                categoricalIndexers ++ categoricalEncoders ++ Array(assembler, rf)
                                                              }else{
                                                                Array(assembler, rf)
                                                              }

        val pipeline = new Pipeline().setStages(steps)

        val Array(training, test) = data.randomSplit(Array(0.70, 0.30), seed = 12345)

        val model = pipeline.fit(training)

        val holdout = model.transform(test).select("prediction", "DelayOutputVar")

        // have to do a type conversion for RegressionMetrics
        val rm = new RegressionMetrics(holdout.rdd.map(x =>
          (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

        println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
        println("mean absolute error: " + 	rm.meanAbsoluteError)
        println("R Squared: " + rm.r2)                         
        println("Explained Variance: " + rm.explainedVariance + "\n")
      
      case _ =>
        val gbt = new GBTRegressor()
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")
          .setMaxIter(10)

        val steps:Array[org.apache.spark.ml.PipelineStage] = if(useCategorical){
                                                                categoricalIndexers ++ categoricalEncoders ++ Array(assembler, gbt)
                                                              }else{
                                                                Array(assembler, gbt)
                                                              }

        val pipeline = new Pipeline().setStages(steps)


        val Array(training, test) = data.randomSplit(Array(0.70, 0.30), seed = 12345)

        val model = pipeline.fit(training)

        val holdout = model.transform(test).select("prediction", "DelayOutputVar")

        // have to do a type conversion for RegressionMetrics
        val rm = new RegressionMetrics(holdout.rdd.map(x =>
          (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

        println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
        println("mean absolute error: " + 	rm.meanAbsoluteError)
        println("R Squared: " + rm.r2)                         
        println("Explained Variance: " + rm.explainedVariance + "\n")

    }

    // Linaer Regression - All but forbidden features
    // println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError)) // 9.796740796107883
    // println("mean absolute error: " + 	rm.meanAbsoluteError) // 6.887088798296243
    // println("R Squared: " + rm.r2)                          // 0.9350795166082608
    // println("Explained Variance: " + rm.explainedVariance + "\n") // 1382.225087766879

    // Linaer Regression - All but forbidden features and feaures which are highly correleted
    // println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError)) // 9.85055218341565
    // println("mean absolute error: " + 	rm.meanAbsoluteError) // 6.891695864162034
    // println("R Squared: " + rm.r2)                          // 0.934662584301822
    // println("Explained Variance: " + rm.explainedVariance + "\n") // 1386.6828761967377


    // Linaer Regression - All but forbidden features and feaures which are highly correleted and feaures which are highly correleted to ArrDelay (but including categorical features)
    // println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError)) // 9.858289809687346
    // println("mean absolute error: " + 	rm.meanAbsoluteError) // 6.930385044621098
    // println("R Squared: " + rm.r2)                          // 0.9343839060145654
    // println("Explained Variance: " + rm.explainedVariance + "\n") // 1383.7301688723805

    // Linear Regression - Only with "DepDelayDouble", "TaxiOutDouble"
    // sqrt(MSE): 10.878899669612952
    // mean absolute error: 7.783394230068079
    // R Squared: 0.9200968268506162
    // Explained Variance: 1362.3369912919457

    // Linear Regression - Only with "OriginVec", "DestVec", "DepDelayDouble", "TaxiOutDouble"
    // sqrt(MSE): 9.961831227944037
    // mean absolute error: 7.02540163549372
    // R Squared: 0.9330003640476157
    // Explained Variance: 1381.39053839548

    // Random Forest with numTress=3 and depth=4 - Only with "DepDelayDouble", "TaxiOutDouble"
    // sqrt(MSE): 19.146152143133534
    // mean absolute error: 10.201299302807106
    // R Squared: 0.7525103195695625
    // Explained Variance: 1029.9235548446513

    // Random Forest with numTress=10 and depth=10 - Only with "DepDelayDouble", "TaxiOutDouble"
    // sqrt(MSE): 18.354454337931042
    // mean absolute error: 9.395005755067366
    // R Squared: 0.772554662114239
    // Explained Variance: 1106.8968864726519

    // Random Forest with numTress=50 and depth=15 - Only with "DepDelayDouble", "TaxiOutDouble"
    // sqrt(MSE): 18.433252813401392
    // mean absolute error: 9.40955686165754
    // R Squared: 0.7705975548931269
    // Explained Variance: 1068.258438445279

    // Random Forest with numTress=10 and depth=10 - Only with "OriginVec", "DestVec", "DepDelayDouble", "TaxiOutDouble"
    // sqrt(MSE): 18.266729012017432
    // mean absolute error: 9.602345117676546
    // R Squared: 0.7747236216385236
    // Explained Variance: 1077.442442272066


    // Gradient-Boosted Trees with maxIter=10 - Only with "DepDelayDouble", "TaxiOutDouble"
    // sqrt(MSE): 17.98327804019669
    // mean absolute error: 9.33902838221064
    // R Squared: 0.7816607563685062
    // Explained Variance: 1170.153933584219

    // Gradient-Boosted Trees with maxIter=10 - Only with "OriginVec", "DestVec", "DepDelayDouble", "TaxiOutDouble"
    // sqrt(MSE): 17.80732797194016
    // mean absolute error: 9.213998551801323
    // R Squared: 0.7859123580993281
    // Explained Variance: 1174.7455190081098


    sc.stop()
  }

}
