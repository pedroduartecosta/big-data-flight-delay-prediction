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
    StructField("year", IntegerType, true),
    StructField("month", IntegerType, true),
    StructField("dayofMonth", IntegerType, true),
    StructField("dayOfWeeK", IntegerType, true),
    StructField("depTime", IntegerType, true),
    StructField("crsDepTime", IntegerType, true),
    StructField("arrTime", IntegerType, true),
    StructField("cRSArrTime", IntegerType, true),
    StructField("uniqueCarrier", StringType, true),
    StructField("flightNum", IntegerType, true),
    StructField("tailNum", StringType, true),
    StructField("actualElapsedTime", IntegerType, true),
    StructField("cRSElapsedTime", IntegerType, true),
    StructField("airTime", IntegerType, true),
    StructField("arrDelay", DoubleType, true),
    StructField("depDelay", DoubleType, true),
    StructField("origin", StringType, true),
    StructField("dest", StringType, true),
    StructField("distance", DoubleType, true),
    StructField("taxiIn", IntegerType, true),
    StructField("taxiOut", IntegerType, true),
    StructField("cancelled", IntegerType, true),
    StructField("cancellationCode", IntegerType, true),
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

    val data = rawData.drop("actualElapsedTime")
                .drop("arrTime")
                .drop("airTime")
                .drop("taxiIn")
                .drop("diverted")
                .drop("weatherDelay")
                .drop("nASDelay")
                .drop("securityDelay")
                .drop("lateAircraftDelay")

    val categoricalVariables = Array("uniqueCarrier", "tailNum", "origin", "dest")
    val categoricalIndexers = categoricalVariables
      .map(i => new StringIndexer().setInputCol(i).setOutputCol(i+"Index"))
    val categoricalEncoders = categoricalVariables
      .map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec"))

    
  }
}
