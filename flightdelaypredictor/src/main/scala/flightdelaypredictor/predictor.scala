package FlightDelayPredictor

import org.apache.spark._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object Flight {

  case class Flight(
      year: Integer,
      month: Integer,
      dayofMonth: Integer,
      dayOfWeeK: Integer,
      depTime: Integer,
      crsDepTime: Integer,
      arrTime: Integer,           //forbidden
      cRSArrTime: Integer,
      uniqueCarrier: String,
      flightNum: Integer,
      tailNum: String,
      actualElapsedTime: Integer, //forbidden
      cRSElapsedTime: Integer,
      airTime: Integer,           //forbidden
      arrDelay: Double,
      depDelay: Double,
      origin: String,
      dest: String,
      distance: Double,
      taxiIn: Integer,            //forbidden
      taxiOut: Integer,
      cancelled: Integer,
      cancellationCode: Integer,
      diverted: Integer,          //forbidden
      carrierDelay: Integer,
      weatherDelay: Integer,      //forbidden
      nASDelay: Integer,          //forbidden
      securityDelay: Integer,     //forbidden
      lateAircraftDelay: Integer  //forbidden
    ) 
    extends Serializable

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
    StructField("diverted", IntegerType, true),
    StructField("carrierDelay", IntegerType, true),
    StructField("weatherDelay", IntegerType, true),
    StructField("nASDelay", IntegerType, true),
    StructField("securityDelay", IntegerType, true),
    StructField("lateAircraftDelay", IntegerType, true)
  ))

  def main(args: Array[String]) {

    val spark: SparkSession = SparkSession.builder().appName("predictor").config("spark.master", "local").getOrCreate()

    import spark.implicits._

    

    // Read raw data from csv file
    //.option("treatEmptyValuesAsNulls", "true") - may be useful
    val dfUncleaned = spark.read.format("com.databricks.spark.csv").schema(schema).option("header", "true").load("/home/proton/Documents/UPM-BigData-Spark/flightdelaypredictor/data/2008.csv")
    
    // Cleaning the dataset from forbidden variables
    val ds = dfUncleaned
              .drop("actualElapsedTime")
              .drop("arrTime")
              .drop("airTime")
              .drop("taxiIn")
              .drop("diverted")
              .drop("weatherDelay")
              .drop("nASDelay")
              .drop("securityDelay")
              .drop("lateAircraftDelay")
    
    val first10 = ds.take(10)

    for (v <- first10) println(v)

    println("dataset :" + ds.count())
    
    spark.stop()
  }
}
