import sbt._

object Dependencies {
  // https://mvnrepository.com/artifact/org.apache.spark/spark-core
  lazy val sparkCore = "org.apache.spark" %% "spark-core" % "2.4.0"
  // https://mvnrepository.com/artifact/org.apache.spark/spark-sql
  lazy val sparkSQL = "org.apache.spark" %% "spark-sql" % "2.4.0"
  // https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
  lazy val sparkMLlib = "org.apache.spark" %% "spark-mllib" % "2.4.0"
}