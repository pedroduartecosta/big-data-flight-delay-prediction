import sbt._

object Dependencies {
  // https://mvnrepository.com/artifact/org.apache.spark/spark-core
  lazy val sparkCore = "org.apache.spark" %% "spark-core" % "2.4.0"
  // https://mvnrepository.com/artifact/org.apache.spark/spark-sql
  lazy val sparkSQL = "org.apache.spark" %% "spark-sql" % "2.4.0"
  // https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
  lazy val sparkMLlib = "org.apache.spark" %% "spark-mllib" % "2.4.0"
  lazy val breeze = "org.scalanlp" %% "breeze" % "0.13.2"
  lazy val breezeNatives = "org.scalanlp" %% "breeze-natives" % "0.13.2"
  lazy val breezeViz = "org.scalanlp" %% "breeze-viz" % "0.13.2"
}
