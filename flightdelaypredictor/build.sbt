import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.12.5",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "flightDelayPredictor",
    resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
    fork in run := true,
    connectInput in run := true,
    libraryDependencies += sparkCore,
    libraryDependencies += sparkSQL,
    libraryDependencies += sparkMLlib,
    libraryDependencies += breeze,
    libraryDependencies += breezeNatives,
    libraryDependencies += breezeViz

  )
