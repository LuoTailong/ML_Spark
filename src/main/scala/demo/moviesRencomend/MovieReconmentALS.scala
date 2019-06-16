package demo.moviesRencomend

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
object MovieReconmentALS {
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder()
      .appName("SparkMlilb")
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val ratings = spark.read.textFile("C:\\Users\\Administrator\\IdeaProjects\\ML_Spark\\src\\main\\scala\\demo\\mltest\\sample_movielens_ratings.txt")
      .map(x=>parseRating(x))
      .toDF()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data
    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")
    // Generate top 10 movie recommendations for each user
    val userRecs = model.recommendForAllUsers(10)
    userRecs.show()
    // Generate top 10 user recommendations for each movie
    val movieRecs = model.recommendForAllItems(10)
    movieRecs.show()
    // Generate top 10 movie recommendations for a specified set of users
    val users = ratings.select(als.getUserCol).distinct().limit(3)
    val userSubsetRecs = model.recommendForUserSubset(users, 10)
    userSubsetRecs.show()

    // Generate top 10 user recommendations for a specified set of movies
    val movies = ratings.select(als.getItemCol).distinct().limit(3)
    val movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    movieSubSetRecs.show()
    movieSubSetRecs.select("recommendations").show()

    movieSubSetRecs.createOrReplaceTempView("t") //creatR
    spark.sql("select * from t").show()
    println("888888888888888888888888")
    spark.sql("select * from t").foreach(x=>println(x)) //获取可迭代对象中每一个元素
  }
}
