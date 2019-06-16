package demo.BigData9

/**
  * Created by zhao-chj on 2018/10/16.
  */
  import org.apache.spark.ml.feature.StringIndexer
  import org.apache.spark.sql.SparkSession
object StringIndex {
    def main(args: Array[String]): Unit = {
      val spark: SparkSession = SparkSession.builder()
        .appName("SparkMlilb")
        .master("local[2]")
        .getOrCreate()
      spark.sparkContext.setLogLevel("WARN")
      val df = spark.createDataFrame(
        Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
      ).toDF("id", "category")

      val indexer = new StringIndexer()
        .setInputCol("category")
        .setOutputCol("categoryIndex")

      val indexed = indexer.fit(df).transform(df)
      indexed.show()
    }
  }