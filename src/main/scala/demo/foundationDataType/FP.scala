package demo.foundationDataType

import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.SparkSession

/**
  * Created by zhao-chj on 2018/10/26.
  */
object FP {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("SparkMlilb")
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import  spark.implicits._
    val dataset = spark.createDataset(Seq(
      "1 2 5",
      "1 2 3 5",
      "1 2")
    ).map(t => t.split(" ")).toDF("items")

    val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.5).setMinConfidence(0.6)
    val model = fpgrowth.fit(dataset)

    // Display frequent itemsets.
    model.freqItemsets.show()

    // Display generated association rules.
    model.associationRules.show()

    // transform examines the input items against all the association rules and summarize the
    // consequents as prediction
    model.transform(dataset).show()
  }
}
