package demo.BigData9

import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.SparkSession
/**
  * Created by zhao-chj on 2018/10/17.
  */
object FPGrowthAlg {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("SparkMlilb")
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    // Load and parse the data file, converting it to a DataFrame.
    //val data = spark.read.format("libsvm").load("C:\\Users\\Administrator\\IdeaProjects\\ML_Spark\\src\\main\\scala\\demo\\BigData9\\sample_libsvm_data.txt")
    import spark.implicits._

    val dataset = spark.createDataset(Seq(
      "1 2 5",
      "1 2 3 5",
      "1 2")
    ).map(t => t.split(" ")).toDF("items")
    dataset.show()
//    +------------+
//    |       items|
//    +------------+
//    |   [1, 2, 5]|
//    |[1, 2, 3, 5]|
//    |      [1, 2]|
//    +------------+
    val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.5).setMinConfidence(0.6)
    val model = fpgrowth.fit(dataset)

    // Display frequent itemsets.
    model.freqItemsets.show()
//    +---------+----+
//    |    items|freq|
//    +---------+----+
//    |      [5]|   2|
//      |   [5, 1]|   2|
//      |[5, 1, 2]|   2|
//      |   [5, 2]|   2|
//      |      [2]|   3|
//      |      [1]|   3|
//      |   [1, 2]|   3|
//      +---------+----+
    // Display generated association rules.
    model.associationRules.show()

    21/5000
    // antecedent前件 -consequent后件
    // transform examines the input items against all the association rules and summarize the
    // consequents as prediction
    //transform根据所有关联规则检查输入项，并将结果汇总为预测
//    +----------+----------+------------------+
//    |antecedent|consequent|        confidence|
//    +----------+----------+------------------+
//    |       [2]|       [5]|0.6666666666666666|
//      |       [2]|       [1]|               1.0|
//      |    [5, 2]|       [1]|               1.0|
//      |    [1, 2]|       [5]|0.6666666666666666|
//      |    [5, 1]|       [2]|               1.0|
//      |       [5]|       [1]|               1.0|
//      |       [5]|       [2]|               1.0|
//      |       [1]|       [5]|0.6666666666666666|
//      |       [1]|       [2]|               1.0|
//      +----------+----------+------------------+
    model.transform(dataset).show()
//    +------------+----------+
//    |       items|prediction|
//    +------------+----------+
//    |   [1, 2, 5]|        []|
//      |[1, 2, 3, 5]|        []|
//      |      [1, 2]|       [5]|
//    +------------+----------+
//    为什么只有第3个有输出，是因为[1,2]--->[5]是规则，作为预测
  }
}
