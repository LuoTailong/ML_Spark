package demo.BigData9

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

/**
  * Created by zhao-chj on 2018/10/16.
  */
object DescitionTree {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("SparkMlilb")
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    // Load and parse the data file, converting it to a DataFrame.
    // 将以LIBSVM格式存储的数据作为DataFrame加载
    val data = spark.read.format("libsvm").load("C:\\Users\\Administrator\\IdeaProjects\\ML_Spark\\src\\main\\scala\\demo\\BigData9\\sample_libsvm_data.txt")
    // 索引标签，将元数据添加到标签列
    // 适合整个数据集以包括索引中的所有标签
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // 自动识别分类功能并对其进行索引
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
      .fit(data)
    // 将数据拆分为训练集和测试集（30％用于测试）
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // 训练决策树模型
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
    // 将索引标签转换回原始标签
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    // 管道中的链索引器和树模型
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
    // Train model. This also runs the indexers.
    // 训练模型
    val model = pipeline.fit(trainingData)
    // 预测
    val predictions = model.transform(testData)
    // 选择要显示的示例行
    predictions.select("predictedLabel", "label", "features").show(5)
    // 选择（预测，真实标签）并计算测试误差
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)
  }
}



