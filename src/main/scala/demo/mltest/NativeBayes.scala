package demo.mltest

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
/**
  * Created by zhao-chj on 2018/7/15.
  * http://spark.apache.org/docs/2.0.2/api/scala/index.html#org.apache.spark.ml.classification.NaiveBayes
  */
object NativeBayes {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("SparkMlilb")
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("D:\\BigData\\Workspace\\Spark_Test\\src\\main\\scala\\cn\\apple\\mltest\\sample_libsvm_data.txt")

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

    // Train a NaiveBayes model.
    val model = new NaiveBayes()
      .fit(trainingData)

    // Select example rows to display.
    val predictions = model.transform(testData)
    predictions.show()
    predictions.select("recommendations").show()

    predictions.registerTempTable("t") //creatR
    println("888888888888888888888888")
    spark.sql("select * from t").show()
    //predictions.select("features").show()

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy: " + accuracy)
  }
}

//+-----+--------------------+--------------------+-----------+----------+
//|label|            features|       rawPrediction|probability|prediction|
//+-----+--------------------+--------------------+-----------+----------+
//|  0.0|(692,[95,96,97,12...|[-173678.60946628...|  [1.0,0.0]|       0.0|
//|  0.0|(692,[98,99,100,1...|[-178107.24302988...|  [1.0,0.0]|       0.0|
//|  0.0|(692,[100,101,102...|[-100020.80519087...|  [1.0,0.0]|       0.0|
//|  0.0|(692,[124,125,126...|[-183521.85526462...|  [1.0,0.0]|       0.0|
//|  0.0|(692,[127,128,129...|[-183004.12461660...|  [1.0,0.0]|       0.0|
//|  0.0|(692,[128,129,130...|[-246722.96394714...|  [1.0,0.0]|       0.0|
//|  0.0|(692,[152,153,154...|[-208696.01108598...|  [1.0,0.0]|       0.0|
//|  0.0|(692,[153,154,155...|[-261509.59951302...|  [1.0,0.0]|       0.0|
//|  0.0|(692,[154,155,156...|[-217654.71748256...|  [1.0,0.0]|       0.0|
//|  0.0|(692,[181,182,183...|[-155287.07585335...|  [1.0,0.0]|       0.0|
//|  1.0|(692,[99,100,101,...|[-145981.83877498...|  [0.0,1.0]|       1.0|
//|  1.0|(692,[100,101,102...|[-147685.13694275...|  [0.0,1.0]|       1.0|
//|  1.0|(692,[123,124,125...|[-139521.98499849...|  [0.0,1.0]|       1.0|
//|  1.0|(692,[124,125,126...|[-129375.46702012...|  [0.0,1.0]|       1.0|
//|  1.0|(692,[126,127,128...|[-145809.08230799...|  [0.0,1.0]|       1.0|
//|  1.0|(692,[127,128,129...|[-132670.15737290...|  [0.0,1.0]|       1.0|
//|  1.0|(692,[128,129,130...|[-100206.72054749...|  [0.0,1.0]|       1.0|
//|  1.0|(692,[129,130,131...|[-129639.09694930...|  [0.0,1.0]|       1.0|
//|  1.0|(692,[129,130,131...|[-143628.65574273...|  [0.0,1.0]|       1.0|
//|  1.0|(692,[129,130,131...|[-129238.74023248...|  [0.0,1.0]|       1.0|
//+-----+--------------------+--------------------+-----------+----------+
//only showing top 20 rows
