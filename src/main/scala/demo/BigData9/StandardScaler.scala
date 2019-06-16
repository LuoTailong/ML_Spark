package demo.BigData9

/**
  * Created by zhao-chj on 2018/10/16.
  */
import org.apache.spark.sql.SparkSession

/**
  * Created by zhao-chj on 2018/8/13.
  */
object StandardScaler {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.ml.feature.StandardScaler
    val spark: SparkSession = SparkSession.builder()
      .appName("SparkMlilb")
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val dataFrame = spark.read.format("libsvm").load("C:\\Users\\Administrator\\IdeaProjects\\ML_Spark\\src\\main\\scala\\demo\\BigData9\\sample_libsvm_data.txt")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(dataFrame)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()
  }
}
