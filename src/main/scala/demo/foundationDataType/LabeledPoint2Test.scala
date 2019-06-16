package demo.foundationDataType

/**
  * Created by zhao-chj on 2018/10/16.
  */


import org.apache.spark._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object LabeledPoint2Test {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("testLabeledPoint2") //建立本地环境变量
    val sc = new SparkContext(conf) //建立Spark处理
    val mu: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "D://a.txt") //读取文件
    mu.foreach(println) //打印内容(1.0,(3,[0,1,2],[2.0,3.0,5.0]))
  }
}