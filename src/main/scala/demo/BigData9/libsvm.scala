package demo.BigData9

/**
  * Created by zhao-chj on 2018/10/16.
  */

import org.apache.spark._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object libsvm {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("testLabeledPoint2") //建立本地环境变量
    val sc = new SparkContext(conf) //建立Spark处理
    val mu: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "C:\\Users\\Administrator\\IdeaProjects\\ML_Spark\\src\\a.txt") //读取文件
    mu.foreach(println) //打印内容(1.0,(3,[0,1,2],[2.0,3.0,5.0]))
  }
}
//2 1:5 2:8 3:9
//(2.0,(9,[0,1,2],[5.0,8.0,9.0]))
//1 1:7 2:6 3:7
//1 1:3 2:2 3:1
//2 1:5 2:8 3:9
//1 1:7 2:6 9:7
//(1.0,(9,[0,1,8],[7.0,6.0,7.0]))
//1 1:3 2:2 3:1
//2 1:5 2:8 3:9
//1 1:7 2:6 7:7
//1 1:3 2:2 4:1