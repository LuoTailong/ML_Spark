package demo.mltest

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
/**
  * Created by zhao-chj on 2018/7/15.
  */
object FPGroupth_PatternMinging {
  def main(args: Array[String]): Unit = {

    val sparkconf = new SparkConf().setAppName("SparkGrowthDemo").setMaster("local[2]")
    val sc = new SparkContext(sparkconf)
    val data = sc.textFile("D:\\BigData\\Workspace\\Spark_Test\\src\\main\\scala\\cn\\apple\\mltest\\sample_fpgrowth.txt")

    val transactions: RDD[Array[String]] = data.map(s => s.trim.split(' '))

    val fpg = new FPGrowth()
      .setMinSupport(0.2)
      .setNumPartitions(10)
    val model = fpg.run(transactions)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }
    //    #前件-->后件  [q,t,y,z] => [x], 1.0
    val minConfidence = 0.8
    model.generateAssociationRules(minConfidence).collect().foreach { rule =>
      println(
        rule.antecedent.mkString("[", ",", "]")
          + " => " + rule.consequent .mkString("[", ",", "]")
          + ", " + rule.confidence)
    }
  }
}
