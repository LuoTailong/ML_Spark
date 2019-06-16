package demo.mltest

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.fpm.AssociationRules
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
/**
  * Created by zhao-chj on 2018/7/15.
  */
object AsscoiateRules {
  def main(args: Array[String]): Unit = {
    val sparkconf = new SparkConf().setAppName("SparkGrowthDemo").setMaster("local[2]")
    val sc = new SparkContext(sparkconf)

    val freqItemsets = sc.parallelize(Seq(
      new FreqItemset(Array("a"), 15L),
      new FreqItemset(Array("b"), 35L),
      new FreqItemset(Array("a", "b"), 12L)
    ))

    val ar = new AssociationRules()
      .setMinConfidence(0.8)
    val results = ar.run(freqItemsets)

    results.collect().foreach { rule =>
      println("[" + rule.antecedent.mkString(",")
        + "=>"
        + rule.consequent.mkString(",") + "]," + rule.confidence)
    }
  }
}
