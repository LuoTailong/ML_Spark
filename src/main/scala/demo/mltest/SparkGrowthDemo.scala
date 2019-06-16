package demo.mltest

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhao-chj on 2018/6/29.
  */
object SparkGrowthDemo {
  def main(args: Array[String]): Unit = {
      val sparkconf = new SparkConf().setAppName("SparkGrowthDemo").setMaster("local[2]")
      val sc = new SparkContext(sparkconf)
      val data = sc.textFile("D:\\BigData\\Workspace\\Spark_Test\\src\\main\\scala\\cn\\apple\\mltest\\sample_fpgrowth.txt")
//    切分数据
      val transactions = data.map(s => s.trim.split(' '))
//    构建频繁项集
      val fpg=new FPGrowth()
      .setMinSupport(0.2)
      .setNumPartitions(10)
      val model = fpg.run(transactions)

      model.freqItemsets.collect().foreach{ itemset =>
        println(itemset.items.mkString("[",",","]")+", "+itemset.freq)
      }
//      根据置信度产生规则
    val minConfidence=0.8
    model.generateAssociationRules(minConfidence).collect().foreach{ rule =>
      println(
        rule.antecedent.mkString("[",",","]")
        + "=》"+rule.consequent.mkString("[",",","]")
        + "," +rule.confidence)
    }
  }
}
