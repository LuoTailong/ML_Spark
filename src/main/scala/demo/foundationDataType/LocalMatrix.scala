package demo.foundationDataType

/**
  * Created by zhao-chj on 2018/10/16.
  */
object LocalMatrix {
  def main(args: Array[String]) {
    import org.apache.spark.mllib.linalg.{Matrix, Matrices}
    //记住，MLlib中的局部矩阵以列主要顺序存储.
    // Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))
    println(dm)
    println(dm(2,0))
    // Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
    val sm: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))
    print(sm)
    println(sm(2,1))


  }
}
