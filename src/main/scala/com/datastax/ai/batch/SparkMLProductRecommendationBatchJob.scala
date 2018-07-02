package com.datastax.ai.batch

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.cassandra._
import org.apache.spark.sql.{SaveMode, SparkSession}
import ch.qos.logback.classic.{Level, Logger}
import org.slf4j.LoggerFactory

object SparkMLProductRecommendationBatchJob {
  def main(args: Array[String]): Unit = {
    // val root: Logger = LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME).asInstanceOf[Logger]
    // root.setLevel(Level.INFO)

    val session = SparkSession.builder()
      .appName("SparkMLProductRecommendationBatchJob")
      .config("spark.sql.crossJoin.enabled", "true")
    //  .master("local[2]")
      .getOrCreate()


    val reviewDF = session.read.cassandraFormat("review", "ecommerce")
      .load()
      .select("product_id", "customer_id", "rating")


    // prepare features should transform UUID en int for ALS
    val indexerProduct = new StringIndexer()
      .setInputCol("product_id")
      .setOutputCol("product")

    // prepare features should transform UUID en int for ALS
    val indexerCustomer = new StringIndexer()
      .setInputCol("customer_id")
      .setOutputCol("customer")

    val reviewDFWithProduct = indexerProduct.fit(reviewDF).transform(reviewDF)
    val preparedReviewDF = indexerCustomer.fit(reviewDFWithProduct).transform(reviewDFWithProduct).cache()

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setImplicitPrefs(true)
      .setUserCol("customer")
      .setItemCol("product")
      .setRatingCol("rating")

    val model = als.fit(preparedReviewDF)


    // compute all products with all customers table without rank
    val product = preparedReviewDF.select("product").distinct()
    val customer = preparedReviewDF.select("customer").distinct()
    val crossJoin = product.join(customer)
    val data = crossJoin.except(preparedReviewDF.select("product", "customer"))

    //prediction
    val predictions = model.transform(data)

    // find ids
    val converterProduct = new IndexToString()
      .setInputCol("product")
      .setOutputCol("product_id")

    val converterCustomer = new IndexToString()
      .setInputCol("customer")
      .setOutputCol("customer_id")

    val predictionsDF = converterCustomer.transform(converterProduct.transform(predictions)).select("product_id", "customer_id", "prediction")

    //keep only when a score is > 0.01
    predictionsDF.filter("prediction > 0.01").createOrReplaceTempView("predictions")

    //keep only 5 bests results
    val response = session.sql(
      """SELECT product_id,customer_id,prediction FROM
                  (SELECT product_id,customer_id,prediction,dense_rank()
                  OVER (PARTITION BY customer_id ORDER BY prediction DESC) as rank
                  FROM predictions) tmp
                  WHERE rank <= 5""")

    response.write.cassandraFormat("prediction", "ecommerce").option("confirm.truncate", true).mode(SaveMode.Overwrite).save()

    session.stop()

  }
}