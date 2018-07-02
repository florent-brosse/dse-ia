# ML recommendation

This job uses the **Alternating Least Squares Method** for **Collaborative Filtering** 

From a product review table in DSE 6 we will predict the best products for each customer.

## To create data in DSE:

```
CREATE KEYSPACE IF NOT EXISTS  ecommerce WITH replication = {'class': 'SimpleStrategy' , 'replication_factor': 1 };

CREATE TABLE IF NOT exists ecommerce.review(
  product_id timeuuid,
  customer_id timeuuid,
  review_date timeuuid,
  title text,
  comment text,
  rating int,
  PRIMARY KEY((product_id),customer_id) // one review by customer by product is authorized
);

CREATE TABLE IF NOT exists ecommerce.prediction(
  product_id timeuuid,
  customer_id timeuuid,
  prediction double,
  PRIMARY KEY(customer_id,product_id)
);

COPY ecommerce.review FROM 'src/main/resources/review.csv' WITH HEADER = false;
```

**SparkMLProductRecommendationBatchJob** reads all data from the table ecommerce.review and fills the table ecommerce.prediction

We can get best predictions for an user with a simple select:

`select * from ecommerce.prediction where customer_id=382cccc2-7e00-11e8-9eb8-2b8a8f043a90 ;`

## Detail of the implementation

 Read the data from Cassandra
 ```
 val reviewDF = session.read.cassandraFormat("review", "ecommerce")
       .load()
       .select("product_id", "customer_id", "rating")
 ```
 
 Prepare features: we should transform UUID to int for ALS
 ```
  val indexerProduct = new StringIndexer()
       .setInputCol("product_id")
       .setOutputCol("product")
 
  val indexerCustomer = new StringIndexer()
       .setInputCol("customer_id")
       .setOutputCol("customer")
       
  val reviewDFWithProduct = indexerProduct.fit(reviewDF).transform(reviewDF)
  val preparedReviewDF = indexerCustomer.fit(reviewDFWithProduct).transform(reviewDFWithProduct).cache()
 ```
 
 Build the recommendation model using ALS on the training data
 ```
 val als = new ALS()
       .setMaxIter(5)
       .setRegParam(0.01)
       .setImplicitPrefs(true)
       .setUserCol("customer")
       .setItemCol("product")
       .setRatingCol("rating")
 
 val model = als.fit(preparedReviewDF)
 ```    
 
 Compute all products with all customers table without rank
 ```
 val product = preparedReviewDF.select("product").distinct()
 val customer = preparedReviewDF.select("customer").distinct()
 val crossJoin = product.join(customer)  // need config("spark.sql.crossJoin.enabled", "true")
 val data = crossJoin.except(preparedReviewDF.select("product", "customer"))  
```

 Find the corresponding ids 
 ```
 val converterProduct = new IndexToString()
       .setInputCol("product")
       .setOutputCol("product_id")
 
 val converterCustomer = new IndexToString()
       .setInputCol("customer")
       .setOutputCol("customer_id")
     
 val predictionsDF = converterCustomer.transform(converterProduct.transform(predictions)).select("product_id", "customer_id", "prediction")
 ```
 
 Keep only when the score is > 0.01
 ```
 predictionsDF.filter("prediction > 0.01").createOrReplaceTempView("predictions")
 ```
 
 Keep only 5 best results
 ```
 val response = session.sql(
       """SELECT product_id,customer_id,prediction FROM
                   (SELECT product_id,customer_id,prediction,dense_rank()
                   OVER (PARTITION BY customer_id ORDER BY prediction DESC) as rank
                   FROM predictions) tmp
                   WHERE rank <= 5""")
 ```
 
 Save predictions into Cassandra
 ```
 response.write.cassandraFormat("prediction", "ecommerce").option("confirm.truncate", true).mode(SaveMode.Overwrite).save()
 ```