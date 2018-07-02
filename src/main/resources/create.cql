create KEYSPACE IF NOT EXISTS  ecommerce WITH replication = {'class': 'SimpleStrategy' , 'replication_factor': 1 };

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
