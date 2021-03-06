from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from pyspark.sql.functions import lit

import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("LAB2")\
        .getOrCreate()

desired_seed = sys.argv[1]

lines = spark.read.text("data/sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2], seed=desired_seed)

global_mean = training.groupby().avg('rating').collect()[0]['avg(rating)']
user_mean_df = training.groupby('userId').agg({'rating' : 'avg'}).withColumnRenamed('avg(rating)', 'usermean')
item_mean_df = training.groupby('movieId').agg({'rating' : 'avg'}).withColumnRenamed('avg(rating)', 'itemmean')

training = training.join(user_mean_df, ['userId'])
training = training.join(item_mean_df, ['movieId'])
test = test.join(user_mean_df, ['userId'])
test = test.join(item_mean_df, ['movieId'])

reordered_training = training.withColumn('globalmean', lit(global_mean))
reordered_test = test.withColumn('globalmean', lit(global_mean))

reordered_training = reordered_training.withColumn('user-item-interaction', reordered_training.rating-(reordered_training.usermean+reordered_training.itemmean-reordered_training.globalmean))
reordered_test = reordered_test.withColumn('user-item-interaction', reordered_test.rating-(reordered_test.usermean+reordered_test.itemmean-reordered_test.globalmean))

als = ALS(rank=70, maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="user-item-interaction",
              coldStartStrategy="drop")
als = als.setSeed(int(desired_seed))
model = als.fit(reordered_training)

predictions = model.transform(reordered_test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="user-item-interaction",
                            predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(str(rmse))