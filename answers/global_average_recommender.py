from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

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

global_avg = training.groupby().avg().collect()[0]['avg(rating)']
print(global_avg)
training = training.withColumn('global_avg', global_avg)
test = test.withColumn('global_avg', global_avg)

als = ALS(rank=70, maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="global_avg", 
              coldStartStrategy="drop")
als = als.setSeed(int(desired_seed))

model = als.fit(training)

predictions = model.transform(test)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="global_avg",
                            predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(str(rmse))
