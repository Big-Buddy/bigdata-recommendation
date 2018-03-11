from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit

import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("LAB2")\
        .getOrCreate()

desired_seed = sys.argv[1]
desired_rows = sys.argv[2]

lines = spark.read.text("data/sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2], seed=desired_seed)

global_mean = training.groupby().avg('rating').collect()

user_mean_df = training.groupby('userId').agg({'rating' : 'avg'})
item_mean_df = training.groupby('movieId').agg({'rating' : 'avg'})

training = training.join(item_mean_df, ['movieId'])
training = training.join(user_mean_df, ['userId'])
test = test.join(item_mean_df, ['movieId'])
test = test.join(user_mean_df, ['userId'])

training.orderBy('userId', 'movieId').show(desired_seed)

als = ALS(rank=70, maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
als = als.setSeed(int(desired_seed))
model = als.fit(training)

final_df = model.transform(test)


### Order dataframe and show(desired_rows)