from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, upper
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("Military Aviation Accident Severity ML") \
    .enableHiveSupport() \
    .getOrCreate()

# Read the Hive managed table created in Objective 2
df = spark.sql("SELECT * FROM aviation_accidents")

# Standardize text fields for military aircraft filtering
df = df.withColumn("make_upper", upper(col("make"))) \
       .withColumn("model_upper", upper(col("model")))

# Military-related aircraft patterns and manufacturers
military_patterns = (
    "C-130|KC-135|C-17|C-5|F-16|F-15|F-22|A-10|B-52|"
    "UH-60|MH-60|CH-47|AH-64|P-3|E-3|T-38|T-6|"
    "BLACK HAWK|CHINOOK|HERCULES|STRATOTANKER"
)

military_df = df.filter(
    col("model_upper").rlike(military_patterns) |
    col("make_upper").rlike("LOCKHEED|BOEING|SIKORSKY|MCDONNELL|NORTHROP")
)

# Create severity label from fatal injury count
military_df = military_df.withColumn(
    "severity",
    when(col("fatal_injury_count") == 0, 0)
    .when(col("fatal_injury_count") <= 5, 1)
    .otherwise(2)
)

# Select features for ML
ml_df = military_df.select(
    "aircraft_damage",
    "weather_condition",
    "aircraft_category",
    "purpose_of_flight",
    "number_of_engines",
    "severity"
).dropna()

print("Military-related records:", military_df.count())
print("Records used for ML:", ml_df.count())

# Convert categorical fields to numeric indexes
damage_indexer = StringIndexer(inputCol="aircraft_damage", outputCol="damage_index", handleInvalid="keep")
weather_indexer = StringIndexer(inputCol="weather_condition", outputCol="weather_index", handleInvalid="keep")
category_indexer = StringIndexer(inputCol="aircraft_category", outputCol="category_index", handleInvalid="keep")
purpose_indexer = StringIndexer(inputCol="purpose_of_flight", outputCol="purpose_index", handleInvalid="keep")

# Assemble features
assembler = VectorAssembler(
    inputCols=[
        "damage_index",
        "weather_index",
        "category_index",
        "purpose_index",
        "number_of_engines"
    ],
    outputCol="features"
)

# Random Forest model
rf = RandomForestClassifier(
    labelCol="severity",
    featuresCol="features",
    numTrees=20,
    seed=42
)

pipeline = Pipeline(stages=[
    damage_indexer,
    weather_indexer,
    category_indexer,
    purpose_indexer,
    assembler,
    rf
])

# Train/test split
train_data, test_data = ml_df.randomSplit([0.7, 0.3], seed=42)

model = pipeline.fit(train_data)
predictions = model.transform(test_data)

# Evaluation
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="severity",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="severity",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = accuracy_eval.evaluate(predictions)
f1 = f1_eval.evaluate(predictions)

print("Training completed successfully.")
print("Training records:", train_data.count())
print("Testing records:", test_data.count())
print("Accuracy:", accuracy)
print("F1 Score:", f1)

predictions.select(
    "aircraft_damage",
    "weather_condition",
    "aircraft_category",
    "purpose_of_flight",
    "number_of_engines",
    "severity",
    "prediction"
).show(20, truncate=False)

spark.stop()
