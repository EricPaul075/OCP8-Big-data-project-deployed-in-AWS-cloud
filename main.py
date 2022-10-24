import boto3
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, PCA

from PIL import Image
import json
import pickle
import gc

from functions import *

# Configuration d'exécution
LOCAL = True  # False si exécution sur aws EMR
DEBUG = True  # Mode debug
print('\n\n')
print(f"Exécution du script en local={LOCAL} et debug={DEBUG}")

# Session spark
# Note: lancer l'application avec "spark-submit --conf spark.driver.maxResultSize=0 main.py"
# si besoin que le volume de données sérialisées (collectées) par le driver soit > 1GB (défaut)
# utiliser la valeur 0 pour illimité (erreur si mémoire insuffisante) ou une valeur (ex:2g) sinon
# La configuration s'ajuste également dans le fichier SPARK_HOME/conf/spark-defaults.conf
# Dans EMR, sous aws linux: sudo vim /etc/spark/conf/spark-defaults.conf
spark = SparkSession \
    .builder \
    .config('spark.driver.cores', '1') \
    .config('spark.driver.memory', '8g') \
    .config('spark.executor.instances', '2') \
    .config('spark.executor.cores', '4') \
    .config('spark.executor.memory', '1g') \
    .appName('ocp8') \
    .getOrCreate()
prev = spark.conf.get("spark.sql.execution.arrow.pyspark.enabled")  # get previous conf
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", True)  # Arrow optimization for pyspark.sql.DataFrame
conf = spark.sparkContext.getConf().getAll()
print('\n\n')
for item in conf:
    print(item)
sc = spark.sparkContext
sc.setLogLevel('ERROR')

# Structure de stockage du projet dans S3
# Nota: clé user AWS à spécifier dans SER/.aws/credentials.
AWS_S3_BUCKET = "ocp8project"
INPUT_DATA_FOLDER = "input_data/"
OUTPUT_DATA_FOLDER = "output_data/"

# Lecture des données d'entrées
s3_client = boto3.client("s3")
objects = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=INPUT_DATA_FOLDER)
n_objects = objects['KeyCount']
print(f"Nombre d'objets dans s3://{AWS_S3_BUCKET}/{INPUT_DATA_FOLDER}: {n_objects}")
if n_objects<=1:
    print("Erreur: le répertoire des données d'entrée est vide")

# Extraction des features de manière itérative, image par image
print('\n\n')
print("Lecture des images et extraction des features avec le CNN 'EfficientNetB0':")
labels = dict()
label_count = 0
img_count = 0
max_img = 10 if DEBUG else n_objects  # remplacer 'n_objects' par une constante selon la capacité du cluster
max_feat = 10 if DEBUG else 1280
model = get_model_from_EfficientNetB0()

for obj in objects['Contents']:
    label = obj['Key'].split('/')[-2]
    filename = obj['Key'].split('/')[-1]
    if filename!='':

        # Codage du label
        if label not in labels.keys():
            labels.update({label: label_count})
            label_count += 1

        # Lecture et redimensionnement de l'image
        file = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=obj['Key'])
        img = Image.open(file['Body'])
        print(f"{label}: {filename}, {img.format}, {img.size}, {img.mode}")
        if img.size != (224, 224):
            img = img.resize((224, 224))

        # labels et features des images
        sdf_label = spark.createDataFrame([(labels[label], )], ['label'])

        features = feature_extraction(model, img=img, debug=DEBUG, debug_feat_size=max_feat)
        sdf_features = spark.createDataFrame([(features, )], ['features'])
        if img_count==0:
            y = sdf_label
            X = sdf_features
            features_size = len(features)
        else:
            y = y.union(sdf_label)
            X = X.union(sdf_features)
        img_count += 1
    if img_count==max_img: break  # mode debug

# Écriture des dataframes et du dictionnaire des labels dans S3
print('\n\n')
print(f"Sauvegarde du dataset constitué de {img_count} images de {label_count}"
      f" fruits, vectorisées en {features_size} features")
labels_file = 'labels.json'
with open(labels_file, "w") as file:
    json.dump(labels, file)
print(f"Sauvegarde du dictionnaire des labels dans "
      f"s3://{AWS_S3_BUCKET}/{OUTPUT_DATA_FOLDER}{labels_file}")
s3_client.upload_file(labels_file, AWS_S3_BUCKET, OUTPUT_DATA_FOLDER+labels_file)

if not LOCAL:
    print(f"Sauvegarde du dataset-labels au format '.parquet' dans "
          f"s3://{AWS_S3_BUCKET}/{OUTPUT_DATA_FOLDER}y.parquet")
    y.write.mode('overwrite').parquet('s3://' + AWS_S3_BUCKET + '/' + OUTPUT_DATA_FOLDER + 'y.parquet')
    print(f"Sauvegarde du dataset-features au format '.parquet' dans "
          f"s3://{AWS_S3_BUCKET}/{OUTPUT_DATA_FOLDER}X.parquet")
    X.write.mode('overwrite').parquet('s3://' + AWS_S3_BUCKET + '/' + OUTPUT_DATA_FOLDER + 'X.parquet')

# Nettoyage mémoire
del y, img
gc.collect()

# Standardisation des features de X
print('\n\n')
print(f"Mise à l'échelle des features avec StandardScaler:")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures").fit(X)
X = scaler.transform(X).select('scaledFeatures')
print(f"Données X mises à l'échelle avec StandardScaler: X:{X.count()} lignes * {len(X.columns)} colonne")

# Sauvegarde de scaler et nettoyage mémoire
scaler_file = 'std_scaler'
if LOCAL:
    scaler.write().overwrite().save(scaler_file)
else:
    print(f"Sauvegarde du scaler dans s3://{AWS_S3_BUCKET}/{OUTPUT_DATA_FOLDER}{scaler_file}")
    scaler.write().overwrite().save('s3://' + AWS_S3_BUCKET + '/' + OUTPUT_DATA_FOLDER + scaler_file)
del scaler
gc.collect()

# Réduction de dimension de X avec PCA
n_features = max(2, max_feat//10) if DEBUG else features_size//10
print('\n\n')
print(f"Réduction de dimension de X avec PCA (k={n_features}):")
pca = PCA(k=n_features, inputCol='scaledFeatures', outputCol='pcaFeatures').fit(X)
X = pca.transform(X).select('pcaFeatures')
print(f"Dataset réduit (X):{X.count()} lignes * {len(X.columns)} colonne")

# Écriture de X dans S3
if not LOCAL:
    print(f"Sauvegarde du dataset-features réduit dans "
          f"s3://{AWS_S3_BUCKET}/{OUTPUT_DATA_FOLDER}Xpca.parquet")
    X.write.mode('overwrite').parquet('s3://' + AWS_S3_BUCKET + '/' + OUTPUT_DATA_FOLDER + 'Xpca.parquet')
del X
gc.collect()

# Variance expliquée
print(f"Variance expliquée totale: {100*pca.explainedVariance.sum():.2f}%")
explained_var = {'explained_var_vec': pca.explainedVariance,
                 'explained_var_vsum': pca.explainedVariance.sum()}
explained_var_file = 'explained_var.pkl'
with open(explained_var_file, 'wb') as file:
    pickle.dump(explained_var, file)
print(f"Sauvegarde de la variance expliquée dans s3://"
      f"{AWS_S3_BUCKET}/{OUTPUT_DATA_FOLDER}{explained_var_file}")
s3_client.upload_file(explained_var_file, AWS_S3_BUCKET, OUTPUT_DATA_FOLDER+explained_var_file)

# Sauvegarde PCA et nettoyage mémoire
pca_file = 'pca'
if LOCAL:
    pca.write().overwrite().save(pca_file)
else:
    print(f"Sauvegarde du PCA dans s3://{AWS_S3_BUCKET}/{OUTPUT_DATA_FOLDER}{pca_file}")
    pca.write().overwrite().save('s3://' + AWS_S3_BUCKET + '/' + OUTPUT_DATA_FOLDER + pca_file)
del pca
gc.collect()

# Reset spark configuration
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", prev)  # Restaure conf précédente
print("*** Fin de l'application ***")
