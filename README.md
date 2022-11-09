# OCP8-Big-data-project-deployed-in-AWS-cloud
Define a big data architecture and deploy in the cloud a distributed application on an EMR cluster using AWS

Le projet consiste à mettre en œuvre une architecture et une application big data dans le but de construire un moteur de classification d'images de fruits.
Les données d'entrée sont des images de différents fruits pris sous différents angles.

Il s'agit de prendre en considération le volume important de données pour réaliser une application en pyspark à exécuter de manière distribuée dans le cloud sur une architecture big data.
L'architecture big data (concentrée sur le batch layer) est mise en place sur AWS en utilisant S3 pour le stockage des données (images d'entrée et dataset de sortie) et un cluster EMR pour l'exécution de l'application. Le cluster est constitué de 3 instances EC2 uniformes (m5.xlarge: 4 vcpu et 16 GB RAM): 1 nœud maitre (driver avec un processus) et 2 nœuds esclaves (8 processus d'exécution en parallèle). La gestion mémoire des instances est gérée de manière différenciée selon leur rôle.

L'application réalise l'acquisition des images et leur prétraitement comprenant l'extraction des features et la réduction de leur dimension.

L'extraction des features des images s'effectue avec le modèle CNN EfficientNetB0 (pré-entrainé sur ImageNet) pour lequel la couche de classification est remplacée par GlobalAveragePooling2D.

La réduction de dimension des features s'effectue avec un PCA après mise à l'échelle.

Points d'attentions dans le passage à l'échelle:
- Acquisition des images d'entrée par batch < 1000 images (limitation de boto3) ;
- Taille mémoire du driver à surveiller avec l'accroissement du nombre d'image (maximiser la mémoire disponible pour le driver) ;
- Diminution de la variance expliquée à surveiller avec l'accroissement du nombre d'image.