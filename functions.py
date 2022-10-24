from tensorflow import keras
from keras.applications import EfficientNetB0
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

def get_model_from_EfficientNetB0(layer='top_activation'):
    """
    Contruit le modèle basé sur EfficientNetB0 jusqu'à la couche
        spécifiée, selon le niveau attendu des features à extraire,
        puis ajoute une couche GlobalAveragePooling2D pour prendre
        la valeur moyenne (de la matrice 7*7) de chaque feature.
    :param layer: str, nom de la dernière couche keras layer du
        modèle EfficientNetB0 à utiliser, par défaut la dernière
        couche avant celle de classification.
    :return: keras.models.Model, modèle produisant les features.
    """
    # Modèle de base EfficientNet sans la couche de classification
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    #print(base_model.summary())  # Donne les noms de chaque layer

    # Sélection du modèle jusqu'à une couche spécifique selon le niveau  attendu des features
    x = base_model.get_layer(layer).output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


from keras.utils import load_img, img_to_array
from keras.applications.efficientnet import preprocess_input
from pyspark.ml.linalg import Vectors

def feature_extraction(model, img_path=None, img=None, debug=False, debug_feat_size=10):
    """
    Extrait les features de l'image avec le modèle (1280 features avec
        EfficientNetB0).
    :param model: keras.models.Model, modèle produisant les features,
        issu de la fonction get_model_from_EfficientNetB0.
    :param img_path: str, chemin de l'image dont les features seront
        extraites par la fonction.
    :param img: PIL image au format (224, 224).
    :param debug: bool, mode debug, default=False.
    :param debug_feat_size: int, dimension du vecteur de sortie pour
        le mode debug, défault=10.
    :return: pyspark.ml.linalg.Vectors, vecteur 1D des features de
        l'image.
    """
    if img_path is not None:
        # Charge et redim filtre par défaut
        img = load_img(img_path, target_size=(224, 224), keep_aspect_ratio=True)
    elif img is None:
        print('Aucune image spécifiée en entrée')
        return None
    img = img_to_array(img)  # Conversion de l'image en np.array
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Reshape format CNN
    img = preprocess_input(img)  # Preprocessing efficientnet
    # Prédiction + reshape 1D + format liste + Vectorisation
    features = Vectors.dense(model.predict(img).ravel().tolist()[:debug_feat_size]) \
        if debug else Vectors.dense(model.predict(img).ravel().tolist())
    return features
