import numpy as np
import keras
import cv2


def imagePredict(model_path: str, X: np.ndarray) -> float:

    """Determina a probabilidade de uma imagem ser classificada como 1
    
    Parameters
    ----------
    model_path : str
        Caminho do modelo a ser utilizado para a predição
    X : np.ndarray
        Representação da imagem a ser classificada em um array 2D
    
    Returns
    -------
    float
        Probabilidade da imagem ser classificada como 1

    """

    # A classe 1 é a que representa a presença de derrame pleural

    model = keras.models.load_model(model_path)

    X = np.reshape(X, (X.shape[0], X.shape[1]))

    X = X.astype(np.uint8)

    # Aplicação de aumento de constraste por equalização de histograma
    X = cv2.equalizeHist(X)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Casting para float32 e normalização
    X = np.expand_dims(X, axis=0)
    X = X.astype(np.float32) / 255

    return model.predict(X).flatten()[0]