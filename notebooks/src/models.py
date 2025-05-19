import numpy as np
import keras


def getProbability(model: keras.Sequential, X: np.ndarray) -> float:

    """Determina a probabilidade de uma imagem ser classificada como 1
    
    Parameters
    ----------
    model : keras.Sequential
        Modelo a ser utilizado para a predição
    X : np.ndarray
        Representação da imagem a ser classificada em um array 2D

    """

    # A classe 1 é a que representa a presença de derrame pleural

    return model.predict(X).reshape(-1)[0]