import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plotConfusionMatrix(matrix: np.ndarray[int]) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(12, 12))

    sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm",
        ax=ax
        )
    
    plt.show()

    return fig
