import pandas as pd
import numpy as np
import cv2


import os
import glob
import random
import concurrent.futures
from typing import Iterable


from src.config import DATA_DIR, TRAIN_DIR, TEST_DIR

def getClassDataframe(filepath: str, class_label: str) -> pd.DataFrame:

    """Retorna um DataFrame contendo apenas instâncias de class_label ou 'No Finding', a partir de um arquivo csv.

    Objetivo da função é filtrar uma instância de interesse em aplicações de classificação binária

    Parameters
    ----------
    filepath : str
        Caminho do csv com os dados compatíveis com formato tabular
    
    Returns
    -------
    pandas.DataFrame
        DataFrame contendo instâncias de class_label e 'No Finding'
    """

    df = pd.read_csv(filepath)

    df.loc[df["Finding Labels"].str.lower().str.contains(class_label), "Finding Labels"] = class_label.title()

    df_classification = df[
        (df["Finding Labels"] == class_label) | 
        (df["Finding Labels"] == "No Finding")
        ]

    return df_classification[["Image Index", "Finding Labels"]]

def createDirectories(dirnames: Iterable[str], dirpath: str) -> None:

    """Cria diretórios em um diretório específico

    Parameters
    ----------
    classes : Iterable[str]
        Iterável contendo os nomes das classes
    dirpath : str
        Caminho do diretório em que os diretórios serão criados

    """
    for dirname in dirnames:

        dirname = dirname.lower().replace(" ", "_")

        if dirname not in os.listdir(dirpath):

            os.mkdir(
                os.path.join(
                    dirpath,
                    dirname,
                    )
                    )

def copyImagesOnDirectory(dataframe: pd.DataFrame, dirpath: str) -> None:

    """Copia as imagens correspondentes a cada classe no seu diretório.

    Passos:

        1. Cria os diretórios das classes em 'dirpath', caso não existam
        2. Busca as imagens no diretório 'dirpath'
        3. Copia as imagens para o diretório da classe correspondente


    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe contento as seguintes colunas:
            - Image Index: Nome da imagem
            - Finding Labels: Classe que representa
    """

    classes = dataframe["Finding Labels"].unique()

    # Cria os diretórios das classes apenas se não existirem
    createDirectories(classes, dirpath)

    for _, row in dataframe.iterrows():

        image_name = row["Image Index"]
        class_label = row["Finding Labels"].lower().replace(" ", "_")

        
        class_label_dirpath = os.path.join(
            dirpath,
            class_label
        )

        # Obtém o caminho relativo da imagem a partir de uma busca recursiva em 'dirpath'
        relpaths = glob.glob(
            pathname=f"images_*{os.sep}images{os.sep}{image_name}",
            root_dir=dirpath,
            recursive=True
            )
        
        # Caso não seja encontrado nenhuma imagem, já vai para a próxima iteração
        if not relpaths:

            continue

        abspath = os.path.join(DATA_DIR, relpaths[0])

        # O software de cópia depende do sistema operacional
        if os.sys.platform.lower().startswith("win"):

            os.system(f"copy '{abspath}' '{class_label_dirpath}'")
        
        else:

            os.system(f"cp '{abspath}' '{class_label_dirpath}'")

def __saveImageOnDirectory(image_path: str, dir_path: str) -> None:

    """Salva uma imagem em um diretório.



    Parameters
    ----------
    image_path : str
        Caminho da imagem
    dir_path : str
        Caminho do diretório em que a imagem será armazenada
    """

    img = cv2.imread(image_path)

    basename = os.path.basename(image_path)

    cv2.imwrite(
        os.path.join(
            dir_path,
            basename
        ),
        img
    )

def splitImagesInTrainAndTest(dir_path: str, test_size: float) -> None:

    """Separa as imagens de um diretório em treino e teste

    Parameters
    ----------
    dir_path : str
        Caminho do diretório contendo a imagem
    test_size : float
        Proporção das imagens para teste (test_size pertence a [0, 1])
    """

    images = np.array([os.path.join(
        dir_path,
        basename
    ) for basename in os.listdir(dir_path)])

    images_amount = images.shape[0]
    random.shuffle(images)

    test_amount = int(round(images_amount * test_size))

    createDirectories(["train", "test"], dir_path)

    test_images = images[:test_amount]
    train_images = images[test_amount:]

    # Nome da pasta mais interna do diretório passado, que espera-se ser uma classe
    basename = os.path.basename(dir_path).replace("_processed", "")

    # Criando os diretórios das classes dentro das pastas de treino e teste
    for context_set in (TRAIN_DIR, TEST_DIR):

        print(os.path.join(context_set, basename))

        if basename not in os.listdir(context_set):

            os.mkdir(
                os.path.join(
                    context_set, 
                    basename
                    )
                    )
        
    # Divide em unidades de execução para não sobrecarregar a memória
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

        executor.map(
            lambda x: __saveImageOnDirectory(x, os.path.join(
                TRAIN_DIR,
                basename
            )), 
            train_images
            )

        executor.map(
            lambda x: __saveImageOnDirectory(x, os.path.join(
                TEST_DIR,
                basename
            )), 
            test_images
            )

def doRandomUndersampling(source_path: str, target_path: str) -> None:

    """Faz um balanceamento aleatório das classes presentes em 'source_path', de modo a mover as imagens 'removidas' da classe majoritária para 'target_path'

    Parameters
    ----------
    source_path : str
        Caminho do diretório que se encontra as classes com as imagens
    target_path : str
        Caminho do diretório em que as imagens da classe majoritária serão movidas
    
    """

    classes = list(filter(lambda x: os.path.isdir(
        os.path.join(source_path, x)
    ), os.listdir(source_path)))

    classes_dirpaths = list(map(lambda x: os.path.join(
        source_path,
        x
    ), classes))

    samples_lenght = list(
        map(lambda x: len(os.listdir(x)), 
            classes_dirpaths
        )
        )

    max_index = np.argmax(samples_lenght)
    
    min_index = np.argmin(samples_lenght)

    image_paths = np.array(
        [os.path.join(
            classes_dirpaths[max_index],
            basename,
        ) for basename in os.listdir(classes_dirpaths[max_index])]
        )
    
    # Índices das imagens que serão movidas para outra pasta

    diff_samples = (
        samples_lenght[max_index] - samples_lenght[min_index]
    )

    drop_indexes = random.sample(
        range(0, len(image_paths)), 
        k=diff_samples
        )
    
    # O software de mover depende do sistema operacional
    for i in drop_indexes:

        if os.sys.platform.startswith("win"):

            os.system(f"MOVE {image_paths[i]} {target_path}")
        
        else:
            
            os.system(f"mv {image_paths[i]} {target_path}")
    
    

