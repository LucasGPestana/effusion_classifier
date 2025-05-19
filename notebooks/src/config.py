import os

# Caminho do diretório do projeto
PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Caminho do diretório dos dados
DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

# Caminho do diretório das imagens de efusão
EFFUSION_DIR = os.path.join(
    DATA_DIR,
    "effusion"
)

# Caminho do diretório das imagens de efusão
EFFUSION_DIR = os.path.join(
    DATA_DIR,
    "effusion"
)

# Caminho do diretório das imagens sem doença
NO_DISEASE_DIR = os.path.join(
    DATA_DIR,
    "no_finding"
)

# Caminho do diretório das imagens de efusão processadas
EFFUSION_PROCESSED_DIR = os.path.join(
    DATA_DIR,
    "effusion_processed"
)

# Caminho do diretório das imagens sem doença processadas
NO_DISEASE_PROCESSED_DIR = os.path.join(
    DATA_DIR,
    "no_finding_processed"
)

# Caminho do diretório de treino
TRAIN_DIR = os.path.join(
    DATA_DIR, 
    "train"
    )

# Caminho do diretório de teste
TEST_DIR = os.path.join(
    DATA_DIR, 
    "test"
    )

# Caminho para os metadados das imagens
METADATA_PATH = os.path.join(
    DATA_DIR,
    "Data_Entry_2017.csv"
)

# Caminho do diretório dos modelos
MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

# Caminho do diretório dos logs
LOGS_DIR = os.path.join(
    PROJECT_DIR,
    "logs"
)

# Random state para seed
RANDOM_STATE = 42