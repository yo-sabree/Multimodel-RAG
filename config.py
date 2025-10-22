import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

for directory in [DATA_DIR, REPORTS_DIR, IMAGES_DIR, MODELS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-exp"

TEXT_INDEX_PATH = MODELS_DIR / "text_index.faiss"
IMAGE_INDEX_PATH = MODELS_DIR / "image_index.faiss"
TEXT_META_PATH = MODELS_DIR / "text_meta.npy"
IMAGE_META_PATH = MODELS_DIR / "image_meta.npy"
MEMORY_PATH = MODELS_DIR / "conversation_memory.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5
SIMILARITY_THRESHOLD = 0.3

TEXT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

MAX_CONTEXT_LENGTH = 12000
MAX_MEMORY_TURNS = 5