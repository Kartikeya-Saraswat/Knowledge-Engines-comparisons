"""Use the repository shared PDF chunker."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from retrieval_shared.pdf_chunking import *
from retrieval_shared.pdf_chunking import _is_heading
