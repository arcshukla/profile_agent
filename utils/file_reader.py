
import os
import re
import pandas as pd
import fitz  # pymupdf
from utils.logger import get_logger

logger = get_logger(__name__)
#-----------------------
# Public method
# ----------------------
def read_file(path) -> str:
    path = str(path)
    content = ""
    if path and os.path.exists(path):
        if any(x in path for x in [".txt", ".css",".js","html"]):        
            return _read_txt_file(path)
        if ( ".pdf" in path):
            return _read_pdf_file(path)
        if ( ".csv" in path):
            return _read_csv_file(path)    
        logger.error("File " + path + " not supported.")
    logger.error("File " + path + " does not exists.")
    return content

#-----------------------
# Internal methods
# ----------------------
def _read_csv_file(path) -> str:
    df = pd.read_csv(path)
    content = "\n\n".join(
        df["Text"].dropna().astype(str)
    )
    return content

def _read_txt_file(path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _read_pdf_file(path) -> str:    
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text("text")  # "text" mode preserves proper spacing
        if text.strip():
            pages.append(text)
    return "\n".join(pages)

def _fix_pdf_spacing(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append('')
            continue
        
        tokens = stripped.split()
        total = len(tokens)
        single_char_count = sum(1 for t in tokens if len(t) == 1)
        
        # If more than 50% of tokens are single chars → spaced-out line
        if total > 2 and single_char_count / total > 0.5:
            cleaned.append(''.join(tokens))
        else:
            # Normal line — just collapse multiple spaces to one
            cleaned.append(re.sub(r'  +', ' ', stripped))
    
    result = '\n'.join(cleaned)
    
    # Final pass: fix "word  word" double-space between normal words
    result = re.sub(r'  +', ' ', result)
    
    return result