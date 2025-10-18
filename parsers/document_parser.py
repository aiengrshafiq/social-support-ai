# parsers/document_parser.py
import json
import base64
import os
from io import BytesIO
from typing import List, Dict, Any, Optional
import pdfplumber
from PIL import Image
import pandas as pd
import ollama  # Uses the direct Ollama client
from apps.api.core.config import settings # Import our settings

# --- This is the JSON schema we want the LLM to fill ---
SCHEMA_HINT = """
Return ONLY JSON. Do not include any other text, greetings, or markdown.
The JSON object must have these keys:
- "full_name": string or null
- "address": string or null
- "total_income": number or null  # monthly income if visible, else null
- "employer": string or null
- "id_number": string or null
- "family_members": [ { "name": string, "relation": string } ] or []

If a value is unknown, use null or [].
"""

VLM_MODEL = "qwen2-vl:7b"
TEXT_MODEL = "llama3.1:8b"
OLLAMA_HOST = settings.OLLAMA_HOST


def _image_to_b64(img: Image.Image) -> str:
    """Converts a PIL Image to a base64 string for the API."""
    buf = BytesIO()
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _file_to_b64_images(path: str, max_pages: int = 3) -> List[str]:
    """Rasterize first N pages of PDF or load image; returns list of base64 images."""
    images = []
    try:
        if path.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages[:max_pages]):
                    pil_img = page.to_image(resolution=150).original
                    images.append(_image_to_b64(pil_img))
        else:
            img = Image.open(path)
            images.append(_image_to_b64(img))
    except Exception as e:
        print(f"Error converting file to image: {e}")
        return []
    return images

def _parse_llm_json_output(text: str) -> Dict[str, Any]:
    """Robustly parse JSON from the LLM's raw text output."""
    # Find the first '{' and the last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    
    try:
        data = json.loads(text)
    except Exception as e:
        print(f"Failed to parse JSON from LLM output: {e}")
        data = {"_raw_output": text, "_parse_error": str(e)}
    return data

def _vlm_extract(images_b64: List[str]) -> Dict[str, Any]:
    """Call Ollama VLM with multiple images and a strict JSON-only instruction."""
    print(f"Extracting with VLM ({VLM_MODEL})...")
    messages = [
        {"role": "system", "content": f"You are an expert data extractor. {SCHEMA_HINT}"},
    ]
    
    # Add all images to the message context
    for b64 in images_b64:
        messages.append({
            "role": "user",
            "content": "Analyze this page and extract the data based on the schema. "
                       "Aggregate data from all pages I send you.",
            "images": [b64], # The ollama client takes a list of b64 strings
        })
    
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        resp = client.chat(model=VLM_MODEL, messages=messages, options={"temperature": 0.0})
        raw_text = resp.get("message", {}).get("content", "")
        return _parse_llm_json_output(raw_text)
    except Exception as e:
        print(f"Ollama VLM call failed: {e}")
        return {"_error": f"Ollama VLM call failed: {e}"}

def _text_llm_extract(doc_text: str) -> Dict[str, Any]:
    """Call text LLM (Llama3.1) with text and JSON schema."""
    print(f"Extracting with Text LLM ({TEXT_MODEL})...")
    
    prompt = f"""
    {SCHEMA_HINT}

    Here is the document text. Extract the information into the JSON schema.
    
    DOCUMENT TEXT:
    ---
    {doc_text[:8000]} 
    ---
    """
    
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        # Use 'generate' for simpler text-in, text-out
        resp = client.generate(
            model=TEXT_MODEL, 
            prompt=prompt, 
            options={"temperature": 0.0}
        )
        raw_text = resp.get("response", "")
        return _parse_llm_json_output(raw_text)
    except Exception as e:
        print(f"Ollama text call failed: {e}")
        return {"_error": f"Ollama text call failed: {e}"}


def extract_data_from_document(file_path: str) -> Dict[str, Any]:
    """
    Main extraction function.
    Tries Text-First, then falls back to VLM.
    Handles Excel files separately.
    """
    
    # 1. Handle Excel files with Pandas (easy)
    if file_path.lower().endswith((".xls", ".xlsx")):
        print(f"Parsing Excel file: {file_path}")
        try:
            df = pd.read_excel(file_path)
            return {"assets_liabilities": df.to_dict(orient="records")}
        except Exception as e:
            return {"_error": f"Failed to parse Excel: {e}"}

    # 2. Try Text-First path for Digital PDFs
    extracted_text = ""
    if file_path.lower().endswith(".pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                # Limit to 6 pages for performance
                for page in pdf.pages[:6]:
                    t = page.extract_text() or ""
                    extracted_text += "\n\n" + t
            
            # If we got a decent amount of text, it's a digital PDF.
            if len(extracted_text.strip()) > 200:
                print(f"Digital PDF detected. Using {TEXT_MODEL}.")
                return _text_llm_extract(extracted_text)
            else:
                print("Scanned PDF detected (no text found). Falling back to VLM.")
        except Exception as e:
            print(f"pdfplumber failed, falling back to VLM: {e}")
            pass # Fall through to VLM path

    # 3. VLM Fallback Path (for Scans and Images)
    print("Using VLM path for image or scanned PDF.")
    images_b64 = _file_to_b64_images(file_path)
    if not images_b64:
        return {"_error": "Could not convert file to images for VLM."}
    
    return _vlm_extract(images_b64)