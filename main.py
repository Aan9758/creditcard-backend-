# app/main.py

import os
from pathlib import Path
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .ocr_engine import BankStatementOCR

# ============================================================
# ENV + OCR INITIALIZATION
# ============================================================
print("🔧 [main] Loading environment variables...")
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

print("🔧 [main] GEMINI_API_KEY loaded prefix:", GEMINI_API_KEY[:6], "******")

print("🚀 [main] Initializing BankStatementOCR...")
ocr = BankStatementOCR(GEMINI_API_KEY)
print("✅ [main] BankStatementOCR ready")

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Flip & Pay – Statement OCR & Spend Categorizer",
    version="1.0.0",
    description="Upload a bank statement (PDF/image) and get structured transactions."
)

# (Optional) CORS for frontend / mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
def health():
    print("💚 [main] /health called")
    return {"status": "ok"}


@app.post("/parse-statement")
async def parse_statement(file: UploadFile = File(...)):
    print("\n📥 [main] API CALL: /parse-statement")
    print(f"📄 [main] Uploaded file: {file.filename} (content_type={file.content_type})")

    # Decide temporary path
    suffix = Path(file.filename).suffix or ".pdf"
    temp_filename = f"temp_{file.filename}"
    temp_path = Path(temp_filename).resolve()

    try:
        # Save uploaded file to disk
        print(f"💾 [main] Saving temp file at: {temp_path}")
        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        print(f"📂 [main] File saved. Size: {len(content)} bytes")
        print("🚀 [main] Sending to OCR engine...")

        # Call OCR engine
        result = ocr.process(str(temp_path))

        print("🎉 [main] OCR completed successfully!")
        print(f"📊 [main] Transactions extracted: {len(result.transactions)}")
        print(f"⏱️ [main] Processing time: {result.processing_time:.2f}s")

        return result.to_dict()

    except Exception as e:
        print("❌ [main] ERROR in /parse-statement:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
                print(f"🗑️ [main] Temp file removed: {temp_path}")
            except Exception as cleanup_err:
                print(f"⚠️ [main] Could not remove temp file: {cleanup_err}")
