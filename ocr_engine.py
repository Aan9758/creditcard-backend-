# app/ocr_engine.py

# ============================================================
# IMPORTS
# ============================================================
import os
import re
import json
import cv2
import subprocess
import numpy as np
import pandas as pd
import pytesseract
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import google.generativeai as genai
import time
from datetime import datetime

# ============================================================
# OPTIONAL: TESSERACT FIX (LINUX/DOCKER)
# ============================================================
def fix_tesseract():
    """
    Find Tesseract language data directory and set TESSDATA_PREFIX.
    On Windows this 'find' command won't work, so it's wrapped in try/except.
    """
    try:
        result = subprocess.run(
            ['find', '/usr', '-name', 'eng.traineddata'],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            tessdata_dir = os.path.dirname(result.stdout.strip().split('\n')[0])
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            print(f"✓ [ocr] Tesseract configured: {tessdata_dir}")
    except Exception as e:
        print(f"⚠️ [ocr] Tesseract auto-config skipped: {e}")


# ============================================================
# MERCHANT CATEGORY DATABASE
# ============================================================
class MerchantCategorizer:
    """Smart categorization based on merchant names"""

    def __init__(self):
        # Comprehensive merchant database (same as before)
        self.merchant_db = {
            # Food & Dining
            'Food & Dining': [
                'zomato', 'swiggy', 'uber eats', 'dunzo', 'zepto', 'blinkit', 'instamart',
                'dominos', 'pizza hut', 'mcdonalds', 'mcdonald', 'burger king', 'kfc',
                'subway', 'starbucks', 'cafe coffee day', 'ccd', 'costa coffee', 'barista',
                'haldiram', 'barbeque nation', 'mainland china', 'ohri', 'paradise',
                'saravana bhavan', 'sagar ratna', 'rajdhani', 'bikanervala', 'anand sweets',
                'monginis', 'theobroma', 'natural ice cream', 'baskin robbins', 'kwality walls',
                'restaurant', 'cafe', 'dhaba', 'food', 'pizza', 'burger', 'biryani',
                'chinese', 'dine', 'eatery', 'kitchen', 'grill', 'bakery', 'sweet',
                'ice cream', 'juice', 'beverage', 'snacks', 'mithai'
            ],
            # Groceries
            'Groceries': [
                'bigbasket', 'big basket', 'grofers', 'jiomart', 'dmart', 'd-mart', 'd mart',
                'reliance fresh', 'reliance smart', 'more supermarket', 'star bazaar',
                'spencer', 'hypercity', 'big bazaar', 'easyday', 'nature basket',
                'amazon fresh', 'amazon pantry', 'flipkart grocery',
                'kirana', 'provision', 'grocery', 'supermarket', 'vegetables', 'fruits',
                'dairy', 'milk', 'amul', 'mother dairy', 'verka'
            ],
            # Shopping
            'Shopping': [
                'amazon', 'flipkart', 'myntra', 'ajio', 'meesho', 'snapdeal', 'shopclues',
                'nykaa', 'purplle', 'mamaearth', 'tata cliq', 'croma', 'vijay sales',
                'reliance digital', 'apple store', 'samsung store', 'oneplus', 'xiaomi',
                'zara', 'h&m', 'pantaloons', 'westside', 'lifestyle', 'max fashion',
                'shoppers stop', 'central', 'brand factory', 'trends', 'fbb', 'v-mart',
                'reliance trends', 'bata', 'metro shoes', 'woodland', 'puma', 'nike', 'adidas',
                'decathlon', 'skechers', 'crocs', 'lenskart', 'titan', 'tanishq',
                'ikea', 'urban ladder', 'pepperfry', 'home centre', 'hometown',
                'mall', 'store', 'mart', 'retail', 'boutique', 'shop'
            ],
            # Travel
            'Travel': [
                'makemytrip', 'goibibo', 'cleartrip', 'yatra', 'ixigo', 'easemytrip',
                'air india', 'indigo', 'spicejet', 'vistara', 'go first', 'akasa',
                'air asia', 'emirates', 'etihad', 'qatar airways', 'singapore airlines',
                'oyo', 'treebo', 'fabhotels', 'taj hotels', 'itc hotels', 'oberoi',
                'marriott', 'hyatt', 'hilton', 'radisson', 'holiday inn', 'ibis',
                'airbnb', 'booking.com', 'agoda', 'trivago',
                'uber', 'ola', 'rapido', 'meru', 'savaari',
                'irctc', 'railway', 'train', 'metro card', 'dmrc',
                'redbus', 'abhibus', 'ksrtc', 'upsrtc', 'gsrtc', 'tsrtc', 'apsrtc',
                'flight', 'airline', 'airport', 'travel', 'tour', 'trip', 'holiday',
                'vacation', 'hotel', 'resort', 'booking', 'ticket'
            ],
            # Fuel
            'Fuel': [
                'indian oil', 'iocl', 'bharat petroleum', 'bpcl', 'hp petrol',
                'hindustan petroleum', 'hpcl', 'reliance petroleum', 'essar',
                'shell', 'nayara', 'petrol pump', 'fuel', 'petrol', 'diesel', 'gas station',
                'cng', 'ev charging', 'ather', 'tata power ez charge'
            ],
            # Entertainment
            'Entertainment': [
                'netflix', 'amazon prime', 'hotstar', 'disney plus', 'zee5', 'sony liv',
                'voot', 'jiocinema', 'mxplayer', 'youtube premium', 'spotify', 'gaana',
                'wynk', 'jiosaavn', 'apple music', 'audible',
                'playstation', 'xbox', 'steam', 'epic games', 'google play games',
                'pubg', 'free fire', 'dream11', 'my11circle', 'mpl',
                'bookmyshow', 'paytm movies', 'pvr', 'inox', 'cinepolis', 'carnival',
                'wonderla', 'imagica', 'essel world', 'fun city',
                'movie', 'cinema', 'theatre', 'gaming', 'concert', 'event', 'show',
                'amusement', 'park', 'club', 'disco', 'pub', 'bar', 'lounge'
            ],
            # Utilities
            'Utilities': [
                'tata power', 'adani electricity', 'bses', 'bescom', 'msedcl',
                'electricity', 'power bill', 'electric',
                'water bill', 'jal board', 'water supply',
                'mahanagar gas', 'indraprastha gas', 'adani gas', 'gas bill', 'lpg',
                'indane', 'bharat gas', 'hp gas',
                'jio', 'airtel', 'vodafone', 'idea', 'vi ', 'bsnl', 'mtnl',
                'airtel xstream', 'jio fiber', 'act fibernet', 'hathway', 'tata sky',
                'dish tv', 'd2h', 'sun direct',
                'recharge', 'prepaid', 'postpaid', 'broadband', 'wifi', 'dth',
                'cable', 'internet', 'telecom', 'mobile bill'
            ],
            # Healthcare
            'Healthcare': [
                'apollo pharmacy', 'medplus', 'netmeds', 'pharmeasy', '1mg', 'tata 1mg',
                'medlife', 'wellness forever',
                'apollo hospital', 'fortis', 'max healthcare', 'medanta', 'manipal',
                'aiims', 'narayana health', 'kokilaben', 'lilavati', 'hinduja',
                'dr lal path', 'srl diagnostics', 'thyrocare', 'metropolis',
                'health insurance', 'star health', 'max bupa', 'care health',
                'hospital', 'clinic', 'doctor', 'medical', 'pharmacy', 'medicine',
                'diagnostic', 'lab', 'pathology', 'healthcare', 'health', 'dental',
                'eye', 'optical', 'wellness', 'gym', 'fitness'
            ],
            # Education
            'Education': [
                'udemy', 'coursera', 'unacademy', 'byjus', 'vedantu', 'upgrad',
                'simplilearn', 'edx', 'skillshare', 'linkedin learning', 'pluralsight',
                'udacity', 'whitehat jr', 'toppr', 'meritnation', 'extramarks',
                'amazon kindle', 'google books', 'scribd', 'blinkist',
                'school fee', 'college fee', 'university', 'tuition', 'coaching',
                'education', 'course', 'class', 'training', 'tutorial', 'learn',
                'exam', 'test', 'books', 'stationery', 'uniform'
            ],
            # EMI
            'EMI': [
                'emi', 'loan', 'installment', 'equated monthly', 'finance',
                'bajaj finserv', 'hdfc credila', 'tata capital', 'capital float',
                'flexmoney', 'zestmoney', 'lazypay', 'simpl', 'slice',
                'home loan', 'car loan', 'personal loan', 'education loan',
                'credit card emi', 'no cost emi'
            ],
            # Payment/Banking
            'Payment': [
                'payment received', 'payment credited', 'credit received', 'refund',
                'cashback', 'reward', 'reversal', 'bank transfer', 'neft', 'imps',
                'rtgs', 'upi', 'paytm', 'phonepe', 'googlepay', 'google pay',
                'bhim', 'cred', 'mobikwik', 'freecharge', 'amazon pay'
            ],
            # Cashback
            'Cashback': [
                'cashback', 'cash back', 'reward', 'bonus', 'offer credit',
                'promotional credit', 'referral bonus'
            ],
            # Refund
            'Refund': [
                'refund', 'reversal', 'chargeback', 'return credit', 'cancellation'
            ],
            # Insurance
            'Insurance': [
                'lic', 'life insurance', 'term insurance', 'hdfc life', 'icici prudential',
                'max life', 'sbi life', 'tata aia', 'bajaj allianz', 'policy bazaar',
                'motor insurance', 'car insurance', 'bike insurance', 'vehicle insurance',
                'general insurance', 'travel insurance', 'acko', 'digit insurance'
            ],
            # Investment
            'Investment': [
                'mutual fund', 'sip', 'zerodha', 'groww', 'upstox', 'kite', 'coin',
                'paytm money', 'et money', 'kuvera', 'scripbox', 'smallcase',
                'stock', 'share', 'investment', 'trading', 'nse', 'bse', 'demat'
            ],
            # Rent
            'Rent': [
                'rent', 'house rent', 'flat rent', 'pg', 'paying guest', 'hostel',
                'nobroker', 'magicbricks', '99acres', 'housing.com', 'nestaway',
                'maintenance', 'society charges'
            ],
            # Subscriptions
            'Subscriptions': [
                'subscription', 'membership', 'annual fee', 'monthly fee',
                'amazon prime', 'netflix', 'spotify', 'zomato pro', 'swiggy one',
                'newspaper', 'magazine', 'premium'
            ],
            # Personal Care
            'Personal Care': [
                'salon', 'spa', 'beauty', 'parlour', 'parlor', 'haircut', 'grooming',
                'urban company', 'urban clap', 'lakme', 'naturals', 'jawed habib',
                'looks salon', 'vlcc', 'o2 spa', 'tattva spa'
            ],
            # Charity
            'Charity': [
                'donation', 'charity', 'ngo', 'foundation', 'trust', 'relief fund',
                'milaap', 'ketto', 'give india', 'goonj', 'akshaya patra'
            ],
            # Government
            'Government': [
                'govt', 'government', 'tax', 'gst', 'income tax', 'challan',
                'passport', 'visa', 'rto', 'traffic fine', 'municipality', 'court',
                'stamp duty', 'registration', 'aadhaar', 'pan card'
            ],
            # ATM
            'ATM Withdrawal': [
                'atm', 'cash withdrawal', 'atm withdrawal', 'cash advance'
            ],
            # Transfers
            'Transfers': [
                'transfer', 'self transfer', 'fund transfer', 'account transfer'
            ],
        }

        self._build_lookup()

    def _build_lookup(self):
        """Build reverse lookup dictionary"""
        self.keyword_to_category: Dict[str, str] = {}
        for category, keywords in self.merchant_db.items():
            for keyword in keywords:
                self.keyword_to_category[keyword.lower()] = category

    def categorize(self, description: str, current_category: str = "Others") -> Tuple[str, str]:
        if not description:
            return current_category, ""

        desc_lower = description.lower().strip()
        sorted_keywords = sorted(self.keyword_to_category.keys(), key=len, reverse=True)
        for keyword in sorted_keywords:
            if keyword in desc_lower:
                return self.keyword_to_category[keyword], keyword
        return current_category, ""

    def get_merchant_name(self, description: str) -> str:
        prefixes_to_remove = [
            'pos ', 'pos/', 'ecom ', 'online ', 'ach d-', 'ach/',
            'neft ', 'imps ', 'upi/', 'upi-', 'www.', 'http://', 'https://'
        ]

        result = description.lower()
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix):]

        result = re.sub(r'\d{10,}', '', result)
        result = re.sub(r'[a-z0-9]{12,}', '', result)
        result = ' '.join(result.split())
        return result.title().strip()


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Transaction:
    date: str = ""
    description: str = ""
    amount: float = 0.0
    transaction_type: str = "DEBIT"
    category: str = "Others"
    merchant: str = ""
    file_source: str = ""

    def to_dict(self) -> Dict:
        return {
            'date': self.date,
            'description': self.description,
            'amount': self.amount,
            'transaction_type': self.transaction_type,
            'category': self.category,
            'merchant': self.merchant,
            'file_source': self.file_source
        }


@dataclass
class ParsedStatement:
    bank_name: str = ""
    card_number: str = ""
    statement_period: str = ""
    transactions: List[Transaction] = field(default_factory=list)
    raw_text: str = ""
    file_name: str = ""
    processing_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'bank_name': self.bank_name,
            'card_number': self.card_number,
            'statement_period': self.statement_period,
            'file_name': self.file_name,
            'processing_time': self.processing_time,
            'total_transactions': len(self.transactions),
            'transactions': [t.to_dict() for t in self.transactions]
        }


@dataclass
class BatchResult:
    statements: List[ParsedStatement] = field(default_factory=list)
    total_files: int = 0
    successful_files: int = 0
    failed_files: List[str] = field(default_factory=list)
    total_transactions: int = 0
    total_processing_time: float = 0.0

    def add_statement(self, statement: ParsedStatement):
        self.statements.append(statement)
        self.successful_files += 1
        self.total_transactions += len(statement.transactions)

    def add_failure(self, filename: str):
        self.failed_files.append(filename)

    def get_all_transactions(self) -> List[Transaction]:
        all_txns: List[Transaction] = []
        for statement in self.statements:
            all_txns.extend(statement.transactions)
        return all_txns

    def to_dict(self) -> Dict:
        return {
            'total_files': self.total_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'total_transactions': self.total_transactions,
            'total_processing_time': self.total_processing_time,
            'statements': [s.to_dict() for s in self.statements]
        }


# ============================================================
# GEMINI LLM EXTRACTOR
# ============================================================
class GeminiExtractor:
    def __init__(self, api_key: str):
        if not api_key or len(api_key) < 20:
            raise ValueError("❌ Invalid API key! Please add your Gemini API key.")

        print("⚙️ [gemini] Configuring Gemini client...")
        genai.configure(api_key=api_key)
        self.model = None
        self._init_model()

    def _init_model(self):
        model_name = "models/gemini-2.5-flash"

        print("\n🔍 [gemini] Connecting to Gemini...")
        print(f"   [gemini] Using model: {model_name} ...", end=" ")

        try:
            self.model = genai.GenerativeModel(model_name)
            response = self.model.generate_content("Reply with: OK")
            if response and getattr(response, "text", None):
                print("✓")
                print(f"✓ [gemini] Connected to {model_name}")
                return
            print("✗ (empty response)")
        except Exception as e:
            print(f"✗ ({str(e)[:80]})")

        raise ValueError(
            "❌ Could not connect to Gemini with model 'models/gemini-2.5-flash'. "
            "Check your API key or model availability."
        )

    def get_prompt(self) -> str:
        return """You are a bank statement transaction extractor. Extract ONLY actual financial transactions.

EXTRACT:
- Purchases at stores/merchants
- Online payments
- Food orders (Zomato, Swiggy, etc.)
- Bill payments
- EMI payments
- Refunds and cashback
- Payment credits

IGNORE:
- Bank addresses, contact info
- Terms and conditions
- Promotional offers
- Account summaries
- Headers/footers
- Credit limits, interest rates

FOR EACH TRANSACTION:
- date: YYYY-MM-DD format (use 2025 if year not shown)
- description: FULL merchant name as shown (DO NOT abbreviate)
- amount: Number only (no currency symbols)
- transaction_type: "DEBIT" (money spent) or "CREDIT" (payment/refund received)
- category: One of:
  "Food & Dining", "Groceries", "Shopping", "Travel", "Fuel",
  "Entertainment", "Utilities", "Healthcare", "Education",
  "EMI", "Payment", "Cashback", "Refund", "Insurance",
  "Investment", "Rent", "Subscriptions", "Personal Care",
  "Charity", "Government", "ATM Withdrawal", "Transfers", "Others"

RETURN ONLY JSON ARRAY:
[
  {"date": "2025-09-04", "description": "ZOMATO ORDER BANGALORE", "amount": 543.00, "transaction_type": "DEBIT", "category": "Food & Dining"},
  {"date": "2025-09-05", "description": "AMAZON PAY INDIA PVT LTD", "amount": 2499.00, "transaction_type": "DEBIT", "category": "Shopping"}
]"""

    def extract(self, text: str) -> List[Dict]:
        max_len = 25000
        original_len = len(text)
        if original_len > max_len:
            print(f"   [gemini] Text truncated: {original_len} → {max_len} chars")
            text = text[:max_len]

        prompt = f"""{self.get_prompt()}

BANK STATEMENT TEXT:
\"\"\" 
{text}
\"\"\" 

Return ONLY the JSON array of transactions:"""

        for attempt in range(3):
            try:
                print(f"   [gemini] Extraction attempt {attempt + 1}/3...", end=" ")

                start = time.time()
                response = self.model.generate_content(prompt)
                duration = time.time() - start
                print(f" (LLM time: {duration:.2f}s)", end=" ")

                if not response or not getattr(response, "text", None):
                    print("→ Empty response")
                    time.sleep(1)
                    continue

                result = response.text.strip()

                # Clean markdown wrappers if present
                result = re.sub(r'^```json\s*\n?', '', result)
                result = re.sub(r'^```\s*\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
                result = result.strip()

                # Handle wrapped object like {"transactions":[...]}
                if result.startswith('{') and '"transactions"' in result:
                    obj = json.loads(result)
                    if 'transactions' in obj:
                        result = json.dumps(obj['transactions'])

                transactions = json.loads(result)

                if isinstance(transactions, list):
                    print(f"→ ✓ {len(transactions)} transactions")
                    return transactions
                else:
                    print("→ JSON not list")
            except json.JSONDecodeError:
                print("→ JSON error")
                time.sleep(1)
            except Exception as e:
                print(f"→ Error: {str(e)[:60]}")
                time.sleep(2)

        print("   ⚠️ [gemini] Could not extract transactions after 3 attempts")
        return []


# ============================================================
# PDF PROCESSOR
# ============================================================
class PDFProcessor:
    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def extract_text(self, pdf_path: str) -> str:
        print(f"   [pdf] Extracting text from: {pdf_path}")
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                print(f"   [pdf] PDF has {num_pages} page(s)")
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        print(f"   [pdf] Page {i+1}: {len(page_text)} chars")
                        text += f"\n--- Page {i+1} ---\n"
                        text += page_text + "\n"
                    else:
                        print(f"   [pdf] Page {i+1}: no text extracted")
        except Exception as e:
            print(f"   [pdf] PDF error: {e}")

        print(f"   [pdf] Total extracted chars: {len(text)}")
        return text

    def is_scanned(self, pdf_path: str) -> bool:
        text = self.extract_text(pdf_path)
        is_scanned = len(text.strip()) < 100
        print(f"   [pdf] is_scanned={is_scanned}")
        return is_scanned

    def to_images(self, pdf_path: str) -> List[Image.Image]:
        try:
            print(f"   [pdf] Converting PDF to images at {self.dpi} DPI...")
            images = convert_from_path(pdf_path, dpi=self.dpi)
            print(f"   [pdf] Converted to {len(images)} image(s)")
            return images
        except Exception as e:
            print(f"   [pdf] Image conversion error: {e}")
            return []


# ============================================================
# IMAGE OCR PROCESSOR
# ============================================================
class ImageOCR:
    def preprocess(self, pil_image: Image.Image) -> Image.Image:
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return Image.fromarray(gray)

    def extract_text(self, pil_image: Image.Image) -> str:
        try:
            print("   [ocr] Running Tesseract on image...")
            processed = self.preprocess(pil_image)
            text = pytesseract.image_to_string(
                processed, lang='eng', config='--oem 3 --psm 6'
            )
            print(f"   [ocr] OCR text length: {len(text)}")
            return text
        except Exception as e:
            print(f"   [ocr] OCR error: {e}")
            return ""


# ============================================================
# MAIN OCR CLASS
# ============================================================
class BankStatementOCR:
    def __init__(self, api_key: str):
        print("\n🚀 [ocr] Initializing Bank Statement OCR...")
        self.pdf_processor = PDFProcessor()
        self.image_ocr = ImageOCR()
        self.llm = GeminiExtractor(api_key)
        self.categorizer = MerchantCategorizer()
        print("✓ [ocr] Bank Statement OCR Ready!\n")

    def process(self, file_path: str) -> ParsedStatement:
        print("\n==================================================")
        print("🔍 [ocr] OCR PROCESS STARTED")
        print("==================================================")
        print(f"📁 [ocr] File path: {file_path}")

        path = Path(file_path)
        result = ParsedStatement()
        result.file_name = path.name

        start_time = time.time()

        ext = path.suffix.lower()
        print(f"🔎 [ocr] File extension: {ext}")

        # Step 1: Extract text from PDF or image
        if ext == '.pdf':
            print("📝 [ocr] Step 1: PDF detected")
            text = self._process_pdf(str(path))
            print("📝 [ocr] Step 1 DONE: PDF text extracted")
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            print("🖼 [ocr] Step 1: Image detected")
            text = self._process_image(str(path))
            print("🖼 [ocr] Step 1 DONE: Image text extracted")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        result.raw_text = text
        print(f"   [ocr] Total text length: {len(text)} characters")

        # Step 2: Metadata
        print("\n📑 [ocr] Step 2: Extracting metadata...")
        result.bank_name = self._detect_bank(text)
        print(f"   [ocr] Bank: {result.bank_name}")
        result.card_number = self._find_card_number(text)
        print(f"   [ocr] Card number: {result.card_number}")
        result.statement_period = self._find_period(text)
        print(f"   [ocr] Statement period: {result.statement_period}")

        # Step 3: LLM extraction
        print("\n🤖 [ocr] Step 3: Calling Gemini for transaction extraction...")
        raw_transactions = self.llm.extract(text)
        print(f"🤖 [ocr] LLM returned {len(raw_transactions)} raw transaction objects")

        # Step 4: Convert to Transaction objects
        print("\n📦 [ocr] Step 4: Normalizing transactions + categorization...")
        for idx, tx_data in enumerate(raw_transactions, start=1):
            if not tx_data or not isinstance(tx_data, dict):
                print(f"   [ocr] Skipping non-dict tx at index {idx}")
                continue

            amount = self._parse_amount(tx_data.get('amount'))
            if amount <= 0:
                print(f"   [ocr] Skipping tx index {idx}: non-positive amount {tx_data.get('amount')}")
                continue

            description = self._clean_description(tx_data.get('description', ''))
            original_category = str(tx_data.get('category', 'Others')).strip()
            smart_category, matched_keyword = self.categorizer.categorize(
                description, original_category
            )
            merchant_name = self.categorizer.get_merchant_name(description)

            tx = Transaction(
                date=str(tx_data.get('date', '')).strip(),
                description=description,
                amount=amount,
                transaction_type=str(tx_data.get('transaction_type', 'DEBIT')).upper().strip(),
                category=smart_category,
                merchant=merchant_name if merchant_name else description[:30],
                file_source=path.name
            )

            if tx.transaction_type not in ['DEBIT', 'CREDIT']:
                print(f"   [ocr] Fixing invalid transaction_type '{tx.transaction_type}' → 'DEBIT'")
                tx.transaction_type = 'DEBIT'

            result.transactions.append(tx)

        result.processing_time = time.time() - start_time
        print("\n✅ [ocr] PROCESS COMPLETE")
        print(f"   [ocr] Total transactions: {len(result.transactions)}")
        print(f"   [ocr] Total time: {result.processing_time:.2f}s")

        return result

    def process_batch(self, file_paths: List[str]) -> BatchResult:
        batch_result = BatchResult()
        batch_result.total_files = len(file_paths)

        print("\n📦 [ocr] BATCH PROCESSING STARTED")
        start_time = time.time()

        for i, file_path in enumerate(file_paths, 1):
            print(f"\n──────── [ocr] File {i}/{len(file_paths)}: {Path(file_path).name}")
            try:
                statement = self.process(file_path)
                batch_result.add_statement(statement)
                print(f"✓ [ocr] Successfully extracted {len(statement.transactions)} transactions")
            except Exception as e:
                print(f"❌ [ocr] Failed for {file_path}: {e}")
                batch_result.add_failure(file_path)

        batch_result.total_processing_time = time.time() - start_time
        print("\n📦 [ocr] BATCH DONE")
        print(f"   [ocr] Total files: {batch_result.total_files}")
        print(f"   [ocr] Successful: {batch_result.successful_files}")
        print(f"   [ocr] Failed: {len(batch_result.failed_files)}")
        print(f"   [ocr] Total time: {batch_result.total_processing_time:.2f}s")

        return batch_result

    def _process_pdf(self, pdf_path: str) -> str:
        if self.pdf_processor.is_scanned(pdf_path):
            print("   [ocr] Scanned PDF detected → using OCR on images...")
            images = self.pdf_processor.to_images(pdf_path)
            all_text = []
            for i, img in enumerate(images):
                print(f"   [ocr] OCR on page {i+1}/{len(images)}...")
                all_text.append(self.image_ocr.extract_text(img))
            combined = "\n".join(all_text)
            print(f"   [ocr] Combined OCR text length: {len(combined)}")
            return combined
        else:
            print("   [ocr] Digital PDF detected → direct text extraction...")
            return self.pdf_processor.extract_text(pdf_path)

    def _process_image(self, image_path: str) -> str:
        print(f"   [ocr] Loading image: {image_path}")
        image = Image.open(image_path)
        return self.image_ocr.extract_text(image)

    def _detect_bank(self, text: str) -> str:
        text_lower = text.lower()
        banks = {
            'American Express': ['american express', 'amex', 'membership rewards'],
            'HDFC Bank': ['hdfc bank', 'hdfc credit card'],
            'ICICI Bank': ['icici bank', 'icici credit'],
            'SBI Card': ['sbi card', 'state bank of india'],
            'Axis Bank': ['axis bank', 'axis credit'],
            'Kotak Mahindra': ['kotak mahindra', 'kotak bank'],
            'Amazon Pay ICICI': ['amazon pay icici'],
            'Citi Bank': ['citibank', 'citi credit card'],
            'IDFC First': ['idfc first', 'idfc bank'],
            'RBL Bank': ['rbl bank'],
            'IndusInd Bank': ['indusind bank'],
            'Yes Bank': ['yes bank'],
            'HSBC': ['hsbc bank', 'hsbc credit'],
            'Standard Chartered': ['standard chartered'],
        }
        for bank_name, keywords in banks.items():
            if any(kw in text_lower for kw in keywords):
                return bank_name
        return "Unknown Bank"

    def _find_card_number(self, text: str) -> str:
        patterns = [
            r'XXXX-XXXXXX-\d{5}',
            r'XXXX[\s-]*XXXX[\s-]*XXXX[\s-]*\d{4}',
            r'\d{4}[\s-]*\*{4,}[\s-]*\*{4,}[\s-]*\d{4}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return "Not Found"

    def _find_period(self, text: str) -> str:
        patterns = [
            r'[Ss]tatement\s*[Pp]eriod[\s:]*[Ff]rom\s*(.+?)\s*to\s*(.+?)(?:\n|$)',
            r'[Ff]rom\s+([A-Za-z]+\s+\d{1,2})\s+to\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return f"{match.group(1).strip()} to {match.group(2).strip()}"
        return "Not Found"

    def _parse_amount(self, value) -> float:
        if value is None:
            return 0.0
        try:
            if isinstance(value, (int, float)):
                return abs(float(value))
            cleaned = re.sub(r'[^\d.]', '', str(value))
            return abs(float(cleaned)) if cleaned else 0.0
        except Exception:
            return 0.0

    def _clean_description(self, desc: str) -> str:
        if not desc:
            return ""
        desc = ' '.join(str(desc).split())
        return desc.strip()

    def to_dataframe(self, data) -> pd.DataFrame:
        if isinstance(data, ParsedStatement):
            transactions = data.transactions
        elif isinstance(data, BatchResult):
            transactions = data.get_all_transactions()
        else:
            transactions = data

        if not transactions:
            return pd.DataFrame(
                columns=['Date', 'Description', 'Merchant', 'Amount', 'Type', 'Category', 'Source']
            )

        records = []
        for tx in transactions:
            records.append({
                'Date': tx.date,
                'Description': tx.description,
                'Merchant': tx.merchant,
                'Amount': tx.amount,
                'Type': tx.transaction_type,
                'Category': tx.category,
                'Source': tx.file_source
            })

        return pd.DataFrame(records)

    def save_csv(self, data, path: str):
        df = self.to_dataframe(data)
        df.to_csv(path, index=False)
        return path

    def save_json(self, data, path: str):
        if isinstance(data, ParsedStatement):
            output = data.to_dict()
        elif isinstance(data, BatchResult):
            output = data.to_dict()
        else:
            output = data

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return path
