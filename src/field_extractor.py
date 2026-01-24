"""
Field Extractor Module
Extracts structured fields from OCR text using regex, NLP, and layout heuristics.
Generalizable to different invoice types through configuration.
Supports multilingual extraction with language detection.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from rapidfuzz import fuzz, process
import yaml
from pathlib import Path


# ============================================================================
# MULTILINGUAL SUPPORT: Unicode Script Detection for Indian Languages
# ============================================================================

# Unicode ranges for Indian scripts - covers all major languages
INDIC_SCRIPTS = {
    'devanagari': (r'[\u0900-\u097F]', ['hi', 'mr', 'sa']),   # Hindi, Marathi, Sanskrit
    'gujarati': (r'[\u0A80-\u0AFF]', ['gu']),                  # Gujarati
    'gurmukhi': (r'[\u0A00-\u0A7F]', ['pa']),                  # Punjabi
    'tamil': (r'[\u0B80-\u0BFF]', ['ta']),                     # Tamil
    'telugu': (r'[\u0C00-\u0C7F]', ['te']),                    # Telugu
    'kannada': (r'[\u0C80-\u0CFF]', ['kn']),                   # Kannada
    'malayalam': (r'[\u0D00-\u0D7F]', ['ml']),                 # Malayalam
    'odia': (r'[\u0B00-\u0B7F]', ['or']),                      # Odia
    'bengali': (r'[\u0980-\u09FF]', ['bn', 'as']),             # Bengali, Assamese
}


def detect_script(text: str) -> Tuple[str, float]:
    """
    Detect the primary script of the text using Unicode ranges.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (script_name, confidence)
    """
    if not text:
        return 'latin', 0.0
    
    total_chars = len(text.replace(' ', ''))
    if total_chars == 0:
        return 'latin', 0.0
    
    script_counts = {}
    for script_name, (pattern, _) in INDIC_SCRIPTS.items():
        matches = re.findall(pattern, text)
        if matches:
            script_counts[script_name] = len(matches)
    
    if script_counts:
        best_script = max(script_counts, key=script_counts.get)
        confidence = script_counts[best_script] / total_chars
        return best_script, confidence
    
    return 'latin', 1.0


def is_indic_text(text: str) -> bool:
    """Check if text contains any Indic script characters."""
    for script_name, (pattern, _) in INDIC_SCRIPTS.items():
        if re.search(pattern, text):
            return True
    return False


def get_language_code(script: str) -> str:
    """Get ISO language code from script name."""
    if script in INDIC_SCRIPTS:
        return INDIC_SCRIPTS[script][1][0]  # Return first lang code
    return 'en'


@dataclass
class ExtractedField:
    """Container for an extracted field."""
    field_name: str
    value: Any
    confidence: float
    source_text: str = ""
    bbox: Optional[List[int]] = None
    extraction_method: str = ""
    language: str = "en"  # Language of extracted text
    language_confidence: float = 0.0  # Language detection confidence


@dataclass
class ExtractionResult:
    """Container for all extracted fields from a document."""
    doc_id: str
    fields: Dict[str, ExtractedField] = field(default_factory=dict)
    processing_time_sec: float = 0.0
    overall_confidence: float = 0.0
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary matching problem specification."""
        # Always include all 6 required fields with proper defaults
        output_fields = {
            'dealer_name': None,
            'model_name': None,
            'horse_power': None,
            'asset_cost': None,
            'signature': {'present': False, 'bbox': []},
            'stamp': {'present': False, 'bbox': []}
        }
        
        # Language metadata for multilingual output
        language_metadata = {}
        
        # Fill in extracted values with strict validation for numeric fields
        for name, f in self.fields.items():
            if f.field_name in ['signature', 'stamp']:
                output_fields[name] = {
                    'present': bool(f.value),
                    'bbox': f.bbox or []
                }
            else:
                value = f.value
                # STRICT: horse_power and asset_cost MUST be integers (digits only)
                if name in ['horse_power', 'asset_cost'] and value is not None:
                    if isinstance(value, (int, float)):
                        output_fields[name] = int(value)
                    elif isinstance(value, str):
                        clean = re.sub(r'[^\d]', '', str(value))
                        output_fields[name] = int(clean) if clean.isdigit() else None
                    else:
                        output_fields[name] = None
                else:
                    output_fields[name] = value
        
        return {
            'doc_id': self.doc_id,
            'fields': output_fields,
            'confidence': round(self.overall_confidence, 2),
            'processing_time_sec': round(self.processing_time_sec, 2),
            'cost_estimate_usd': 0.0  # Open-source CPU-based inference
        }


class FieldExtractor:
    """
    Extracts structured fields from OCR results using NLP and layout analysis.
    
    Pipeline: OCR blocks → NLP scoring → Layout analysis → Post-processing
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize field extractor.
        
        Args:
            config_path: Path to field configuration YAML
        """
        self.config = self._load_config(config_path)
        self.invoice_type = 'tractor_quotation'
        self._layout_detector = None
        self._validator = None
        
        # Initialize layout detector
        try:
            from layout_detector import LayoutDetector
            self._layout_detector = LayoutDetector()
        except Exception as e:
            print(f"  ⚠️ Layout detector not available: {e}")
        
        # Initialize validator
        try:
            from validators import FieldValidator
            self._validator = FieldValidator()
        except Exception as e:
            print(f"  ⚠️ Validator not available: {e}")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load field configuration from YAML."""
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / 'config' / 'field_config.yaml'
        
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Return default config
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default extraction config."""
        return {
            'invoice_types': {
                'tractor_quotation': {
                    'fields': {
                        'dealer_name': {'type': 'text', 'keywords': ['dealer', 'motors', 'tractors']},
                        'model_name': {'type': 'text', 'keywords': ['model', 'tractor']},
                        'horse_power': {'type': 'numeric', 'patterns': [r'(\d{2,3})\s*[Hh][Pp]']},
                        'asset_cost': {'type': 'currency', 'patterns': [r'(\d{1,2},?\d{2},?\d{3})']},
                    }
                }
            }
        }
    
    def set_invoice_type(self, invoice_type: str):
        """Set the invoice type to use for extraction."""
        self.invoice_type = invoice_type

    def _sanitize_text(self, text: str) -> str:
        """
        Clean text by removing excessive symbols while preserving Hindi, Gujarati, and alphanumeric.
        """
        if not text: return ""
        # Remove dots/symbols at start/end
        text = text.strip('. ,:-_=+*#@')
        # Keep only alphanumeric, common punctuation, and Indic scripts
        # \u0900-\u097F: Devanagari, \u0A80-\u0AFF: Gujarati
        sanitized = re.sub(r'[^\w\s\-\u0900-\u097F\u0A80-\u0AFF]', ' ', text)
        # Collapse whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized

    def _calculate_noise_score(self, text: str) -> float:
        """
        Calculate noise score 0.0-1.0 (1.0 = pure noise).
        Identifies junk OCR by symbol density and fragmentation.
        """
        if not text: return 1.0
        total_len = len(text)
        # Count symbols (including dots used for fragmentation)
        symbols = len(re.findall(r'[^\w\s\u0900-\u097F\u0A80-\u0AFF]', text))
        noise_ratio = symbols / total_len if total_len > 0 else 1.0
        
        # FRAGMENTATION: Check for many single characters or short blocks
        # Split by any symbol or space to find real word fragments
        frags = re.split(r'[^\w\u0900-\u097F\u0A80-\u0AFF]+', text)
        frags = [f for f in frags if f]
        if len(frags) > 3:
            single_char_frags = sum(1 for f in frags if len(f) <= 2)
            fragment_ratio = single_char_frags / len(frags)
            if fragment_ratio > 0.6:
                noise_ratio = max(noise_ratio, fragment_ratio)
        
        # REPETITION: Check for repeated chars like "...." or "0000"
        if re.search(r'([\. \-_0])\1{2,}', text):
            noise_ratio = max(noise_ratio, 0.6)
            
        # DIGIT JUNK: Mostly digits but no decimals (likely serial nos/junk)
        digit_count = len(re.findall(r'\d', text))
        if total_len > 4 and digit_count / total_len > 0.7 and '.' not in text:
            noise_ratio = max(noise_ratio, 0.7)

        return min(1.0, noise_ratio)

    def _is_suspicious_amount(self, value: int, source_text: str) -> bool:
        """Check if an amount is suspicious for an asset cost."""
        val_str = str(value)
        # Mobile number fragments or serial numbers
        if len(val_str) > 8: return True
        
        # Tractor prices are almost always rounded.
        # If not rounded to at least 10s (e.g. 98155), it's very suspicious.
        if value % 10 != 0:
            if not re.search(r'[₹Rs]', source_text):
                return True
        
        # Larger amounts (most tractors) should be rounded to 100s
        if value > 50000 and value % 100 != 0:
            if not re.search(r'[₹Rs]', source_text):
                return True
        
        return False
    
    def extract_all_fields(self, ocr_data: Dict[str, Any], doc_id: str = "unknown") -> ExtractionResult:
        """
        Extract all configured fields from OCR data.
        
        Args:
            ocr_data: OCR output with text blocks and positions
            doc_id: Document identifier
            
        Returns:
            ExtractionResult with all extracted fields
        """
        result = ExtractionResult(doc_id=doc_id)
        
        full_text = ocr_data.get('full_text', '')
        blocks = ocr_data.get('blocks', [])
        raw_results = ocr_data.get('raw_results', [])
        
        # Get field configs for current invoice type
        invoice_config = self.config.get('invoice_types', {}).get(self.invoice_type, {})
        field_configs = invoice_config.get('fields', {})
        
        # Extract each field
        for field_name, field_cfg in field_configs.items():
            field_type = field_cfg.get('type', 'text')
            
            if field_type == 'visual':
                # Skip visual fields (handled by YOLO detector)
                continue
            elif field_type == 'numeric':
                extracted = self._extract_numeric_field(full_text, blocks, field_cfg, field_name)
            elif field_type == 'currency':
                extracted = self._extract_currency_field(full_text, blocks, field_cfg, field_name)
            else:  # text
                extracted = self._extract_text_field(full_text, blocks, field_cfg, field_name)
            
            if extracted:
                result.fields[field_name] = extracted
        
        # Calculate overall confidence
        if result.fields:
            confidences = [f.confidence for f in result.fields.values()]
            result.overall_confidence = sum(confidences) / len(confidences)
        
        return result
    
    def _extract_numeric_field(self, full_text: str, blocks: List[Dict], 
                                config: Dict, field_name: str) -> Optional[ExtractedField]:
        """Extract a numeric field using regex patterns."""
        patterns = config.get('patterns', [])
        keywords = config.get('keywords', [])
        
        # Enhanced patterns for HP extraction (handles OCR errors)
        if field_name == 'horse_power':
            # Common HP patterns in tractor documents
            hp_patterns = [
                r'(\d{2,3})\s*[Hh][Pp]',
                r'[Hh][Pp][-:\s]*(\d{2,3})',
                r'(\d{2,3})\s*H\.?P\.?',
                r'HP[-\s]?(\d{2,3})',
                r'(\d{2})[Hh][Pp]',  # No space like 41HP
                r'\((\d{2,3})[Hh][Pp]\)',  # In parentheses
                r'(\d{2,3})\s*अश्वशक्ति',  # Hindi
                r'(\d{2,3})\s*hp',
                # OCR error tolerant patterns
                r'(\d{2})H[Pp]',  # 41HP
                r'(\d{2})[Hh]\.?[Pp]\.?',
            ]
            
            for pattern in hp_patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    try:
                        value = int(match)
                        # Validate HP range (10-200)
                        if 10 <= value <= 200:
                            return ExtractedField(
                                field_name=field_name,
                                value=value,
                                confidence=0.85,
                                source_text=str(match),
                                extraction_method='regex'
                            )
                    except ValueError:
                        continue
            
            # Look for HP in context near "HP" keyword
            hp_context = re.findall(r'.{0,10}[Hh][Pp].{0,10}', full_text)
            for ctx in hp_context:
                # Extract digits from context
                digits = re.findall(r'(\d{2,3})', ctx)
                for d in digits:
                    value = int(d)
                    if 10 <= value <= 200:
                        return ExtractedField(
                            field_name=field_name,
                            value=value,
                            confidence=0.75,
                            source_text=ctx.strip(),
                            extraction_method='context_extraction'
                        )
            
            return None
        
        # Default numeric extraction for other fields
        for pattern in patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        value = int(re.sub(r'[,\s]', '', str(match)))
                        if field_name == 'horse_power' and (value < 10 or value > 200):
                            continue
                        return ExtractedField(
                            field_name=field_name,
                            value=value,
                            confidence=0.85,
                            source_text=str(match),
                            extraction_method='regex'
                        )
                    except ValueError:
                        continue
        
        # Try keyword-based extraction
        for keyword in keywords:
            kw_pattern = rf'{keyword}[-:\s]*(\d{{1,3}})'
            matches = re.findall(kw_pattern, full_text, re.IGNORECASE)
            if matches:
                try:
                    value = int(matches[0])
                    if field_name == 'horse_power' and (value < 10 or value > 200):
                        continue
                    return ExtractedField(
                        field_name=field_name,
                        value=value,
                        confidence=0.9,
                        source_text=f"{keyword}: {matches[0]}",
                        extraction_method='keyword_proximity'
                    )
                except ValueError:
                    continue
        
        return None
    
    def _extract_currency_field(self, full_text: str, blocks: List[Dict],
                                 config: Dict, field_name: str) -> Optional[ExtractedField]:
        """Extract a currency/amount field."""
        patterns = config.get('patterns', [])
        keywords = config.get('keywords', ['total', 'amount', 'price'])
        
        candidates = []
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean and parse amount
                    clean_match = re.sub(r'[,\s₹Rs\.]', '', str(match))
                    clean_match = re.sub(r'[^\d]', '', clean_match)
                    if clean_match:
                        value = int(clean_match)
                        # Reasonable range for tractor costs (50k - 50 lakhs)
                        if 50000 <= value <= 5000000:
                            candidates.append((value, 0.7, str(match)))
                except ValueError:
                    continue
        
        # Score candidates based on keyword proximity in context
        for keyword in keywords:
            # Find keyword in text and look for nearby numbers
            kw_matches = list(re.finditer(keyword, full_text, re.IGNORECASE))
            for kw_match in kw_matches:
                # Look for numbers within 50 chars of keyword
                start = max(0, kw_match.start() - 50)
                end = min(len(full_text), kw_match.end() + 50)
                context = full_text[start:end]
                
                # Find numbers in context
                numbers = re.findall(r'[\d,]+', context)
                for num_str in numbers:
                    try:
                        clean = re.sub(r'[,]', '', num_str)
                        if clean and len(clean) >= 5:  # At least 5 digits
                            value = int(clean)
                            if 50000 <= value <= 5000000:
                                # Penalize suspicious amounts
                                conf = 0.9
                                if self._is_suspicious_amount(value, num_str):
                                    conf = 0.4
                                
                                # Higher confidence for keyword-proximate matches
                                candidates.append((value, conf, num_str))
                    except ValueError:
                        continue
        
        if candidates:
            # Filter out very low confidence candidates if better ones exist
            high_conf = [c for c in candidates if c[1] > 0.5]
            if high_conf:
                candidates = high_conf
            # Choose the highest confidence match
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]
            
            # If multiple candidates have same confidence, choose largest amount
            same_conf = [c for c in candidates if c[1] == best[1]]
            if len(same_conf) > 1:
                best = max(same_conf, key=lambda x: x[0])
            
            return ExtractedField(
                field_name=field_name,
                value=best[0],
                confidence=best[1],
                source_text=best[2],
                extraction_method='pattern_match'
            )
        
        return None
    
    def _extract_text_field(self, full_text: str, blocks: List[Dict],
                            config: Dict, field_name: str) -> Optional[ExtractedField]:
        """Extract a text field (dealer name, model name)."""
        keywords = config.get('keywords', [])
        known_brands = config.get('known_brands', [])
        position_hint = config.get('position_hint', '')
        
        if field_name == 'dealer_name':
            return self._extract_dealer_name(full_text, blocks, config)
        elif field_name == 'model_name':
            return self._extract_model_name(full_text, blocks, config)
        
        return None
    
    def _extract_dealer_name(self, full_text: str, blocks: List[Dict], 
                             config: Dict) -> Optional[ExtractedField]:
        """Heuristic dealer name extractor.

        Goals:
        - Prefer business-like names ("Krishna Traders", "XYZ Tractors")
        - Strongly avoid numeric/amount lines like "550000.00" / "Grand Total: 550000.00"
        - Prefer header area, but still allow footer dealer stamps (M/s ...)
        """
        dealer_keywords = [
            'tractors', 'motors', 'agency', 'enterprises', 'pvt', 'ltd', 'traders',
            'trader', 'auto', 'sales', 'spares', 'borwells', 'services', 'showroom',
            'authorized', 'distributor', 'dealer', 'corporation', 'trading', 'works',
            # Common Hindi/Marathi/Gujarati business words
            'विपुल', 'ट्रैक्टर्स', 'मोटर्स', 'एजेंसी', 'ट्रेडर्स', 'कृष्णा', 'लिमिटेड', 
            'एन्टरप्राइजेज', 'सेल्स', 'सर्विसेज', 'ऑटो', 'शोरूम', 'शोरमजवळ', 'ट्रेडिंग'
        ]

        manufacturers = ['mahindra', 'kubota', 'escorts', 'sonalika', 'swaraj', 'eicher', 'farmtrac', 'powertrac', 'tafe', 'massey ferguson', 'john deere', 'massey']


        # Phrases that are clearly not dealer names
        exclude_words = [
            'gstin', 'gst', 'address', 'mobile', 'phone', 'email', 'invoice', 'quotation', 'quote', 'date',
            'son/d/w', 'finance', 'village', 'tehsil', 'district', 'customer',
            'total', 'grand total', 'amount', 'rupees', 'in words', 'price', 'ex showroom', 'showroom price',
            'terms & condition', 'terms and condition', 'bank', 'ifsc', 'a/c no',
            'idfc', 'hdfc', 'icici', 'axis', 'sbi', 'idbi', 'bob', 'punjab national', 'kotak',
            'प्रति', 'सेवा', 'पत्ता', 'दिनांक', 'नगर', 'राज्य' # Hindi stop words
        ]

        candidates = []

        for block in blocks:
            text = block.get('text', '').strip()
            # print(f"DEBUG: Checking block: '{text}'")
            if len(text) < 4 or len(text) > 100:
                continue

            text_lower = text.lower()
            # print(f"DEBUG: Text lower: '{text_lower}'")

            # Basic character statistics
            letters = re.findall(r'[A-Za-z\u0900-\u097F]', text_lower)
            digits = re.findall(r'\d', text_lower)

            # Pure numeric or almost-numeric lines should never be dealer names
            if not letters:
                continue
            if digits and len(digits) >= len(letters) and not any(kw in text_lower for kw in dealer_keywords):
                # e.g. "550000.00", "Grand Total: 550000.00"
                continue

            # 4. ROBUST BANK EXCLUSION: If it's clearly bank/finance info, drop it immediately
            banks = ['bank', 'idfc', 'idsc', 'hdfc', 'icici', 'axis', 'sbi', 'idbi', 'bob', 'pnb', 'kotak', 'finance', 'fin.', 'bak', 'banc']
            if any(re.search(rf'\b{b}\b', text_lower) for b in banks):
                continue

            # NEW: Reject generic "Authorised Dealer" lines that lack a specific business name
            generic_titles = ['authorised dealer', 'authorized dealer', 'authorised distributor', 'showroom', 'sales & service']
            is_generic = False
            for title in generic_titles:
                if title in text_lower:
                    # If the whole line is just "Authorised Dealer Mahidra Tractors", it's generic
                    clean_test = re.sub(rf'{title}|mahindra|tractors|motors', '', text_lower).strip()
                    if len(clean_test) < 3:
                        is_generic = True
                        break
            if is_generic: continue

            # Calculate font prominence (average bbox height)
            heights = []
            for el in block.get('elements', []):
                bbox = el.get('bbox', [0, 0, 0, 0])
                heights.append(bbox[3] - bbox[1])
            avg_height = sum(heights) / len(heights) if heights else 0
            
            # 1. Prominence score (larger fonts = likely headers/dealer names)
            # Normal font size is around 20-30px, headers are 40-80px
            prominence_score = min(150, avg_height * 2)

            # 2. Layout score (prefer top 25% of page for headers)
            y_pos = block.get('y_position', 0)
            layout_score = 0
            if y_pos < 500:
                layout_score = 100
            elif y_pos < 1000:
                layout_score = 50
            elif y_pos > 3000:
                # Potential footer dealer info
                layout_score = 40

            # 3. Keyword match score
            keyword_score = 0
            for kw in dealer_keywords:
                if kw in text_lower:
                    keyword_score += 80

            total_score = prominence_score + layout_score + keyword_score

            # 4. Clean and validate candidate
            clean_name = re.sub(r'^(M/S|M/s|Dealer|Auth\.|From|To|Name)\s*:?\s*', '', text, flags=re.IGNORECASE)
            # Remove GSTIN-like tokens so they don't pollute name
            clean_name = re.sub(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{3}\b', '', clean_name).strip()
            # Remove "GSTIN No:" or similar prefixes if present
            clean_name = re.sub(r'(GSTIN|GST)\s*(No\.?)?\s*:?\s*', '', clean_name, flags=re.IGNORECASE).strip()
            # If we have a label like "Dealer Name: Krishna Traders", keep the RHS
            if ':' in clean_name:
                label, value_part = clean_name.split(':', 1)
                if re.search(r'(dealer|name|m/s)', label, re.IGNORECASE):
                    clean_name = value_part
            clean_name = re.split(r'[,\n]', clean_name)[0].strip()

            is_valid = True
            if self._validator:
                is_valid, _, _ = self._validator.validate_dealer_name(clean_name)

            # If validator rejected but we see a strong dealer keyword or large font, give it a chance
            if not is_valid:
                if any(kw in text_lower for kw in dealer_keywords) or avg_height > 40:
                    is_valid = True

            # Hard-filter obvious non-name lines even if validator passes
            lower_clean = clean_name.lower()
            non_name_cues = ['tehsil', 'district', 'village', 'address', 'mobile', 'phone', 'email', 'son/d/w', 'invoice', 'quotation', 'price', 'total', 'bank']
            if any(w in lower_clean for w in non_name_cues):
                is_valid = False
            
            # Filter matches that look like model specifications (e.g. "1. Swaraj 855")
            if re.search(r'^\d+[\.\s]+', clean_name) and any(m in lower_clean for m in manufacturers):
                is_valid = False

            if is_valid:
                script, _ = detect_script(clean_name)
                # print(f"DEBUG: Found candidate: {clean_name} | Score: {total_score}")
                candidates.append({
                    'text': clean_name,
                    'score': total_score,
                    'y': y_pos,
                    'script': script,
                    'original': text,
                    'height': avg_height
                })
            # else:
            #     print(f"DEBUG: Candidate REJECTED: {clean_name}")

        if not candidates:
            return None

        # Sort by score (higher is better), then prefer higher-up text
        candidates.sort(key=lambda x: (-x['score'], x['y']))
        best = candidates[0]
        name = best['text']

        # Final cleanup: strip any GSTIN-like tokens and single trailing chars
        name = re.sub(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{3}\b', '', name).strip()

        # Remove common non-name prefixes left over (GSTIN No:, Address:, etc.)
        name = re.sub(r'^(GSTIN|GST)\s*(No\.?)?\s*:?\s*', '', name, flags=re.IGNORECASE).strip()

        # If line still has a lot of header noise (GSTIN, company name, etc.),
        # keep only the span around the main business keyword (e.g., "Krishna Traders").
        lower_name = name.lower()
        business_tail_keywords = [
            'tractors', 'motors', 'traders', 'agency', 'enterprises', 'auto', 'sales', 
            'spares', 'borwells', 'शोरूम', 'एजेंसी', 'ट्रेडर्स', 'शोरमजवळ', 'सर्विस'
        ]
        
        # Improvement: Be less aggressive with pruning, keep more context if it looks like a proper name
        for kw in business_tail_keywords:
            if kw in lower_name:
                words = name.split()
                # Find last word containing the keyword
                for i in range(len(words) - 1, -1, -1):
                    if kw in words[i].lower():
                        # Gather preceding words (increased limit to 6 for better context)
                        preceding = []
                        for j in range(i - 1, -1, -1):
                            w = words[j]
                            # Skip common noise but keep conjunctions
                            if w.lower() in ('or', 'and', 'the', 'of', '&'):
                                preceding.insert(0, w)
                                continue
                            if w.lower() in ('no', 'no.', 'limited', 'limted', 'ltd', 'pvt', 'llp'):
                                continue
                            
                            # Keep words if they look like part of a name
                            if re.match(r'^[A-Z\u0900-\u097F]', w) or len(w) > 3:
                                preceding.insert(0, w)
                            
                            if len(preceding) >= 6:
                                break
                        name = ' '.join(preceding + words[i:]) # Keep everything after keyword too
                        lower_name = name.lower()
                        break
                break

        # Remove leading/trailing noise and single chars
        name = re.sub(r'^(stLmed|stmed|estd|pref|name|m/s|dealer)\s+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^[A-Za-z]\s+', '', name)
        name = re.sub(r'\s+[A-Za-z0-9]$', '', name)
        name = name.strip()

        # Final Validation: If it still contains "Total" or "Amount", it's likely a misidentification
        # Use word boundaries to avoid matching "Tractors" or "Motors" as "rs" or "ors"
        # Truncate common tails that pollute dealer name
        for tail in ['total', 'amount', 'rupees', 'price', 'rs', 'performa bill', 'proforma invoice', 'invoice no']:
            match = re.search(rf'\s+{tail}', name, re.IGNORECASE)
            if match:
                name = name[:match.start()].strip()

        # Truncate phone numbers (e.g. 98153-64031)
        name = re.sub(r'[\d\-]{8,}', '', name).strip()
        
        if any(re.search(rf'\b{w}\b', name.lower()) for w in ['total', 'amount', 'rupees', 'price']):
            return None
            
        # Noise check on original text first
        noise = self._calculate_noise_score(name)
        if '₹' in name or noise > 0.4:
            return None

        # Sanitize final name
        name = self._sanitize_text(name)
        if len(name) < 4: return None

        lang_code = get_language_code(best['script'])

        return ExtractedField(
            field_name='dealer_name',
            value=name,
            confidence=min(0.95, best['score'] / 300),
            source_text=best['original'],
            extraction_method='keyword_first_matching_v3',
            language=lang_code,
            language_confidence=0.9 if lang_code != 'en' else 0.0
        )

        
        return None
    
    def _extract_model_name(self, full_text: str, blocks: List[Dict],
                            config: Dict) -> Optional[ExtractedField]:
        """Extract tractor/vehicle model name using labels and layout."""
        model_labels = [
            'model', 'variant', 'description', 'particulars', 'item', 'product',
            'मॉडल', 'वेरिएंट', 'विवरण', 'तपशील', 'आइटम'  # Hindi/Marathi labels
        ]
        
        known_brands = config.get('known_brands', [
            'Mahindra', 'Sonalika', 'John Deere', 'Kubota', 'Eicher',
            'Swaraj', 'New Holland', 'Massey Ferguson', 'TAFE', 'Escorts',
            'Farmtrac', 'Powertrac', 'Force', 'Preet', 'Digitrac', 'Captain'
        ])
        
        candidates = []
        
        # 1. Label-Value Proximity Search (Spatial)
        for block in blocks:
            text = block.get('text', '')
            text_lower = text.lower()
            
            # Look for model labels
            for label in model_labels:
                if label in text_lower:
                    # Look for value in same block (e.g. "Model: Farmtrac 45")
                    parts = re.split(rf'{label}\s*[:\-]?\s*', text, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        val = parts[1].strip()
                        # Clean value (take only first line/segment)
                        val = re.split(r'[,\n]', val)[0].strip()
                        if len(val) > 2:
                            candidates.append((val, 0.95, text))
                    
                    # Also look in NEXT block (common in tables)
                    idx = -1
                    for i, b in enumerate(blocks):
                        if b == block:
                            idx = i
                            break
                    
                    if idx != -1 and idx + 1 < len(blocks):
                        next_text = blocks[idx+1].get('text', '').strip()
                        # If next text is very short or looks like a code, it's a good candidate
                        if next_text and len(next_text) > 2 and len(next_text) < 40:
                            candidates.append((next_text, 0.9, next_text))

        # 2. Brand + Pattern
        for brand in known_brands:
            brand_pat = brand.replace(' ', r'\s*')
            # Look for brand followed by alphanumeric model code
            pattern = rf'({brand_pat}[\s\-]+[A-Z0-9]+[\s\-]*(?:DI|HP|XP|FE|MS|Plus|Power|Pro|Max|Super)?[\s\-]*(?:\d+)?[\s\-]*(?:HP)?)'
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                clean = ' '.join(match.split())
                if len(clean) > len(brand) + 1:
                    candidates.append((clean, 0.9, match))

        # 3. Enhanced Alphanumeric Sequence Search
        # Common patterns: "855 FE", "575 DI", "XP 330", "M-604", "41.5 HP"
        general_patterns = [
            r'\b(\d{3}\s*(?:DI|XP|FE|MS|Plus|Power|Pro|Max|Super|HP))\b',
            r'\b([A-Z]{1,3}\s*\d{3})\b',
            r'\b(\d{2,4}[A-Z]{1,2})\b',
            r'\b(NOVO\s*\d{3}\s*[A-Z]{0,2})\b' # Specific for Mahindra Novo
        ]
        for pattern in general_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                candidates.append((match, 0.75, match))

        if candidates:
            # Dedupe and score
            unique = []
            seen = set()
            for val, conf, source in candidates:
                c_clean = val.lower().strip()
                if c_clean not in seen:
                    if len(c_clean) < 3 or c_clean in model_labels: continue
                    seen.add(c_clean)
                    unique.append({'value': val, 'conf': conf, 'source': source})
            
            def score_candidate(c):
                score = c['conf']
                text = c['value'].lower()
                # Bonus for numeric content
                if re.search(r'\d', text): score += 0.1
                # Bonus for brand names
                if any(brand.lower() in text for brand in known_brands): score += 0.1
                # Penalty for noise
                if any(w in text for w in ['total', 'price', 'amt', 'rupees', 'address', 'tel']): score -= 0.6
                if len(text) < 4: score -= 0.2
                return score

            unique.sort(key=score_candidate, reverse=True)
            best = unique[0]
            
            # Clean final value
            final_val = best['value']
            # Strip trailing punctuation
            final_val = re.sub(r'[:,\-]$', '', final_val).strip()
            
            # Penalize noise based on original string quality
            noise = self._calculate_noise_score(final_val)
            
            # Advanced sanitization for model name (remove OCR dots)
            final_val = self._sanitize_text(final_val)

            # Truncate if we see other field keywords on the same line
            for field_tail in [r'\bHP\b', r'\bCOST\b', r'\bVARIANT\b', r'\bCOLOUR\b', r'\bCOLOR\b', r'\bCHASSIS\b', r'\b\d{5,}\b']:
                match = re.search(field_tail, final_val, re.IGNORECASE)
                if match:
                    final_val = final_val[:match.start()].strip()
            
            # Heavily penalize high noise
            penalty = 1.0 - (noise * 1.5) # Stronger penalty
            final_confidence = max(0.0, min(0.98, score_candidate(best) * penalty))
            
            if len(final_val) < 3 or noise > 0.5:
                # print(f"DEBUG REJECT: Model {final_val} noise {noise}")
                return None
            
            # Reject if it's just a bunch of numbers without any letters (likely a serial/invoice no)
            if re.match(r'^[\d\s\.\-]+$', final_val) and not any(b.lower() in final_val.lower() for b in known_brands):
                return None
            
            return ExtractedField(
                field_name='model_name',
                value=final_val,
                confidence=final_confidence,
                source_text=best['source'],
                extraction_method='improved_spatial_pattern'
            )
        
        return None
    
    def add_visual_detection(self, result: ExtractionResult, 
                             signature_data: Optional[Dict],
                             stamp_data: Optional[Dict]):
        """Add visual detection results (signature/stamp) to extraction result."""
        if signature_data:
            result.fields['signature'] = ExtractedField(
                field_name='signature',
                value=signature_data.get('present', False),
                confidence=signature_data.get('confidence', 0.0),
                bbox=signature_data.get('bbox'),
                extraction_method='yolo'
            )
        
        if stamp_data:
            result.fields['stamp'] = ExtractedField(
                field_name='stamp',
                value=stamp_data.get('present', False),
                confidence=stamp_data.get('confidence', 0.0),
                bbox=stamp_data.get('bbox'),
                extraction_method='yolo'
            )
        
        # Recalculate overall confidence
        if result.fields:
            confidences = [f.confidence for f in result.fields.values()]
            result.overall_confidence = sum(confidences) / len(confidences)
