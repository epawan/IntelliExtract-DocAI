"""
Language Detection Module
Detects language of extracted text (English, Hindi, Gujarati, etc.)
Provides language-aware processing for multilingual invoices.
"""

from typing import Dict, List, Tuple, Optional
from langdetect import detect, detect_langs, LangDetectException
import re


class LanguageDetector:
    """Detects language of text and provides language-aware processing."""
    
    # Language code mappings
    LANGUAGE_NAMES = {
        'en': 'English',
        'hi': 'Hindi',
        'gu': 'Gujarati',
        'mr': 'Marathi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
    }
    
    # Script patterns for Indian languages
    SCRIPT_PATTERNS = {
        'hi': r'[\u0900-\u097F]',  # Devanagari (Hindi)
        'gu': r'[\u0A80-\u0AFF]',  # Gujarati
        'mr': r'[\u0900-\u097F]',  # Devanagari (Marathi)
        'ta': r'[\u0B80-\u0BFF]',  # Tamil
        'te': r'[\u0C60-\u0C7F]',  # Telugu
        'kn': r'[\u0C80-\u0CFF]',  # Kannada
        'ml': r'[\u0D00-\u0D7F]',  # Malayalam
        'pa': r'[\u0A00-\u0A7F]',  # Punjabi
    }
    
    @staticmethod
    def detect_language(text: str) -> Tuple[str, float]:
        """
        Detect language of text.
        Optimized for invoice extraction with short text segments.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or len(text.strip()) < 2:
            return 'en', 0.0
        
        # First: Check for Indian script patterns (more reliable for short text)
        for lang_code, pattern in LanguageDetector.SCRIPT_PATTERNS.items():
            if re.search(pattern, text):
                return lang_code, 0.95  # High confidence if script detected
        
        # Second: Try langdetect for longer text (>20 chars)
        if len(text) > 20:
            try:
                probs = detect_langs(text)
                if probs:
                    lang_code = probs[0].lang
                    confidence = probs[0].prob
                    # Map common misdetections for short English text
                    if lang_code in ['sw', 'de', 'id', 'sq'] and len(text) < 50:
                        return 'en', 0.6
                    return lang_code, confidence
            except LangDetectException:
                pass
        
        # Default: If mostly alphanumeric/English characters, assume English
        if re.match(r'^[a-zA-Z0-9\s\.\,\-\_:]+$', text):
            return 'en', 0.7
        
        # Fallback
        return 'en', 0.5
    
    @staticmethod
    def detect_languages_mixed(text: str) -> List[Dict[str, any]]:
        """
        Detect multiple languages in mixed-language text.
        
        Args:
            text: Input text (possibly multilingual)
            
        Returns:
            List of detected languages with confidence scores
        """
        detected = []
        
        # Split by common delimiters and detect each segment
        segments = re.split(r'[\s\-\,\.]', text)
        
        lang_confidence = {}
        for segment in segments:
            if len(segment) < 2:
                continue
            
            lang_code, conf = LanguageDetector.detect_language(segment)
            if lang_code not in lang_confidence:
                lang_confidence[lang_code] = []
            lang_confidence[lang_code].append(conf)
        
        # Aggregate by language
        for lang_code, confidences in lang_confidence.items():
            avg_conf = sum(confidences) / len(confidences)
            detected.append({
                'language': lang_code,
                'language_name': LanguageDetector.LANGUAGE_NAMES.get(lang_code, lang_code),
                'confidence': round(avg_conf, 2),
                'occurrences': len(confidences)
            })
        
        # Sort by confidence
        return sorted(detected, key=lambda x: x['confidence'], reverse=True)
    
    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """Get full language name from code."""
        return LanguageDetector.LANGUAGE_NAMES.get(lang_code, lang_code)
    
    @staticmethod
    def is_multilingual(text: str) -> bool:
        """Check if text contains multiple languages."""
        languages = LanguageDetector.detect_languages_mixed(text)
        return len(languages) > 1
    
    @staticmethod
    def get_dominant_language(text: str) -> str:
        """Get the dominant language in text."""
        language, _ = LanguageDetector.detect_language(text)
        return language
