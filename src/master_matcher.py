"""
Master File Matcher
Performs fuzzy matching for dealer names (≥90%) and exact matching for model names.
"""

import yaml
from pathlib import Path
from typing import Optional, Tuple, List
from rapidfuzz import fuzz, process


class MasterFileMatcher:
    """
    Matches extracted fields against master files.
    - Dealer Name: Fuzzy matching (≥90% similarity)
    - Model Name: Exact matching
    """
    
    def __init__(self, dealer_master_path: str = None, model_master_path: str = None):
        """
        Initialize matcher with master files.
        
        Args:
            dealer_master_path: Path to dealer master YAML
            model_master_path: Path to model master YAML
        """
        config_dir = Path(__file__).parent.parent / 'config'
        
        self.dealer_master_path = dealer_master_path or str(config_dir / 'dealer_master.yaml')
        self.model_master_path = model_master_path or str(config_dir / 'model_master.yaml')
        
        self.dealers = self._load_dealers()
        self.models = self._load_models()
    
    def _load_dealers(self) -> List[dict]:
        """Load dealer master file."""
        try:
            with open(self.dealer_master_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('dealers', [])
        except Exception as e:
            print(f"Warning: Could not load dealer master: {e}")
            return []
    
    def _load_models(self) -> List[str]:
        """Load model master file and flatten to list."""
        try:
            with open(self.model_master_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                brands = data.get('models', [])
                
                # Flatten all models into single list
                all_models = []
                for brand_data in brands:
                    all_models.extend(brand_data.get('models', []))
                
                return all_models
        except Exception as e:
            print(f"Warning: Could not load model master: {e}")
            return []
    
    def match_dealer(self, extracted_name: str, threshold: float = 90.0) -> Tuple[Optional[str], float]:
        """
        Fuzzy match dealer name against master file.
        
        Args:
            extracted_name: Dealer name extracted from document
            threshold: Minimum similarity score (default 90%)
            
        Returns:
            Tuple of (matched_name, confidence_score)
        """
        if not extracted_name or not self.dealers:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        # Build list of all dealer names and variations
        dealer_options = []
        dealer_map = {}  # Maps variation to canonical name
        
        for dealer in self.dealers:
            canonical_name = dealer['name']
            dealer_options.append(canonical_name)
            dealer_map[canonical_name] = canonical_name
            
            # Add variations
            for variation in dealer.get('variations', []):
                dealer_options.append(variation)
                dealer_map[variation] = canonical_name
        
        # Fuzzy match using rapidfuzz
        result = process.extractOne(
            extracted_name,
            dealer_options,
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )
        
        if result:
            matched_variation, score, _ = result
            canonical_name = dealer_map.get(matched_variation, matched_variation)
            return canonical_name, score / 100.0  # Convert to 0-1 scale
        
        return None, 0.0
    
    def match_model(self, extracted_model: str) -> Tuple[Optional[str], float]:
        """
        Exact match model name against master file.
        
        Args:
            extracted_model: Model name extracted from document
            
        Returns:
            Tuple of (matched_name, confidence_score)
            confidence is 1.0 for exact match, 0.0 otherwise
        """
        if not extracted_model or not self.models:
            return None, 0.0
        
        # Try exact match (case-insensitive)
        extracted_upper = extracted_model.upper().strip()
        
        for model in self.models:
            if model.upper().strip() == extracted_upper:
                return model, 1.0  # Exact match
        
        # Try fuzzy match as fallback (≥95% for models)
        result = process.extractOne(
            extracted_model,
            self.models,
            scorer=fuzz.ratio,
            score_cutoff=95.0
        )
        
        if result:
            matched_model, score, _ = result
            return matched_model, score / 100.0
        
        return None, 0.0
    
    def validate_and_match(self, extraction_result: dict) -> dict:
        """
        Validate and match all fields against master files.
        
        Args:
            extraction_result: Dictionary with extracted fields
            
        Returns:
            Updated dictionary with matched fields and confidences
        """
        result = extraction_result.copy()
        
        # Match dealer name (fuzzy, ≥90%)
        if 'dealer_name' in result['fields']:
            extracted_dealer = result['fields']['dealer_name']
            matched_dealer, dealer_conf = self.match_dealer(extracted_dealer)
            
            if matched_dealer:
                result['fields']['dealer_name'] = matched_dealer
                result['fields']['dealer_match_confidence'] = dealer_conf
                print(f"  Dealer matched: '{extracted_dealer}' → '{matched_dealer}' ({dealer_conf:.2%})")
            else:
                result['fields']['dealer_match_confidence'] = 0.0
                print(f"  Dealer not matched: '{extracted_dealer}'")
        
        # Match model name (exact)
        if 'model_name' in result['fields']:
            extracted_model = result['fields']['model_name']
            matched_model, model_conf = self.match_model(extracted_model)
            
            if matched_model:
                result['fields']['model_name'] = matched_model
                result['fields']['model_match_confidence'] = model_conf
                print(f"  Model matched: '{extracted_model}' → '{matched_model}' ({model_conf:.2%})")
            else:
                result['fields']['model_match_confidence'] = 0.0
                print(f"  Model not matched: '{extracted_model}'")
        
        return result


# Export
__all__ = ['MasterFileMatcher']
