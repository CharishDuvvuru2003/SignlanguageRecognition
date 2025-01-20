# File path: slr/utils/translation.py

# English to Hindi character mapping dictionary
ENGLISH_TO_HINDI = {
    'A': 'अ', 'B': 'ब', 'C': 'क', 'D': 'ड', 'E': 'ए',
    'F': 'फ', 'G': 'ग', 'H': 'ह', 'I': 'इ', 'J': 'ज',
    'K': 'क', 'L': 'ल', 'M': 'म', 'N': 'न', 'O': 'ओ',
    'P': 'प', 'Q': 'क्यू', 'R': 'र', 'S': 'स', 'T': 'ट',
    'U': 'उ', 'V': 'व', 'W': 'व', 'X': 'एक्स', 'Y': 'य',
    'Z': 'ज़', ' ': ' '
}

def translate_to_hindi(text):
    """Translate English text to Hindi using character mapping"""
    if not text:  # Skip empty text
        return ""
    return ''.join(ENGLISH_TO_HINDI.get(c.upper(), c) for c in text)