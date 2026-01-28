#!/usr/bin/env python3
"""
Hindi Devanagari Medical Sentence Noisy Data Generator

This script generates training pairs of (noisy, clean) Hindi medical sentences.
The noise patterns are designed to simulate real ASR (Automatic Speech Recognition) errors.

Noise patterns include:
1. Character substitution (phonetically similar characters)
2. Missing/wrong matras (vowel signs)
3. Character deletion/insertion
4. Word-level distortions
5. Spacing issues (merging/splitting words)
"""

import random
import csv
import json
from typing import List, Tuple, Dict
from collections import defaultdict
import re

# Seed for reproducibility
random.seed(42)

# ==============================================================================
# Hindi Medical Vocabulary and Sentence Templates
# ==============================================================================

# Body parts in Hindi
BODY_PARTS = [
    "सिर", "गर्दन", "कंधा", "बाँह", "हाथ", "उँगली", "कलाई", "छाती", "पेट", "कमर",
    "पीठ", "पैर", "घुटना", "टखना", "एड़ी", "पंजा", "नाक", "कान", "आँख", "मुँह",
    "गला", "जीभ", "दाँत", "होंठ", "नाखून", "त्वचा", "बाल", "दिमाग", "दिल", "फेफड़े",
    "जिगर", "गुर्दा", "हड्डी", "मांसपेशी", "नस", "रीढ़", "जोड़", "पिंडली", "जाँघ",
    "नथुना", "पलक", "भौं", "गाल", "ठोड़ी", "माथा", "कोहनी", "कूल्हा", "पसली"
]

# Medical symptoms in Hindi
SYMPTOMS = [
    "दर्द", "सूजन", "जलन", "खुजली", "बुखार", "कमजोरी", "थकान", "चक्कर", "मतली",
    "उल्टी", "सिरदर्द", "खांसी", "जुकाम", "छींक", "बलगम", "दस्त", "कब्ज", "गैस",
    "एसिडिटी", "भूख न लगना", "नींद न आना", "घबराहट", "बेचैनी", "सुन्नपन",
    "झनझनाहट", "जकड़न", "खिंचाव", "ऐंठन", "धड़कन", "सांस फूलना", "खून आना",
    "पसीना", "ठंड लगना", "गर्मी लगना", "चुभन", "फोड़ा", "घाव", "कट", "चोट",
    "फ्रैक्चर", "मोच", "संक्रमण", "एलर्जी", "रैश", "दाने", "छाले", "नकसीर"
]

# Medical conditions in Hindi
CONDITIONS = [
    "डायबिटीज़", "बीपी", "थायरॉयड", "अस्थमा", "माइग्रेन", "आर्थराइटिस", "एनीमिया",
    "डिप्रेशन", "एंग्जायटी", "इंफेक्शन", "वायरल", "बैक्टीरियल", "फंगल", "यूटीआई",
    "एसिडिटी", "अल्सर", "कोलेस्ट्रॉल", "हृदय रोग", "किडनी स्टोन", "गॉलब्लैडर",
    "स्लिप्ड डिस्क", "साइनस", "टॉन्सिल", "पाइल्स", "फिशर", "हर्निया"
]

# Time expressions
TIME_EXPRESSIONS = [
    "सुबह", "शाम", "रात", "दोपहर", "कल", "परसों", "पिछले हफ्ते", "पिछले महीने",
    "कई दिनों से", "लंबे समय से", "अचानक", "धीरे-धीरे", "कभी-कभी", "अक्सर",
    "हमेशा", "रोज़", "हर दिन", "कुछ दिनों से", "एक हफ्ते से", "दो महीने से"
]

# Intensity words
INTENSITY = [
    "बहुत", "थोड़ा", "हल्का", "तेज़", "असहनीय", "लगातार", "रुक-रुक कर",
    "कम", "ज़्यादा", "बेहद", "काफी", "अत्यधिक"
]

# Verbs related to symptoms
SYMPTOM_VERBS = [
    "होता है", "होती है", "रहता है", "रहती है", "लगता है", "लगती है",
    "आता है", "आती है", "महसूस होता है", "महसूस होती है", "बढ़ जाता है",
    "बढ़ जाती है", "कम हो जाता है", "कम हो जाती है", "शुरू होता है",
    "शुरू होती है", "बना रहता है", "बनी रहती है"
]

# Sentence templates for medical sentences
SENTENCE_TEMPLATES = [
    "मुझे {body_part} में {symptom} {verb}",
    "{time} से {body_part} में {intensity} {symptom} {verb}",
    "मेरे {body_part} में {symptom} और {symptom2} दोनों {verb}",
    "{body_part} हिलाने पर {symptom} {verb}",
    "खाना खाने के बाद {symptom} {verb}",
    "रात को सोते समय {body_part} में {symptom} {verb}",
    "चलने पर {body_part} में {symptom} {verb}",
    "मुझे {condition} है और {body_part} में {symptom} {verb}",
    "{intensity} {symptom} के साथ {symptom2} भी {verb}",
    "{body_part} के आसपास {symptom} और {intensity} {symptom2} {verb}",
    "सुबह उठते ही {body_part} में {symptom} {verb}",
    "{time} {body_part} में {symptom} होने लगती है",
    "दवा लेने के बाद भी {symptom} {verb}",
    "{body_part} को छूने पर {intensity} {symptom} {verb}",
    "गर्मी में {body_part} से {symptom} {verb}",
    "ठंड में {body_part} में {symptom} {verb}",
    "व्यायाम करने पर {symptom} {verb}",
    "{body_part} में सूजन है और {symptom} भी {verb}",
    "डॉक्टर ने कहा {condition} है",
    "जाँच में {condition} निकला है",
    "परिवार में {condition} की समस्या रही है",
    "{body_part} में {symptom} के कारण {symptom2} भी {verb}",
    "झुकने पर {body_part} में {symptom} {verb}",
    "बैठने के बाद {body_part} में {symptom} {verb}",
    "{body_part} की त्वचा पर {symptom} {verb}",
]

# Additional complete sentences for variety
COMPLETE_SENTENCES = [
    "मुझे बार-बार मतली जैसी महसूस होती है।",
    "सांस लेने में तकलीफ होती है।",
    "भूख बिल्कुल नहीं लगती है।",
    "नींद में खलल पड़ रही है।",
    "वजन अचानक बढ़ गया है।",
    "वजन अचानक कम हो गया है।",
    "चेहरे पर दाने निकल आए हैं।",
    "आँखों में जलन और लालिमा है।",
    "कान में आवाज़ आती है।",
    "सुनने में दिक्कत हो रही है।",
    "याददाश्त कमजोर हो गई है।",
    "ध्यान केंद्रित नहीं कर पाती हूँ।",
    "बेवजह रोना आता है।",
    "चिड़चिड़ापन बढ़ गया है।",
    "खाना निगलने में दिक्कत है।",
    "गला बैठा हुआ है।",
    "आवाज़ नहीं निकल रही।",
    "पेशाब में जलन होती है।",
    "पेशाब बार-बार आता है।",
    "पेशाब का रंग बदल गया है।",
    "मासिक धर्म अनियमित है।",
    "पीरियड्स में बहुत दर्द होता है।",
    "पीरियड्स के दौरान थकान रहती है।",
    "शरीर में दर्द रहता है।",
    "जोड़ों में अकड़न है।",
    "मांसपेशियों में खिंचाव है।",
    "हाथ-पैर सुन्न हो जाते हैं।",
    "उँगलियों में झनझनाहट होती है।",
    "नाक से खून आता है।",
    "नाक बंद रहती है।",
    "गले में खराश है।",
    "खाँसी में खून आता है।",
    "छाती में दबाव महसूस होता है।",
    "दिल की धड़कन तेज़ हो जाती है।",
    "पैरों में सूजन आ गई है।",
    "त्वचा पर खुजली होती है।",
    "त्वचा सूखी और खुरदरी हो गई है।",
    "बाल झड़ रहे हैं।",
    "सिर में रूसी हो गई है।",
    "नाखून टूट रहे हैं।",
    "होंठ फट रहे हैं।",
    "मुँह में छाले हो गए हैं।",
    "दाँत में दर्द है।",
    "मसूड़ों से खून आता है।",
    "जीभ पर सफेद परत है।",
    "पेट में गैस बनती है।",
    "खट्टी डकार आती है।",
    "पेट फूला रहता है।",
    "कब्ज़ की समस्या है।",
    "दस्त लग गए हैं।",
    "शौच में खून आता है।",
    "बवासीर की समस्या है।",
    "फिशर हो गया है।",
    "कमर में दर्द रहता है।",
    "पीठ में अकड़न है।",
    "गर्दन घुमाने में दिक्कत है।",
    "कंधे में दर्द है।",
    "कोहनी में सूजन है।",
    "कलाई में मोच आ गई है।",
    "उँगली में चोट लगी है।",
    "पैर की अंगुली टूट गई है।",
    "घुटने में पानी भर गया है।",
    "टखने में मोच है।",
    "एड़ी में दर्द है।",
    "तलवे में जलन होती है।",
]

# ==============================================================================
# Noise Generation Functions
# ==============================================================================

# Devanagari character mappings for phonetically similar sounds
CHAR_SUBSTITUTIONS = {
    # Consonant confusions (common ASR errors)
    'क': ['ख', 'ग', 'क़'],
    'ख': ['क', 'ग', 'ख़'],
    'ग': ['क', 'ख', 'घ', 'ग़'],
    'घ': ['ग', 'ध'],
    'च': ['छ', 'ज'],
    'छ': ['च', 'श'],
    'ज': ['च', 'झ', 'ज़'],
    'झ': ['ज', 'ष'],
    'ट': ['ठ', 'त', 'ड'],
    'ठ': ['ट', 'थ', 'ड'],
    'ड': ['ट', 'ठ', 'ढ', 'द'],
    'ढ': ['ड', 'ध'],
    'त': ['ट', 'थ', 'द'],
    'थ': ['त', 'ठ', 'ध'],
    'द': ['त', 'ड', 'ध'],
    'ध': ['द', 'ढ', 'घ'],
    'न': ['ण', 'म'],
    'ण': ['न', 'ञ'],
    'प': ['फ', 'ब'],
    'फ': ['प', 'भ', 'फ़'],
    'ब': ['प', 'भ', 'व'],
    'भ': ['ब', 'फ', 'म'],
    'म': ['न', 'भ'],
    'य': ['ज', 'ई'],
    'र': ['ल', 'ड़'],
    'ल': ['र', 'ळ'],
    'व': ['ब', 'भ'],
    'श': ['ष', 'स', 'छ'],
    'ष': ['श', 'स'],
    'स': ['श', 'ष', 'छ'],
    'ह': ['अ', 'ग'],
    
    # Nukta characters
    'ज़': ['ज', 'झ'],
    'फ़': ['फ', 'प'],
    'क़': ['क', 'ख'],
    'ख़': ['ख', 'क'],
    'ग़': ['ग', 'घ'],
    'ड़': ['ड', 'र'],
    'ढ़': ['ढ', 'ध'],
}

# Matra (vowel sign) confusions
MATRA_SUBSTITUTIONS = {
    'ा': ['े', 'ो', ''],  # aa -> e, o, or dropped
    'ि': ['ी', 'े', ''],  # i -> ee, e, or dropped
    'ी': ['ि', 'े', ''],  # ee -> i, e, or dropped
    'ु': ['ू', 'ो', ''],  # u -> oo, o, or dropped
    'ू': ['ु', 'ो', ''],  # oo -> u, o, or dropped
    'े': ['ै', 'ि', 'ा'],  # e -> ai, i, aa
    'ै': ['े', 'ा', 'ि'],  # ai -> e, aa, i
    'ो': ['ौ', 'ा', 'ु'],  # o -> au, aa, u
    'ौ': ['ो', 'ा', 'ु'],  # au -> o, aa, u
    'ं': ['न', 'म', ''],  # anusvara -> n, m, or dropped
    'ँ': ['ं', ''],  # chandrabindu -> anusvara or dropped
    '्': ['', 'अ'],  # halant -> dropped or schwa
}

# Common word-level substitutions (ASR confusions)
WORD_SUBSTITUTIONS = {
    'पेट': ['पेन', 'पेत', 'पैट'],
    'दर्द': ['दर्द', 'दरद', 'डर्ड', 'दर्ड'],
    'सूजन': ['सूज', 'सजन', 'सून'],
    'बुखार': ['बुखा', 'बुखाद', 'बखार'],
    'खांसी': ['खासी', 'कासी', 'खाशी', 'खांबी'],
    'जुकाम': ['जुकाप', 'जुकाव', 'जकाम'],
    'सिरदर्द': ['सिरर्द', 'सरदर्द', 'सिरद'],
    'थकान': ['थनका', 'तकान', 'थकन'],
    'कमजोरी': ['कमरीजो', 'कमजोर', 'करीजोम'],
    'चक्कर': ['चकर', 'चक्', 'चक्के', 'चक्ते'],
    'घबराहट': ['घहटबरा', 'घबरा', 'घबराट'],
    'धड़कन': ['कनधड़', 'धकन', 'धड़न'],
    'सांस': ['सास', 'सस', 'सां'],
    'नींद': ['नींर', 'नीं', 'नींत', 'नींन'],
    'भूख': ['वीख', 'भख', 'भूक'],
    'पानी': ['पान', 'पनी', 'पाणी'],
    'खाना': ['खन', 'खाण', 'कना'],
    'दवाई': ['दबाई', 'दवई', 'दाई'],
    'डॉक्टर': ['डाक्टर', 'डॉटर', 'डाटर'],
    'अस्पताल': ['असपताल', 'अस्पतल', 'स्पताल'],
    'थायरॉयड': ['थाडरॉय', 'थराइड', 'थायडरॉ'],
    'डायबिटीज़': ['डायटीबिज़', 'डयबिटीज', 'डाइटीज'],
    'बीपी': ['बपी', 'वीपी', 'बीती'],
    'इंफेक्शन': ['इंफेशन', 'इंफेक्सन', 'इंफेशक'],
    'एलर्जी': ['एलजी', 'अलर्जी', 'एलरी'],
    'सर्जरी': ['सर्जर', 'सरी', 'सर्री'],
    'ऑपरेशन': ['ऑशनपरे', 'ऑरेपशन', 'ऑपेशन'],
    'टेस्ट': ['टैस्ट', 'टेट', 'टसट'],
    'रिपोर्ट': ['रपोट', 'रिपोट', 'रीपोर्ट'],
    'दवाइयाँ': ['दवइयां', 'दवया', 'दबाइयां'],
    'गर्मी': ['र्मीग', 'गमी', 'गरी'],
    'ठंड': ['ठंन', 'ठड', 'टंड'],
    'मौसम': ['सममौ', 'मोसम', 'मसम'],
    'त्वचा': ['चात्व', 'त्वशा', 'त्वच'],
    'मांसपेशियों': ['मांशिसपेयों', 'मांसशिपेयों', 'माशपेयों'],
    'हड्डी': ['हडद्दी', 'हड्डि', 'हडी'],
    'जोड़': ['जड़ो', 'जोर', 'ड़ोंज'],
    'मालिश': ['मासल', 'मलिश', 'मालश'],
    'व्यायाम': ['व्यायम', 'वायाम', 'व्याम'],
    'आराम': ['रामआ', 'आम', 'अराम'],
    'फिजियो': ['जिफियो', 'फिजो', 'फियो'],
    'खुजली': ['खली', 'खजली', 'खुलजी'],
    'जलन': ['जलद', 'जन', 'जदल'],
    'सुन्नपन': ['सुन्नपती', 'सन', 'सपन'],
    'झनझनाहट': ['झनझनात', 'झुनझन', 'झझना'],
    'घुटने': ['घनों', 'घुटन', 'घुटत'],
    'कंधे': ['कंद', 'कधे', 'कंदे'],
    'उँगलियों': ['उंगयां', 'उँलियों', 'उबलियों'],
    'एमआरआई': ['आरएमआई', 'मरआई', 'एरआई'],
    'पीरियड्स': ['पीयरिड्स', 'पीड्सरिय', 'पीड्स'],
    'स्पॉटिंग': ['स्पॉटग', 'सपोटिंग', 'स्टॉपिंग'],
    'डैंड्रफ': ['ड़ाथो', 'डंडरफ', 'डेंडरफ'],
    'मस्सा': ['ससाम', 'मस्', 'मसा'],
    'छाले': ['छो', 'चले', 'छल'],
    'मवाद': ['दमवा', 'मबाद', 'मवद'],
    'बलगम': ['मबग', 'बमलग', 'बगम'],
    'पेशाब': ['शाबपे', 'बपेशा', 'पेसाब'],
    'प्यास': ['सप्या', 'प्यस', 'पास'],
    'गंध': ['गंते', 'गध', 'गंद'],
    'स्वाद': ['स्वात', 'सवाद', 'स्वद'],
    'आवाज़': ['ज़आवा', 'आबाज', 'आज़'],
    'सुगंध': ['सुधगं', 'सगंध', 'सधं'],
    'परफ्यूम': ['परमफ्यू', 'परफूम', 'प्फूम'],
    'अगरबत्ती': ['अत्तीगरब', 'अगबती', 'अगरती'],
}

def add_char_substitution(text: str, prob: float = 0.15) -> str:
    """Substitute characters with phonetically similar ones."""
    result = []
    for char in text:
        if char in CHAR_SUBSTITUTIONS and random.random() < prob:
            result.append(random.choice(CHAR_SUBSTITUTIONS[char]))
        else:
            result.append(char)
    return ''.join(result)

def add_matra_errors(text: str, prob: float = 0.12) -> str:
    """Add matra (vowel sign) errors."""
    result = []
    for char in text:
        if char in MATRA_SUBSTITUTIONS and random.random() < prob:
            replacement = random.choice(MATRA_SUBSTITUTIONS[char])
            result.append(replacement)
        else:
            result.append(char)
    return ''.join(result)

def add_word_substitution(text: str, prob: float = 0.20) -> str:
    """Substitute entire words with common ASR confusions."""
    words = text.split()
    result = []
    for word in words:
        # Remove punctuation for matching
        clean_word = word.rstrip('।,?!')
        punct = word[len(clean_word):]
        
        if clean_word in WORD_SUBSTITUTIONS and random.random() < prob:
            new_word = random.choice(WORD_SUBSTITUTIONS[clean_word])
            result.append(new_word + punct)
        else:
            result.append(word)
    return ' '.join(result)

def add_char_deletion(text: str, prob: float = 0.05) -> str:
    """Randomly delete characters (simulating audio dropout)."""
    result = []
    for char in text:
        if char not in ' ।,?!' and random.random() < prob:
            continue  # Skip (delete) this character
        result.append(char)
    return ''.join(result)

def add_char_insertion(text: str, prob: float = 0.03) -> str:
    """Randomly insert extra characters."""
    devanagari_chars = 'अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह'
    result = []
    for char in text:
        result.append(char)
        if char not in ' ।,?!' and random.random() < prob:
            result.append(random.choice(devanagari_chars))
    return ''.join(result)

def add_spacing_errors(text: str, prob: float = 0.08) -> str:
    """Add spacing errors (merge or split words)."""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if i < len(words) - 1 and random.random() < prob:
            # Merge two words
            result.append(words[i] + words[i+1])
            i += 2
        elif len(words[i]) > 4 and random.random() < prob * 0.5:
            # Split a word
            split_point = random.randint(2, len(words[i]) - 2)
            result.append(words[i][:split_point])
            result.append(words[i][split_point:])
            i += 1
        else:
            result.append(words[i])
            i += 1
    return ' '.join(result)

def add_halant_errors(text: str, prob: float = 0.08) -> str:
    """Add errors related to halant (virama) - the schwa deletion marker."""
    result = []
    for i, char in enumerate(text):
        if char == '्' and random.random() < prob:
            # Either drop the halant or add schwa
            if random.random() < 0.5:
                continue  # Drop halant
            else:
                result.append('अ')  # Add schwa
                continue
        result.append(char)
    return ''.join(result)

def swap_adjacent_chars(text: str, prob: float = 0.04) -> str:
    """Swap adjacent characters (typing errors)."""
    result = list(text)
    for i in range(len(result) - 1):
        if result[i] not in ' ।,?!' and result[i+1] not in ' ।,?!' and random.random() < prob:
            result[i], result[i+1] = result[i+1], result[i]
    return ''.join(result)

def apply_noise(text: str, noise_level: str = 'medium') -> str:
    """Apply multiple noise functions to create realistic ASR-like errors."""
    
    # Noise level configurations
    noise_configs = {
        'low': {
            'char_sub': 0.08,
            'matra': 0.06,
            'word_sub': 0.10,
            'deletion': 0.02,
            'insertion': 0.01,
            'spacing': 0.03,
            'halant': 0.04,
            'swap': 0.02
        },
        'medium': {
            'char_sub': 0.15,
            'matra': 0.12,
            'word_sub': 0.20,
            'deletion': 0.05,
            'insertion': 0.03,
            'spacing': 0.08,
            'halant': 0.08,
            'swap': 0.04
        },
        'high': {
            'char_sub': 0.25,
            'matra': 0.20,
            'word_sub': 0.30,
            'deletion': 0.08,
            'insertion': 0.05,
            'spacing': 0.12,
            'halant': 0.12,
            'swap': 0.06
        }
    }
    
    config = noise_configs.get(noise_level, noise_configs['medium'])
    
    # Apply noise functions in sequence
    noisy_text = text
    
    # Word-level substitutions first
    noisy_text = add_word_substitution(noisy_text, config['word_sub'])
    
    # Character-level modifications
    noisy_text = add_char_substitution(noisy_text, config['char_sub'])
    noisy_text = add_matra_errors(noisy_text, config['matra'])
    noisy_text = add_halant_errors(noisy_text, config['halant'])
    noisy_text = add_char_deletion(noisy_text, config['deletion'])
    noisy_text = add_char_insertion(noisy_text, config['insertion'])
    noisy_text = swap_adjacent_chars(noisy_text, config['swap'])
    noisy_text = add_spacing_errors(noisy_text, config['spacing'])
    
    return noisy_text

def generate_sentence() -> str:
    """Generate a clean Hindi medical sentence."""
    if random.random() < 0.4:
        # Use complete sentence
        return random.choice(COMPLETE_SENTENCES)
    else:
        # Generate from template
        template = random.choice(SENTENCE_TEMPLATES)
        
        # Fill in the template
        sentence = template.format(
            body_part=random.choice(BODY_PARTS),
            symptom=random.choice(SYMPTOMS),
            symptom2=random.choice(SYMPTOMS),
            condition=random.choice(CONDITIONS),
            time=random.choice(TIME_EXPRESSIONS),
            intensity=random.choice(INTENSITY),
            verb=random.choice(SYMPTOM_VERBS)
        )
        
        # Add period if not present
        if not sentence.endswith(('।', '?', '!')):
            sentence += '।'
        
        return sentence

def generate_dataset(num_samples: int = 12000, output_file: str = 'dataset.csv') -> None:
    """Generate the training dataset with noisy-clean pairs."""
    
    print(f"Generating {num_samples} training pairs...")
    
    pairs = []
    noise_levels = ['low', 'medium', 'high']
    noise_weights = [0.3, 0.5, 0.2]  # More medium noise
    
    for i in range(num_samples):
        # Generate clean sentence
        clean_sentence = generate_sentence()
        
        # Choose noise level
        noise_level = random.choices(noise_levels, weights=noise_weights)[0]
        
        # Generate noisy version
        noisy_sentence = apply_noise(clean_sentence, noise_level)
        
        # Ensure noisy is different from clean (at least for most samples)
        attempts = 0
        while noisy_sentence == clean_sentence and attempts < 5:
            noisy_sentence = apply_noise(clean_sentence, noise_level)
            attempts += 1
        
        pairs.append({
            'noisy_input': noisy_sentence,
            'clean_output': clean_sentence
        })
        
        if (i + 1) % 2000 == 0:
            print(f"Generated {i + 1}/{num_samples} pairs...")
    
    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['noisy_input', 'clean_output'])
        writer.writeheader()
        writer.writerows(pairs)
    
    print(f"Dataset saved to {output_file}")
    print(f"Total pairs: {len(pairs)}")
    
    # Print some examples
    print("\nExample pairs:")
    for i in range(5):
        idx = random.randint(0, len(pairs) - 1)
        print(f"\nExample {i+1}:")
        print(f"  Noisy:  {pairs[idx]['noisy_input']}")
        print(f"  Clean:  {pairs[idx]['clean_output']}")

def analyze_noise_patterns(clean: str, noisy: str) -> Dict:
    """Analyze the types of noise introduced."""
    analysis = {
        'char_changes': 0,
        'length_diff': len(noisy) - len(clean),
        'word_count_diff': len(noisy.split()) - len(clean.split())
    }
    
    # Count character-level changes
    for c1, c2 in zip(clean, noisy):
        if c1 != c2:
            analysis['char_changes'] += 1
    
    return analysis

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Hindi medical noisy data')
    parser.add_argument('--num_samples', type=int, default=12000, 
                       help='Number of training pairs to generate')
    parser.add_argument('--output', type=str, default='dataset.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    generate_dataset(args.num_samples, args.output)
