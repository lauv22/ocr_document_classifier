import re

def classify_document(text):
    text_lower = text.lower()
    words = text_lower.split()

    def keyword_matches(keyword, text_lower, words):
        if keyword in text_lower:
            return True
        keyword_words = keyword.split()
        if len(keyword_words) == 1:
            return any(keyword in word for word in words)
        else:
            return all(any(kw in word for word in words) for kw in keyword_words)

    passport_keywords = [
        'passport', 'republic of nepal', 'place of birth',
        'date of issue', 'date of expiry', 'nationality', 'mrp', 'p<npl',
    ]
    citizenship_keywords = [
        'citizenship', 'nagrikta', 'nagarikta',
        'certificate of citizenship', 'permanent residence',
        'ward no', 'tole', 'citizenship no', 'nag. pra. pa',
    ]
    pan_keywords = [
        'pan', 'permanent account number', 'inland revenue',
        'tax', 'pan no', 'office of inland revenue', 'taxpayer', 'vat',
    ]
    national_id_keywords = [
        'national identity card', 'national id', 'government of nepal',
        'nepal government', 'nepal sarkar', 'rastriya parichaya patra',
        'darta number', 'janma miti', 'citizenship number',
        'smart card', 'nepalese', 'identity',
    ]

    passport_score    = sum(1 for kw in passport_keywords    if keyword_matches(kw, text_lower, words))
    citizenship_score = sum(1 for kw in citizenship_keywords if keyword_matches(kw, text_lower, words))
    pan_score         = sum(1 for kw in pan_keywords         if keyword_matches(kw, text_lower, words))
    national_id_score = sum(1 for kw in national_id_keywords if keyword_matches(kw, text_lower, words))

    scores = {
        'Passport'    : passport_score,
        'Citizenship' : citizenship_score,
        'PAN'         : pan_score,
        'National ID' : national_id_score
    }

    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]

    if best_score == 0:
        return 'Unknown', scores

    return best_match, scores


def extract_fields(text, doc_type):
    fields = {}

    def get_value_after(keyword, text, word_count=3):
        idx = text.upper().find(keyword.upper())
        if idx == -1:
            return "Not found"
        after = text[idx + len(keyword):].strip()
        after = re.sub(r'[|=\-_<>{}~]', ' ', after)
        words = [w for w in after.split() if len(w) > 1 and w not in ['A', 'i', 'att', 'ara', 'fry']]
        result = ' '.join(words[:word_count])
        return result if result else "Not found"

    def extract_from_mrz(text):
        mrz_match = re.search(r'P[<]?NPL([A-Z<]+)', text.upper())
        if mrz_match:
            mrz_part = mrz_match.group(1)
            parts = mrz_part.split('<<')
            surname   = parts[0].replace('<', ' ').strip() if len(parts) > 0 else "Not found"
            given     = parts[1].replace('<', ' ').strip() if len(parts) > 1 else "Not found"
            return surname, given
        return "Not found", "Not found"

    def extract_date(keyword, text):
        idx = text.upper().find(keyword.upper())
        if idx == -1:
            return "Not found"
        snippet = text[idx:idx+60]
        date_match = re.search(
            r'(\d{1,2}\s+[A-Z]{3}\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})',
            snippet.upper()
        )
        if date_match:
            return date_match.group(1)
        return "Not found"

    if doc_type == 'Passport':
        surname, given_names = extract_from_mrz(text)
        fields = {
            'Document Type'    : 'Passport',
            'Surname'          : surname,
            'Given Names'      : given_names,
            'Nationality'      : get_value_after('NATIONALITY', text, 1),
            'Date of Birth'    : extract_date('DATE OF BIRTH', text),
            'Date of Issue'    : extract_date('DATE OF ISSUE', text),
            'Date of Expiry'   : extract_date('DATE OF EXPIRY', text),
            'Place of Birth'   : get_value_after('PLACE OF BIRTH', text, 2),
            'Issuing Authority': get_value_after('ISSUING AUTHORITY', text, 4),
        }
    elif doc_type == 'National ID':
        surname, given_names = extract_from_mrz(text)
        fields = {
            'Document Type': 'National Identity Card',
            'Surname'       : surname if surname != "Not found" else get_value_after('SURNAME', text, 2),
            'Given Names'   : given_names if given_names != "Not found" else get_value_after('GIVEN NAMES', text, 3),
            'Date of Birth' : extract_date('DATE OF BIRTH', text),
            'Nationality'   : get_value_after('NATIONALITY', text, 1),
            'Date of Issue' : extract_date('DATE OF ISSUE', text),
            'Date of Expiry': extract_date('DATE OF EXPIRY', text),
        }
    elif doc_type == 'Citizenship':
        fields = {
            'Document Type': 'Citizenship Certificate',
            'Full Name'    : get_value_after('NAME', text, 3),
            'Date of Birth': extract_date('DATE OF BIRTH', text),
            'Ward No'      : get_value_after('WARD NO', text, 1),
            'District'     : get_value_after('DISTRICT', text, 2),
        }
    elif doc_type == 'PAN':
        fields = {
            'Document Type': 'PAN Card',
            'Full Name'    : get_value_after('NAME', text, 3),
            'PAN Number'   : get_value_after('PAN NO', text, 1),
            'Date of Issue': extract_date('DATE OF ISSUE', text),
        }
    else:
        fields = {
            'Document Type': 'Unknown',
            'Note'         : 'Could not extract fields from this document'
        }

    return fields
