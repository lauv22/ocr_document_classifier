def classify_document(text):
    """
    Takes extracted text and checks for keywords
    to determine what type of document it is.
    """

    # Convert to lowercase so matching works regardless
    # of whether text is "PASSPORT" or "Passport" or "passport"
    text_lower = text.lower()

    # Define keywords for each document type
    passport_keywords = [
        'passport',
        'republic of nepal',
        'place of birth',
        'date of issue',
        'date of expiry',
        'nationality',
        'mrp',
        'p<npl',
    ]

    citizenship_keywords = [
        'citizenship',
        'nagrikta',
        'nagarikta',
        'certificate of citizenship',
        'permanent residence',
        'ward no',
        'tole',
        'citizenship no',
        'nag. pra. pa',
    ]

    pan_keywords = [
        'pan',
        'permanent account number',
        'inland revenue',
        'tax',
        'pan no',
        'office of inland revenue',
        'taxpayer',
        'vat',
    ]

    national_id_keywords = [
        'national identity card',
        'national id',
        'government of nepal',
        'nepal government',
        'nepal sarkar',
        'rastriya parichaya patra',
        'darta number',
        'janma miti',
        'citizenship number',
        'smart card',
        'nepalese',
    ]

    # Count how many keywords match for each document type
    passport_score = sum(1 for keyword in passport_keywords if keyword in text_lower)
    citizenship_score = sum(1 for keyword in citizenship_keywords if keyword in text_lower)
    pan_score = sum(1 for keyword in pan_keywords if keyword in text_lower)
    national_id_score = sum(1 for keyword in national_id_keywords if keyword in text_lower)

    # Find which document type has the highest score
    scores = {
        'Passport': passport_score,
        'Citizenship': citizenship_score,
        'PAN': pan_score,
        'National ID': national_id_score
    }

    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]

    # If no keywords matched at all, return Unknown
    if best_score == 0:
        return 'Unknown', scores

    return best_match, scores