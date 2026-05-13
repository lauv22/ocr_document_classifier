def classify_document(text):
    """
    Takes extracted text and checks for keywords
    to determine what type of document it is.
    """

    # Convert to lowercase so matching works regardless
    # of whether text is "PASSPORT" or "Passport" or "passport"
    text_lower = text.lower()

    # Define keywords for each document type
    # Add more keywords if you find the OCR misses some
    passport_keywords = [
        'passport',
        'republic of nepal',
        'place of birth',
        'date of issue',
        'date of expiry',
        'nationality',
        'mrp',                  # Machine Readable Passport
        'p<npl',                # Code found in passport MRZ line
    ]

    citizenship_keywords = [
        'citizenship',
        'nagrikta',             # Nepali word for citizenship
        'nagarikta',
        'certificate of citizenship',
        'permanent residence',
        'ward no',
        'tole',
        'citizenship no',
        'nag. pra. pa',         # Common abbreviation on Nepali citizenship
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

    # Count how many keywords match for each document type
    passport_score = sum(1 for keyword in passport_keywords if keyword in text_lower)
    citizenship_score = sum(1 for keyword in citizenship_keywords if keyword in text_lower)
    pan_score = sum(1 for keyword in pan_keywords if keyword in text_lower)

    # Find which document type has the highest score
    scores = {
        'Passport': passport_score,
        'Citizenship': citizenship_score,
        'PAN': pan_score
    }

    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]

    # If no keywords matched at all, return Unknown
    if best_score == 0:
        return 'Unknown', scores

    return best_match, scores