def detect_language(text, Language, LanguageDetectorBuilder, client, logging):
    """Detect language of a given text using an LLM for short texts and Lingua for longer ones."""
    text = text.lower().strip()

    # Use LLM for short texts
    if len(text.split()) < 5:
        system_prompt = """
        You are a language detection assistant. Identify the language of the given text.
        Return only the language code: "en" for English or "sw" for Swahili.
        If the language is neither English nor Swahili, return "unknown".
        """

        user_message = f"Text: \"{text}\""

        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0  # Deterministic output
            )
            detected_language = completion.choices[0].message.content.strip()
            return detected_language
        except Exception as e:
            logging.warning(f"Language detection error (LLM): {e}")
            return "unknown"

    # Use Lingua for longer texts
    try:
        languages = [Language.ENGLISH, Language.SWAHILI]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
        detected_language = detector.detect_language_of(text)
        return "sw" if detected_language == Language.SWAHILI else "en"
    except Exception as e:
        logging.warning(f"Language detection error (Lingua): {e}")
        return "unknown"


def detect_intention(user_input, client):
    system_prompt = """
    You are an intent classification assistant. Classify the user's message into one of the following categories:

    - "greeting" for messages like "hi", "hello", or similar
    - "acknowledgment" for messages like "thanks", "okay", or similar
    - "message" for anything else that may require a response, including health concerns or information requests

    The user may speak in English or Swahili. Be aware that they might not use proper punctuation or grammar.

    Return only the label: "greeting", "acknowledgment", or "message".
    """

    user_message = f"Message: \"{user_input}\""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0  # for deterministic output
    )

    return completion.choices[0].message.content

def convert_to_date(date_str, datetime):
    """Convert date string in YYYYMMDD format to YYYY-MM-DD."""
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return "Unknown Date"