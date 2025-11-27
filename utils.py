import re


def clean_text(text: str) -> str:
    """
    Basic cleaning of scraped HTML text:
    - Remove HTML tags
    - Remove URLs
    - Remove special characters
    - Normalize spaces
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]*?>", "", text)
    # Remove URLs
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    # Remove special characters except space
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s{2,}", " ", text)
    # Trim leading and trailing whitespace
    text = text.strip()
    # Final normalization
    text = " ".join(text.split())
    return text
