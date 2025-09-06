# simple URL feature extractor
import re
from urllib.parse import urlparse

def extract_features(url: str) -> list[float]:
    u = url.strip()
    parsed = urlparse(u if '://' in u else 'http://' + u)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # Some simple features
    length = len(u)
    dots = hostname.count('.')
    has_https = 1.0 if parsed.scheme == 'https' else 0.0
    has_at = 1.0 if '@' in u else 0.0
    suspicious_words = sum([1 for w in ['login','secure','bank','account'] if w in u.lower()])

    # return floats (model expects numeric vector)
    return [float(length), float(dots), has_https, has_at, float(suspicious_words)]
