"""Unstructured-data analytics: text cleaning, sentiment scoring, aspect and
category extraction for customer reviews and social-media posts.

A rule/lexicon approach (in the spirit of VADER, Hutto & Gilbert, 2014) is used
because it is explainable, needs no model download and runs at >50k texts/sec
on one core, which keeps the prototype runnable on a laptop. In production the
same function signature can be backed by a transformer model running in Spark
NLP; the rest of the pipeline would not change.
"""
from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z']+|[\U0001F300-\U0001FAFF☀-➿]")

POSITIVE = {
    "love": 3.0, "loved": 3.0, "excellent": 3.2, "amazing": 3.1, "awesome": 3.0, "great": 2.5,
    "perfect": 3.0, "fantastic": 3.1, "happy": 2.3, "good": 1.9, "nice": 1.8, "fast": 1.5,
    "quick": 1.4, "comfortable": 1.9, "recommend": 2.0, "recommended": 2.0, "sturdy": 1.7,
    "beautiful": 2.6, "reliable": 1.9, "impressed": 2.4, "bargain": 1.8, "solid": 1.5,
    "helpful": 1.9, "easy": 1.4, "worth": 1.6, "best": 2.8, "superb": 3.0, "pleased": 2.2,
    "stylish": 2.0, "crisp": 1.5, "smooth": 1.5, "friendly": 1.8, "well": 0.8, "brilliant": 3.0,
    "affordable": 1.5, "durable": 1.8, "satisfied": 2.0, "flawless": 3.0, "thanks": 1.2,
}
NEGATIVE = {
    "bad": -2.5, "terrible": -3.2, "awful": -3.1, "poor": -2.4, "broken": -2.8, "broke": -2.6,
    "disappointed": -2.6, "disappointing": -2.6, "worst": -3.4, "slow": -1.8, "late": -1.9,
    "delayed": -1.9, "refund": -1.6, "return": -1.0, "returned": -1.4, "faulty": -2.8,
    "cheap": -1.4, "flimsy": -2.2, "damaged": -2.6, "rude": -2.6, "useless": -3.0,
    "overpriced": -2.2, "waste": -2.7, "wrong": -2.0, "missing": -2.0, "problem": -1.7,
    "issue": -1.5, "issues": -1.5, "never": -1.2, "uncomfortable": -2.0, "small": -0.6,
    "tight": -1.0, "scratched": -2.2, "dies": -2.0, "died": -2.2, "angry": -2.6, "avoid": -2.8,
    "horrible": -3.2, "hate": -3.0, "unhappy": -2.4, "fake": -2.6, "ignored": -2.2, "lost": -1.8,
}
EMOJI = {"\U0001F600": 2.0, "\U0001F60D": 3.0, "\U0001F44D": 2.0, "❤": 2.5, "\U0001F525": 1.5,
         "\U0001F620": -2.5, "\U0001F621": -3.0, "\U0001F44E": -2.0, "\U0001F622": -2.0, "\U0001F92C": -3.0}
NEGATORS = {"not", "no", "never", "isn't", "wasn't", "don't", "didn't", "doesn't", "won't",
            "wouldn't", "can't", "couldn't", "hardly", "nothing"}
INTENSIFIERS = {"very": 1.3, "really": 1.25, "extremely": 1.5, "super": 1.3, "so": 1.2,
                "absolutely": 1.4, "totally": 1.3, "quite": 1.1, "incredibly": 1.5}

ASPECTS = {
    "delivery": {"delivery", "shipping", "arrived", "courier", "dispatch", "late", "delayed", "parcel"},
    "quality": {"quality", "build", "made", "material", "sturdy", "flimsy", "broke", "broken", "faulty",
                "durable", "cheap", "stitching"},
    "price": {"price", "value", "money", "expensive", "overpriced", "bargain", "worth", "affordable", "cost"},
    "customer_service": {"service", "support", "staff", "refund", "rude", "helpful", "ignored", "replied"},
    "size_fit": {"size", "fit", "fits", "tight", "loose", "small", "large", "sizing"},
    "battery": {"battery", "charge", "charging", "dies", "died"},
    "packaging": {"packaging", "box", "packed", "damaged", "scratched"},
}

CATEGORY_KEYWORDS = {
    "Phones": {"phone", "smartphone", "mobile"},
    "Laptops": {"laptop", "notebook", "ultrabook"},
    "Audio": {"headphones", "earbuds", "speaker", "soundbar"},
    "Accessories": {"charger", "cable", "case", "adapter"},
    "Menswear": {"shirt", "jacket", "jeans", "menswear", "hoodie"},
    "Womenswear": {"dress", "skirt", "blouse", "womenswear", "coat"},
    "Footwear": {"shoes", "sneakers", "boots", "runners", "sandals"},
    "Kitchen": {"blender", "kettle", "cookware", "toaster", "kitchen"},
    "Furniture": {"sofa", "desk", "chair", "bookshelf", "furniture"},
    "Decor": {"lamp", "rug", "cushion", "candle", "decor"},
}
SUBCAT_TO_CATEGORY = {
    "Phones": "Electronics", "Laptops": "Electronics", "Audio": "Electronics", "Accessories": "Electronics",
    "Menswear": "Apparel", "Womenswear": "Apparel", "Footwear": "Apparel",
    "Kitchen": "Home", "Furniture": "Home", "Decor": "Home",
}


def clean_text(text) -> str:
    """Strip HTML, decode entities and normalise whitespace (veracity step)."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = html.unescape(str(text))
    t = _TAG_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def sentiment(text: str) -> float:
    """Return a compound sentiment score in [-1, 1]."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    total = 0.0
    for i, tok in enumerate(tokens):
        val = POSITIVE.get(tok) or NEGATIVE.get(tok) or EMOJI.get(tok)
        if not val:
            continue
        window = tokens[max(0, i - 3):i]
        if any(w in NEGATORS for w in window):
            val = -0.75 * val
        if i > 0 and tokens[i - 1] in INTENSIFIERS:
            val *= INTENSIFIERS[tokens[i - 1]]
        total += val
    # caps-lock shouting amplifies the polarity
    if text.isupper() and len(text) > 8:
        total *= 1.2
    # VADER-style normalisation to [-1, 1]
    return round(total / ((total * total + 15) ** 0.5), 4)


def sentiment_label(score: float) -> str:
    if score >= 0.25:
        return "positive"
    if score <= -0.25:
        return "negative"
    return "neutral"


def aspects(text: str) -> list[str]:
    toks = set(tokenize(text))
    return sorted(a for a, kws in ASPECTS.items() if toks & kws)


def detect_subcategory(text: str) -> str | None:
    toks = set(tokenize(text))
    best, hits = None, 0
    for sub, kws in CATEGORY_KEYWORDS.items():
        h = len(toks & kws)
        if h > hits:
            best, hits = sub, h
    return best


def analyse(text) -> dict:
    """Full analysis used by the Silver layer and by the streaming job."""
    clean = clean_text(text)
    score = sentiment(clean)
    return {
        "clean_text": clean,
        "sentiment": score,
        "sentiment_label": sentiment_label(score),
        "aspects": ",".join(aspects(clean)),
    }
