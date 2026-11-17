"""Synthetic EzyShop source-system generator.

Creates the five siloed source domains described in Assessment 2 in their
native formats, so that the prototype has to integrate structured,
semi-structured and unstructured data:

=====================  ==================  ===============================
Source                 Format              Variety
=====================  ==================  ===============================
customers / products   CSV (CRM / PIM)     structured
orders (order lines)   CSV (OLTP export)   structured
inventory snapshot     CSV (WMS)           structured
campaigns              CSV (marketing)     structured
suppliers              nested JSON         semi-structured
clickstream            JSON lines / day    semi-structured, nested
reviews                JSON lines          unstructured free text
social posts           JSON lines          unstructured free text
=====================  ==================  ===============================

Hidden behaviour is simulated so the analytics have something real to find:
seasonality + Black Friday peaks (forecasting), latent customer tastes
(recommendations), satisfaction-driven churn with disengagement before
leaving (churn model), and deliberate data-quality defects (veracity).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Config

SUBCATS = [
    ("Electronics", "Phones", (250, 1600)),
    ("Electronics", "Laptops", (600, 2800)),
    ("Electronics", "Audio", (40, 600)),
    ("Electronics", "Accessories", (10, 120)),
    ("Apparel", "Menswear", (20, 180)),
    ("Apparel", "Womenswear", (20, 220)),
    ("Apparel", "Footwear", (40, 260)),
    ("Home", "Kitchen", (15, 450)),
    ("Home", "Furniture", (80, 1500)),
    ("Home", "Decor", (10, 150)),
]
SUB_NAMES = [s[1] for s in SUBCATS]
CATEGORIES = ["Electronics", "Apparel", "Home"]
NOUNS = {
    "Phones": ["Smartphone", "Phone"], "Laptops": ["Laptop", "Ultrabook"],
    "Audio": ["Headphones", "Earbuds", "Speaker", "Soundbar"], "Accessories": ["Charger", "Cable", "Case", "Adapter"],
    "Menswear": ["Shirt", "Jacket", "Jeans", "Hoodie"], "Womenswear": ["Dress", "Skirt", "Blouse", "Coat"],
    "Footwear": ["Sneakers", "Boots", "Runners", "Sandals"], "Kitchen": ["Blender", "Kettle", "Cookware Set", "Toaster"],
    "Furniture": ["Sofa", "Desk", "Chair", "Bookshelf"], "Decor": ["Lamp", "Rug", "Cushion", "Candle"],
}
BRANDS = ["Novi", "Zentro", "Apexa", "Kora", "Lumen", "Southern Co", "Harbour", "Vantage", "Orbit", "Willow"]
SEGMENT_PREFS = {  # relative affinity per sub-category (same order as SUBCATS)
    "Tech Enthusiast": [5, 4, 4, 4, 1, 0.6, 1, 0.8, 0.5, 0.5],
    "Fashionista":     [1.2, 0.5, 1.2, 1.0, 3, 5, 4, 0.6, 0.4, 1.5],
    "Home Maker":      [0.8, 0.6, 1.0, 1.0, 1, 1.2, 1, 5, 3, 4],
    "Bargain Hunter":  [1.5, 0.8, 2.0, 3.0, 2, 2, 1.5, 2, 0.6, 2],
    "Generalist":      [2, 1.2, 2, 2, 2, 2, 2, 2, 1.2, 2],
}
SEGMENT_SHARE = [0.18, 0.2, 0.2, 0.17, 0.25]
CITIES = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra", "Gold Coast", "Newcastle", "Hobart", "Darwin"]
CITY_W = [0.24, 0.22, 0.12, 0.10, 0.07, 0.05, 0.06, 0.05, 0.05, 0.04]
FIRST = ["Olivia", "Liam", "Mia", "Noah", "Isla", "Jack", "Amelia", "William", "Ava", "Henry", "Chloe", "Leo",
         "Zara", "Arjun", "Mei", "Lucas", "Priya", "Ethan", "Sofia", "Hamish"]
LAST = ["Smith", "Nguyen", "Williams", "Brown", "Wilson", "Taylor", "Patel", "Chen", "Kelly", "Singh",
        "Martin", "Lee", "Walker", "Harris", "Ryan", "Murphy"]

# ---------------------------------------------------------------- review text
POS = {
    "quality": ["The build quality is excellent", "Really sturdy and well made", "Quality is great for the price",
                "Feels durable and solid"],
    "delivery": ["Delivery was fast", "Arrived quickly and well packed", "Shipping was really quick"],
    "price": ["Great value for money", "An absolute bargain", "Worth every cent"],
    "customer_service": ["Customer service was very helpful", "Support staff were friendly and helpful"],
    "battery": ["Battery life is amazing", "Charges fast and the battery lasts all day"],
    "size_fit": ["Fits perfectly", "Size is spot on and comfortable"],
    "general": ["Love it", "Highly recommend", "Very happy with this purchase", "Fantastic product",
                "Exactly as described, really impressed"],
}
NEG = {
    "quality": ["Poor quality, it broke after a week", "Feels cheap and flimsy", "Arrived faulty",
                "Stitching came apart, really disappointing"],
    "delivery": ["Delivery was late", "Parcel was delayed by two weeks", "Courier lost my parcel at first"],
    "price": ["Overpriced for what you get", "Not worth the money"],
    "customer_service": ["Customer service ignored my emails", "Support was rude and unhelpful",
                         "Still waiting on my refund"],
    "battery": ["Battery dies within hours", "Charging is very slow"],
    "size_fit": ["Way too tight, sizing is wrong", "Runs small and is uncomfortable"],
    "packaging": ["Box was damaged and the item scratched"],
    "general": ["Would not recommend", "Terrible experience", "Very disappointed", "Waste of money, avoid"],
}
NEUTRAL = ["It is okay", "Does the job", "Average product", "As expected nothing special", "It works"]
TITLES = {5: ["Excellent!", "Love it", "Five stars"], 4: ["Very good", "Happy overall"], 3: ["It's okay", "Average"],
          2: ["Disappointed", "Not great"], 1: ["Terrible", "Avoid", "Waste of money"]}
SUB_ASPECTS = {"Phones": "battery", "Laptops": "battery", "Audio": "battery", "Menswear": "size_fit",
               "Womenswear": "size_fit", "Footwear": "size_fit"}

SOCIAL_POS = ["Just got my new {n} from @EzyShop and I love it {e}", "@EzyShop delivered my {n} super fast, great service",
              "Best {n} I have bought, thanks @EzyShop", "Huge sale on {n} at @EzyShop, awesome bargain {e}"]
SOCIAL_NEG = ["@EzyShop my {n} arrived broken and support ignored me {e}", "Still waiting on my {n} from @EzyShop, so slow",
              "Really disappointed with the {n} from @EzyShop, poor quality", "@EzyShop worst experience, {n} was faulty"]
SOCIAL_NEU = ["Anyone tried the {n} from @EzyShop?", "Looking at a new {n} on @EzyShop", "@EzyShop do you ship {n} to Darwin?"]


def black_friday(year: int) -> date:
    d = date(year, 11, 1)
    first_thu = d + timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + timedelta(days=22)


def demand_multipliers(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return (total_multiplier[days], subcat_multiplier[days, 10])."""
    n = len(dates)
    doy = dates.day_of_year.values
    t = np.arange(n)
    weekday = np.array([0.95, 0.92, 0.95, 1.0, 1.08, 1.15, 1.05])[dates.day_of_week.values]
    annual = 1 + 0.12 * np.cos(2 * np.pi * (doy - 355) / 365.25)
    trend = 1.2 ** (t / 365.0)  # 20 % annual growth (Assessment 2)
    event = np.ones(n)
    sub = np.ones((n, len(SUBCATS)))
    for i, d in enumerate(dates.date):
        bf = black_friday(d.year)
        if bf <= d <= bf + timedelta(days=3):          # Black Friday -> Cyber Monday
            event[i] = 2.3
            sub[i, :4] *= 1.35                           # electronics heavy
        elif date(d.year, 12, 1) <= d <= date(d.year, 12, 20):
            event[i] = 1.3
            sub[i, 7] *= 1.25; sub[i, 9] *= 1.3          # kitchen + decor gifting
        elif date(d.year, 12, 26) <= d <= date(d.year, 12, 31):
            event[i] = 1.7                               # Boxing Day sales
        elif date(d.year, 6, 15) <= d <= date(d.year, 6, 30):
            event[i] = 1.25                              # EOFY (Australia)
            sub[i, 1] *= 1.4; sub[i, 8] *= 1.3           # laptops, furniture
        if d.month in (6, 7, 8):                         # southern-hemisphere winter
            sub[i, 4] *= 1.2; sub[i, 5] *= 1.2
        if d.month in (9, 10, 11):
            sub[i, 6] *= 1.15                            # spring footwear
    return weekday * annual * trend * event, sub


class SourceGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.p = cfg.params
        self.rng = np.random.default_rng(cfg.seed)
        self.end = pd.Timestamp(cfg.end_date)
        self.start = self.end - pd.Timedelta(days=self.p["days"] - 1)
        self.dates = pd.date_range(self.start, self.end, freq="D")

    # ------------------------------------------------------------- catalogue
    def make_products(self) -> pd.DataFrame:
        rng, n = self.rng, self.p["products"]
        share = np.array([0.08, 0.05, 0.1, 0.12, 0.12, 0.13, 0.1, 0.12, 0.06, 0.12])
        sub_idx = rng.choice(len(SUBCATS), size=n, p=share / share.sum())
        sub_idx[: len(SUBCATS)] = np.arange(len(SUBCATS))  # every sub-category exists
        rows = []
        for i, s in enumerate(sub_idx):
            cat, sub, (lo, hi) = SUBCATS[s]
            price = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            rows.append({
                "product_id": f"P{i + 1:05d}",
                "product_name": f"{rng.choice(BRANDS)} {rng.choice(NOUNS[sub])} {rng.integers(100, 999)}",
                "category": cat, "subcategory": sub, "list_price": round(price, 2),
                "supplier_id": f"S{int(rng.integers(1, 26)):03d}",
                "launch_date": (self.start - pd.Timedelta(days=int(rng.integers(0, 700)))).date().isoformat(),
            })
        df = pd.DataFrame(rows)
        self.sub_idx = sub_idx
        self.cat_idx = np.array([CATEGORIES.index(SUBCATS[s][0]) for s in sub_idx])
        self.quality = rng.beta(5, 2, size=n)                       # hidden product quality
        self.item_vec = rng.normal(0, 1, size=(n, 6))               # hidden taste factors
        self.pop_log = rng.normal(0, 0.6, size=n)                   # hidden popularity
        self.prices = df["list_price"].values
        return df

    def make_customers(self) -> pd.DataFrame:
        rng, n = self.rng, self.p["customers"]
        seg_names = list(SEGMENT_PREFS)
        seg = rng.choice(len(seg_names), size=n, p=SEGMENT_SHARE)
        days = self.p["days"]
        existing = rng.random(n) < 0.6
        signup_off = np.where(existing, -rng.integers(30, 1100, size=n),
                              (rng.random(n) ** 0.8 * (days - 30)).astype(int))
        signup = self.start + pd.to_timedelta(signup_off, unit="D")
        satisfaction = rng.normal(0, 1, size=n)
        life_start = np.where(existing, 0, signup_off)
        lifetime = rng.exponential(1300 * np.exp(0.75 * satisfaction))
        churn_off = (life_start + lifetime).astype(int)            # hidden churn day offset
        self.seg = seg
        self.pref_log = np.log(np.array([SEGMENT_PREFS[seg_names[s]] for s in seg])
                               * rng.gamma(4, 0.25, size=(n, len(SUBCATS))))
        self.user_vec = rng.normal(0, 1, size=(n, 6)) * 0.7
        self.activity = rng.lognormal(0, 0.8, size=n) * np.where(seg == 3, 1.3, 1.0)
        self.satisfaction = satisfaction
        self.signup_off = np.maximum(signup_off, -10_000)
        self.churn_off = churn_off
        first = rng.choice(FIRST, size=n)
        last = rng.choice(LAST, size=n)
        df = pd.DataFrame({
            "customer_id": [f"C{i + 1:06d}" for i in range(n)],
            "first_name": first, "last_name": last,
            "email": [f"{f.lower()}.{l.lower()}{i}@example.com" for i, (f, l) in enumerate(zip(first, last))],
            "phone": [f"04{rng.integers(10_000_000, 99_999_999)}" for _ in range(n)],
            "city": rng.choice(CITIES, size=n, p=CITY_W),
            "age": np.clip(rng.normal(38, 12, size=n).round(), 18, 85).astype(int),
            "segment": [seg_names[s] for s in seg],
            "signup_date": signup.strftime("%Y-%m-%d"),
            "marketing_opt_in": rng.random(n) < 0.7,
        })
        # veracity defects: mixed date formats, duplicate CRM rows
        bad = rng.random(n) < 0.02
        df.loc[bad, "signup_date"] = signup[bad].strftime("%d/%m/%Y")
        dup = df.sample(frac=0.01, random_state=self.cfg.seed)
        return pd.concat([df, dup], ignore_index=True)

    def make_campaigns(self) -> pd.DataFrame:
        rng = self.rng
        rows, cid = [], 1
        n = max(6, int(self.p["days"] / 25))
        for _ in range(n):
            start = self.start + pd.Timedelta(days=int(rng.integers(0, self.p["days"] - 15)))
            dur = int(rng.integers(5, 15))
            cat = CATEGORIES[int(rng.integers(0, 3))]
            rows.append((cid, f"{cat} Flash Sale {cid}", rng.choice(["Email", "Social", "Search", "Display"]), cat,
                         start.date(), (start + pd.Timedelta(days=dur - 1)).date(),
                         round(float(rng.uniform(5_000, 60_000)), 2)))
            cid += 1
        for y in sorted({d.year for d in self.dates}):
            bf = black_friday(y)
            if self.start.date() <= bf <= self.end.date():
                rows.append((cid, f"Black Friday {y}", "Social", "Electronics", bf, bf + timedelta(days=3), 120_000.0))
                cid += 1
        return pd.DataFrame(rows, columns=["campaign_id", "campaign_name", "channel", "target_category",
                                           "start_date", "end_date", "spend_aud"])

    # ---------------------------------------------------------------- orders
    def make_orders(self, campaigns: pd.DataFrame) -> pd.DataFrame:
        rng, p = self.rng, self.p
        n_days, n_prod = len(self.dates), len(self.prices)
        total_mult, sub_mult = demand_multipliers(self.dates)
        sub_log = np.log(sub_mult / sub_mult.mean(axis=0, keepdims=True))
        camp_log = np.zeros((n_days, 3))
        for c in campaigns.itertuples():
            m = (self.dates.date >= c.start_date) & (self.dates.date <= c.end_date)
            camp_log[m, CATEGORIES.index(c.target_category)] += np.log(1.6)
        hours = np.array([1, 0.5, 0.3, 0.2, 0.2, 0.4, 1, 2, 3, 3.5, 4, 4.5, 5, 4.5, 4, 4, 4.5, 5, 6, 7, 7.5, 6.5, 4, 2])
        hours = hours / hours.sum()
        out = []
        order_no = 1
        self.daily_orders = np.zeros(n_days, dtype=int)
        for d in range(n_days):
            active = np.where((self.signup_off <= d) & (self.churn_off > d))[0]
            if len(active) == 0:
                continue
            lam = p["orders_per_day"] * total_mult[d] * rng.lognormal(0, 0.05)
            n = rng.poisson(lam)
            self.daily_orders[d] = n
            w = self.activity[active]
            cust = rng.choice(active, size=n, p=w / w.sum())
            scores = (self.pref_log[cust][:, self.sub_idx] + self.user_vec[cust] @ self.item_vec.T
                      + self.pop_log + sub_log[d][self.sub_idx] + camp_log[d][self.cat_idx])
            scores += rng.gumbel(size=scores.shape)
            k = np.minimum(1 + rng.poisson(0.55, size=n), 5)
            top = np.argsort(-scores, axis=1)[:, :5]
            secs = (rng.choice(24, size=n, p=hours) * 3600 + rng.integers(0, 3600, size=n))
            day_ts = self.dates[d]
            for j in range(n):
                oid = f"O{order_no:08d}"
                order_no += 1
                ts = (day_ts + pd.Timedelta(seconds=int(secs[j]))).isoformat()
                chan = "app" if rng.random() < 0.45 else "web"
                for line, pi in enumerate(top[j, : k[j]], start=1):
                    in_camp = camp_log[d][self.cat_idx[pi]] > 0
                    disc = rng.uniform(0.10, 0.25) if in_camp else (rng.uniform(0, 0.1) if rng.random() < 0.2 else 0.0)
                    qty = 1 if self.sub_idx[pi] < 4 or rng.random() < 0.8 else 2
                    r = rng.random()
                    status = "cancelled" if r < 0.02 else ("returned" if r < 0.02 + 0.01 + 0.07 * (1 - self.quality[pi]) else "completed")
                    out.append((oid, line, ts, int(cust[j]), int(pi), qty,
                                round(self.prices[pi] * (1 - disc), 2), round(disc, 3), chan, status))
        df = pd.DataFrame(out, columns=["order_id", "line_no", "order_ts", "cust_idx", "prod_idx", "quantity",
                                        "unit_price", "discount", "channel", "status"])
        return df

    def finalise_orders(self, orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        df = orders.copy()
        df["customer_id"] = customers["customer_id"].values[df["cust_idx"].values]
        df["product_id"] = products["product_id"].values[df["prod_idx"].values]
        df = df[["order_id", "line_no", "order_ts", "customer_id", "product_id", "quantity", "unit_price",
                 "discount", "channel", "status"]].astype({"quantity": "object", "unit_price": "object"})
        n = len(df)
        # ---- veracity defects ----
        idx = rng.choice(n, size=int(n * 0.003), replace=False)
        df.loc[df.index[idx], "customer_id"] = None                       # missing keys
        idx = rng.choice(n, size=int(n * 0.002), replace=False)
        df.loc[df.index[idx], "quantity"] = -1                            # impossible quantity
        idx = rng.choice(n, size=int(n * 0.001), replace=False)
        df.loc[df.index[idx], "unit_price"] = 999999.0                    # price outlier
        idx = rng.choice(n, size=int(n * 0.0005), replace=False)
        df.loc[df.index[idx], "order_ts"] = "2031-01-01T00:00:00"         # future timestamp
        dups = df.sample(frac=0.01, random_state=self.cfg.seed)           # replayed CDC rows
        return pd.concat([df, dups], ignore_index=True)

    # --------------------------------------------------------------- reviews
    def _review_text(self, rating: int, sub: str) -> tuple[str, str]:
        rng = self.rng
        sub_aspect = SUB_ASPECTS.get(sub)
        pool_pos = ["quality", "delivery", "price", "customer_service", "general"] + ([sub_aspect] if sub_aspect else [])
        pool_neg = pool_pos + ["packaging"]
        parts: list[str] = []
        if rating >= 4:
            for a in rng.choice(pool_pos, size=int(rng.integers(1, 4)), replace=False):
                parts.append(rng.choice(POS[a]))
            if rating == 4 and rng.random() < 0.3:
                parts.append(rng.choice(NEG[rng.choice(["delivery", "price"])]))
        elif rating == 3:
            parts.append(rng.choice(NEUTRAL))
            if rng.random() < 0.5:
                parts.append(rng.choice(POS[rng.choice(pool_pos)]))
            if rng.random() < 0.5:
                parts.append(rng.choice(NEG[rng.choice(pool_neg)]))
        else:
            for a in rng.choice(pool_neg, size=int(rng.integers(1, 4)), replace=False):
                parts.append(rng.choice(NEG[a]))
        text = ". ".join(parts) + "."
        r = rng.random()
        if r < 0.03:
            text = text.replace(". ", ".<br/>")          # HTML from web form
        elif r < 0.05:
            text = text.upper()                           # shouting
        elif r < 0.10:
            text += " " + ("\U0001F60D" if rating >= 4 else "\U0001F621" if rating <= 2 else "")
        return str(rng.choice(TITLES[rating])), text

    def make_reviews(self, orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> list[dict]:
        rng = self.rng
        df = orders[orders["status"] != "cancelled"]
        p_review = np.where(df["status"] == "returned", 0.4, 0.10)
        df = df[rng.random(len(df)) < p_review]
        out = []
        for i, row in enumerate(df.itertuples()):
            ts = pd.Timestamp(row.order_ts) + pd.Timedelta(seconds=int(rng.integers(3 * 86400, 20 * 86400)))
            if ts > self.end + pd.Timedelta(days=1):
                continue
            latent = (3.3 + 2.8 * (self.quality[row.prod_idx] - 0.7) + 0.8 * self.satisfaction[row.cust_idx]
                      - 1.3 * (row.status == "returned") + rng.normal(0, 0.8))
            rating = int(np.clip(round(latent), 1, 5))
            title, body = self._review_text(rating, SUB_NAMES[self.sub_idx[row.prod_idx]])
            rec = {
                "review_id": f"R{i + 1:07d}", "order_id": row.order_id,
                "product_id": products["product_id"].iat[row.prod_idx],
                "customer_id": customers["customer_id"].iat[row.cust_idx],
                "review_ts": ts.isoformat(), "rating": rating, "title": title, "body": body,
            }
            if rng.random() < 0.08:
                rec["rating"] = None                      # free text only (unstructured)
            if rng.random() < 0.01:
                rec["body"] = "   "                       # empty submission
            out.append(rec)
        return out

    def make_social(self) -> list[dict]:
        rng = self.rng
        out, k = [], 1
        total_mult, _ = demand_multipliers(self.dates)
        for d, day in enumerate(self.dates):
            for _ in range(rng.poisson(0.1 * self.p["orders_per_day"] * total_mult[d])):
                sub = SUB_NAMES[int(rng.integers(0, len(SUB_NAMES)))]
                noun = rng.choice(NOUNS[sub]).lower()
                r = rng.random()
                tmpl, emo = ((SOCIAL_POS, "\U0001F525") if r < 0.5 else (SOCIAL_NEG, "\U0001F620") if r < 0.75
                             else (SOCIAL_NEU, ""))
                ts = day + pd.Timedelta(seconds=int(rng.integers(0, 86400)))
                out.append({"post_id": f"SP{k:08d}", "platform": rng.choice(["x", "instagram", "facebook", "tiktok"]),
                            "post_ts": ts.isoformat(), "author": f"user_{rng.integers(1e5, 1e6)}",
                            "text": rng.choice(tmpl).format(n=noun, e=emo) + " #ezyshop",
                            "likes": int(rng.poisson(15)), "shares": int(rng.poisson(2))})
                k += 1
        return out

    # ----------------------------------------------------------- clickstream
    def write_clickstream(self, orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> int:
        rng, cfg = self.rng, self.cfg
        out_dir = cfg.raw_dir / "clickstream"
        out_dir.mkdir(parents=True, exist_ok=True)
        n_days = len(self.dates)
        first_day = n_days - self.p["click_days"]
        pid = products["product_id"].values
        cid = customers["customer_id"].values
        pop = np.exp(self.pop_log)
        pop = pop / pop.sum()
        devices = [("mobile", "android"), ("mobile", "ios"), ("desktop", "windows"), ("desktop", "macos"), ("tablet", "ios")]
        cities = customers["city"].values
        orders = orders.copy()
        orders["day"] = (pd.to_datetime(orders["order_ts"]).dt.normalize() - self.start).dt.days
        total = 0
        ev_no = 0
        sess_no = 0

        def ev(kind, ts, sess, cust, prod=None, extra=None, device=None, city=None):
            nonlocal ev_no
            ev_no += 1
            e = {"event_id": f"E{ev_no:010d}", "event_type": kind, "event_ts": ts.isoformat(),
                 "session_id": sess, "customer_id": cust, "product_id": prod,
                 "page": {"url": f"/p/{prod}" if prod else "/", "referrer": rng.choice(["google", "direct", "email", "social"])},
                 "device": {"type": device[0], "os": device[1]}, "geo": {"city": city}}
            if extra:
                e.update(extra)
            return e

        by_day = {d: g for d, g in orders[orders["day"] >= first_day].groupby("day")}
        for d in range(first_day, n_days):
            day = self.dates[d]
            events = []
            # purchase sessions (one per order)
            g = by_day.get(d)
            if g is not None:
                for oid, lines in g.groupby("order_id", sort=False):
                    sess_no += 1
                    sess = f"S{sess_no:09d}"
                    ci = int(lines["cust_idx"].iat[0])
                    t = pd.Timestamp(lines["order_ts"].iat[0]) - pd.Timedelta(minutes=float(rng.uniform(3, 25)))
                    dev = devices[int(rng.integers(0, 5))]
                    events.append(ev("page_view", t, sess, cid[ci], device=dev, city=cities[ci]))
                    for pi in lines["prod_idx"].values:
                        t += pd.Timedelta(seconds=int(rng.integers(20, 200)))
                        events.append(ev("product_view", t, sess, cid[ci], pid[pi], device=dev, city=cities[ci]))
                        t += pd.Timedelta(seconds=int(rng.integers(10, 90)))
                        events.append(ev("add_to_cart", t, sess, cid[ci], pid[pi], device=dev, city=cities[ci]))
                    t += pd.Timedelta(seconds=int(rng.integers(30, 120)))
                    events.append(ev("checkout", t, sess, cid[ci], device=dev, city=cities[ci]))
                    events.append(ev("purchase", t + pd.Timedelta(seconds=40), sess, cid[ci], device=dev,
                                     city=cities[ci], extra={"order_id": oid}))
            # browse-only sessions; customers close to churning disengage
            active = np.where((self.signup_off <= d) & (self.churn_off > d))[0]
            w = self.activity[active] * np.where(self.churn_off[active] - d < 45, 0.3, 1.0)
            n_browse = int(rng.poisson(1.6 * max(self.daily_orders[d], 1)))
            known = rng.random(n_browse) < 0.7
            custs = rng.choice(active, size=n_browse, p=w / w.sum())
            for b in range(n_browse):
                sess_no += 1
                sess = f"S{sess_no:09d}"
                ci = int(custs[b]) if known[b] else None
                cust = cid[ci] if ci is not None else None
                city = cities[ci] if ci is not None else rng.choice(CITIES, p=CITY_W)
                dev = devices[int(rng.integers(0, 5))]
                t = day + pd.Timedelta(seconds=int(rng.integers(0, 86000)))
                events.append(ev("page_view", t, sess, cust, device=dev, city=city))
                if ci is not None:
                    sc = self.pref_log[ci][self.sub_idx] + self.user_vec[ci] @ self.item_vec.T + self.pop_log
                    views = np.argsort(-(sc + rng.gumbel(size=sc.shape)))[: int(rng.integers(1, 5))]
                else:
                    views = rng.choice(len(pid), size=int(rng.integers(1, 4)), p=pop)
                for pi in views:
                    t += pd.Timedelta(seconds=int(rng.integers(15, 180)))
                    events.append(ev("product_view", t, sess, cust, pid[pi], device=dev, city=city))
                if rng.random() < 0.3:                          # abandoned cart
                    t += pd.Timedelta(seconds=int(rng.integers(20, 120)))
                    events.append(ev("add_to_cart", t, sess, cust, pid[views[0]], device=dev, city=city))
            # ---- veracity defects ----
            lines = []
            for e in events:
                r = rng.random()
                if r < 0.01:
                    e = dict(e, event_ts=int(pd.Timestamp(e["event_ts"]).timestamp() * 1000))  # epoch-ms format
                s = json.dumps(e)
                if r > 0.997:
                    s = s[: len(s) // 2]                     # truncated / malformed record
                lines.append(s)
                if 0.01 <= r < 0.02:
                    lines.append(s)                          # at-least-once duplicate
            (out_dir / f"{day.date().isoformat()}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            total += len(lines)
        return total

    # -------------------------------------------------- inventory & suppliers
    def make_inventory(self, orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        recent = orders[pd.to_datetime(orders["order_ts"]) > self.end - pd.Timedelta(days=28)]
        daily = recent.groupby("prod_idx")["quantity"].sum().reindex(range(len(products)), fill_value=0) / 28
        cover = rng.lognormal(np.log(22), 0.6, size=len(products))
        rows = []
        for i, pidv in enumerate(products["product_id"]):
            total = max(0, int(daily.iat[i] * cover[i] + rng.integers(0, 5)))
            split = rng.dirichlet([3, 3, 2])
            for wh, s in zip(["SYD-DC", "MEL-DC", "BNE-DC"], split):
                rows.append({"snapshot_ts": self.end.isoformat(), "warehouse": wh, "product_id": pidv,
                             "on_hand": int(round(total * s)), "on_order": int(rng.integers(0, 3) * 10)})
        df = pd.DataFrame(rows).astype({"on_hand": "object"})
        bad = rng.random(len(df)) < 0.005
        df.loc[bad, "on_hand"] = None
        return df

    def make_suppliers(self) -> list[dict]:
        rng = self.rng
        out = []
        for s in range(1, 26):
            overseas = rng.random() < 0.5
            rec = {"supplier_id": f"S{s:03d}", "name": f"{rng.choice(BRANDS)} Supply {s}",
                   "country": rng.choice(["China", "Vietnam", "India", "USA"]) if overseas else "Australia",
                   "lead_time_days": int(rng.integers(21, 46) if overseas else rng.integers(4, 15)),
                   "reliability": round(float(rng.uniform(0.8, 0.99)), 2),
                   "contact": {"email": f"orders@supplier{s}.example.com", "phone": None}}
            if rng.random() < 0.1:
                rec.pop("lead_time_days")               # missing attribute
            out.append(rec)
        return out

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.cfg
        cfg.ensure_dirs()
        t0 = time.perf_counter()
        products = self.make_products()
        customers = self.make_customers()
        campaigns = self.make_campaigns()
        orders_idx = self.make_orders(campaigns)
        ucust = customers.drop_duplicates("customer_id").reset_index(drop=True)
        reviews = self.make_reviews(orders_idx, ucust, products)
        social = self.make_social()
        n_click = self.write_clickstream(orders_idx, ucust, products)
        inventory = self.make_inventory(orders_idx, products)
        suppliers = self.make_suppliers()
        orders = self.finalise_orders(orders_idx, ucust, products)

        raw = cfg.raw_dir
        prod_out = products.copy()
        noisy = self.rng.random(len(prod_out)) < 0.05
        prod_out.loc[noisy, "category"] = prod_out.loc[noisy, "category"].map(
            {"Electronics": "electronics ", "Apparel": "APPAREL", "Home": "Home & Living"})
        prod_out["list_price"] = prod_out["list_price"].astype(object)
        dollar = self.rng.random(len(prod_out)) < 0.03
        prod_out.loc[dollar, "list_price"] = prod_out.loc[dollar, "list_price"].map(lambda v: f"${v}")
        prod_out.to_csv(raw / "products.csv", index=False)
        customers.to_csv(raw / "customers.csv", index=False)
        orders.to_csv(raw / "orders.csv", index=False)
        campaigns.to_csv(raw / "campaigns.csv", index=False)
        inventory.to_csv(raw / "inventory_snapshot.csv", index=False)
        (raw / "suppliers.json").write_text(json.dumps(suppliers, indent=2), encoding="utf-8")
        with open(raw / "reviews.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in reviews)
        with open(raw / "social_posts.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in social)
        stats = {"products": len(products), "customers": len(customers), "order_lines": len(orders),
                 "reviews": len(reviews), "social_posts": len(social), "clickstream_events": n_click,
                 "inventory_rows": len(inventory), "campaigns": len(campaigns), "suppliers": len(suppliers),
                 "seconds": round(time.perf_counter() - t0, 1)}
        return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic EzyShop source data")
    ap.add_argument("--scale", choices=["tiny", "small", "medium"], default=None)
    args = ap.parse_args()
    cfg = Config.from_env(scale=args.scale)
    print(json.dumps(SourceGenerator(cfg).run(), indent=2))


if __name__ == "__main__":
    main()
