"""
synthetic_data.py
-----------------
Generates a realistic corpus of synthetic financial analyst research reports.
Each report includes structured metadata (analyst, sector, region, rating,
coverage years, recommendation change) plus multi-paragraph prose body text.

Usage:
    python src/ingestion/synthetic_data.py --count 500 --output data/raw/
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

# ── Domain constants ──────────────────────────────────────────────────────────

SECTORS = [
    "Technology", "Energy", "Healthcare", "Financials", "Consumer Discretionary",
    "Industrials", "Materials", "Real Estate", "Utilities", "Communication Services",
]

REGIONS = [
    "North America", "Europe", "Emerging Markets", "Asia Pacific",
    "Latin America", "Middle East & Africa",
]

RATINGS = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]

RATING_CHANGES = [
    ("Sell", "Buy"), ("Hold", "Buy"), ("Hold", "Strong Buy"),
    ("Buy", "Hold"), ("Buy", "Sell"), ("Strong Buy", "Hold"),
    ("Sell", "Hold"), ("Strong Sell", "Sell"), ("Buy", "Strong Buy"),
]

COMPANIES = [
    "Nexora Systems", "Valdris Energy", "Orion Biotech", "Kestrel Financial",
    "Lumion Retail", "Stratos Industrial", "Celphos Materials", "Vantara REIT",
    "Gridvolt Utilities", "Meridian Telecom", "Apex Semiconductors",
    "BlueRidge Capital", "Horizon Pharma", "Triton Logistics", "Solera Media",
    "Arclight Renewables", "Fortress Insurance", "Polaris Genomics",
    "Cascade Mining", "Zenith Consumer Brands", "Dataflow Analytics",
    "Omni Dynamics", "Velox Payments", "Sable Aerospace", "Prism Healthcare",
]

THESIS_TEMPLATES = [
    (
        "We initiate coverage of {company} with a {rating} rating and a "
        "12-month price target of ${target}. Our conviction is underpinned by "
        "{driver1} and a structurally improving {driver2} backdrop. "
        "Management has demonstrated consistent execution against stated KPIs, "
        "and we view the current valuation as an attractive entry point relative "
        "to peers in the {sector} space."
    ),
    (
        "Following {company}'s Q{quarter} earnings release, we are {action} our "
        "rating from {old_rating} to {rating}. The revision reflects "
        "{driver1}, partially offset by {headwind}. "
        "Our revised price target of ${target} implies {upside}% upside from "
        "current levels. We believe {sector} tailwinds remain intact over the "
        "medium term, particularly in the {region} market."
    ),
    (
        "We reiterate our {rating} rating on {company} following our annual "
        "deep-dive into {sector} fundamentals across {region}. "
        "{driver1} continues to drive above-consensus revenue growth, while "
        "margin expansion driven by {driver2} supports our above-consensus "
        "EPS estimates for FY{year}. Key risks include {risk1} and {risk2}."
    ),
]

DRIVERS = [
    "accelerating cloud adoption", "favorable commodity price dynamics",
    "pipeline optionality not priced in by the market", "margin expansion",
    "market share gains in core verticals", "a de-leveraging balance sheet",
    "strong free cash flow generation", "regulatory tailwinds",
    "a differentiated technology moat", "robust emerging market demand",
    "disciplined capital allocation", "favourable rate sensitivity",
    "best-in-class management team", "product cycle inflection",
    "improving ESG profile attracting institutional inflows",
]

HEADWINDS = [
    "near-term currency headwinds", "supply chain normalisation pressure",
    "elevated capex requirements", "competitive pricing pressure",
    "potential regulatory scrutiny", "macro sensitivity to rate cycle",
    "customer concentration risk", "integration costs from recent acquisition",
]

RISKS = [
    "geopolitical disruption in key markets", "commodity price volatility",
    "slower-than-expected product ramp", "margin compression from input costs",
    "execution risk on new management strategy", "regulatory overhang",
    "credit market tightening", "FX headwinds in emerging markets",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def random_date(start_year: int = 1995, end_year: int = 2024) -> datetime:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_report_body(company: str, sector: str, region: str,
                         rating: str, old_rating: str | None,
                         target: int, year: int) -> str:
    """Generate a multi-paragraph analyst report body."""
    quarter = random.randint(1, 4)
    action = "upgrading" if old_rating and RATINGS.index(rating) < RATINGS.index(old_rating) else "downgrading"
    driver1, driver2 = random.sample(DRIVERS, 2)
    headwind = random.choice(HEADWINDS)
    risk1, risk2 = random.sample(RISKS, 2)
    upside = random.randint(8, 45)

    template = random.choice(THESIS_TEMPLATES)
    para1 = template.format(
        company=company, rating=rating, target=target,
        driver1=driver1, driver2=driver2, sector=sector,
        region=region, quarter=quarter, action=action,
        old_rating=old_rating or "Hold", headwind=headwind,
        upside=upside, year=year, risk1=risk1, risk2=risk2,
    )

    para2 = (
        f"From a valuation standpoint, {company} trades at {random.randint(12, 35)}x "
        f"forward earnings, a {random.randint(5, 30)}% {'premium' if random.random() > 0.5 else 'discount'} "
        f"to the {sector} peer group median. We use a blended DCF and EV/EBITDA "
        f"methodology to arrive at our ${target} price target, assuming a "
        f"{random.randint(7, 14)}% WACC and terminal growth rate of "
        f"{round(random.uniform(1.5, 3.5), 1)}%. "
        f"Sensitivity analysis suggests the stock remains attractive even under "
        f"our bear case assumptions of {random.randint(3, 8)}% revenue growth."
    )

    para3 = (
        f"In the {region} context, {sector} dynamics have shifted meaningfully "
        f"over the past 18 months. {driver1.capitalize()} has emerged as the "
        f"primary growth catalyst, with management guiding for "
        f"{random.randint(8, 28)}% year-over-year revenue acceleration in the "
        f"back half of FY{year}. We note that {random.choice(DRIVERS)} provides "
        f"an additional option value not reflected in consensus estimates. "
        f"Our channel checks corroborate management's commentary on improving "
        f"demand trends across {region} end markets."
    )

    para4 = (
        f"Key risks to our {rating} thesis include {risk1} and {risk2}. "
        f"Additionally, {headwind} could weigh on near-term results and delay "
        f"the expected re-rating. We would revisit our thesis on evidence of "
        f"sustained margin deterioration below {random.randint(15, 35)}% gross "
        f"margins or a material guidance cut at the next earnings event. "
        f"Investors should be aware of the stock's beta of "
        f"{round(random.uniform(0.7, 1.8), 2)} relative to the broader market."
    )

    return "\n\n".join([para1, para2, para3, para4])


# ── Main generator ────────────────────────────────────────────────────────────

def generate_report(report_id: int) -> dict:
    """Generate a single analyst research report with full metadata."""
    analyst_name = fake.name()
    company = random.choice(COMPANIES)
    sector = random.choice(SECTORS)
    region = random.choice(REGIONS)
    rating = random.choice(RATINGS)

    # Determine if this is a rating change event
    is_change = random.random() < 0.35
    old_rating = None
    change_date = None
    if is_change:
        old_rating, rating = random.choice(RATING_CHANGES)
        change_date = random_date(2020, 2024).isoformat()

    pub_date = random_date(1995, 2024)
    years_coverage = max(1, (datetime(2024, 12, 31) - pub_date).days // 365 +
                         random.randint(0, 5))

    target_price = random.randint(20, 800)

    body = generate_report_body(
        company=company, sector=sector, region=region,
        rating=rating, old_rating=old_rating,
        target=target_price, year=pub_date.year,
    )

    return {
        "report_id": f"RPT-{report_id:05d}",
        "analyst_name": analyst_name,
        "company": company,
        "ticker": f"{company[:3].upper()}{random.randint(10, 99)}",
        "sector": sector,
        "region": region,
        "rating": rating,
        "old_rating": old_rating,
        "is_rating_change": is_change,
        "rating_change_date": change_date,
        "target_price": target_price,
        "publication_date": pub_date.isoformat(),
        "years_coverage": years_coverage,
        "word_count": len(body.split()),
        "body": body,
    }


def generate_corpus(count: int, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    reports = [generate_report(i) for i in range(1, count + 1)]

    # Save full corpus as single JSONL
    jsonl_path = out / "corpus.jsonl"
    with open(jsonl_path, "w") as f:
        for r in reports:
            f.write(json.dumps(r) + "\n")

    # Save summary stats
    stats = {
        "total_reports": len(reports),
        "rating_changes": sum(1 for r in reports if r["is_rating_change"]),
        "sectors": {s: sum(1 for r in reports if r["sector"] == s) for s in SECTORS},
        "regions": {r_: sum(1 for r in reports if r["region"] == r_) for r_ in REGIONS},
        "rating_distribution": {
            rat: sum(1 for r in reports if r["rating"] == rat) for rat in RATINGS
        },
        "avg_years_coverage": round(
            sum(r["years_coverage"] for r in reports) / len(reports), 1
        ),
    }

    stats_path = out / "corpus_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Generated {count} reports → {jsonl_path}")
    print(f"✓ Stats → {stats_path}")
    print(f"\nCorpus summary:")
    print(f"  Rating changes : {stats['rating_changes']} ({stats['rating_changes']/count*100:.0f}%)")
    print(f"  Avg coverage   : {stats['avg_years_coverage']} years")
    print(f"  Sectors        : {len(SECTORS)}")
    print(f"  Regions        : {len(REGIONS)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/raw/")
    args = parser.parse_args()
    generate_corpus(args.count, args.output)