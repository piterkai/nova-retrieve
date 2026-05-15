"""Download recent 10-K filings for a list of tickers from SEC EDGAR.

Public API, rate-limited to 10 req/s. Set a real User-Agent (SEC requires it).

Usage:
    python -m scripts.fetch_sec_edgar AAPL MSFT NVDA --years 3 --out data/docs/sec
"""
import argparse
import time
from pathlib import Path

import httpx

UA = "nova-retrieve-research contact@example.com"  # SEC requires a real contact
BASE = "https://data.sec.gov"


def cik_for(ticker: str) -> str:
    r = httpx.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": UA},
        timeout=30,
    )
    r.raise_for_status()
    for row in r.json().values():
        if row["ticker"].upper() == ticker.upper():
            return f"{int(row['cik_str']):010d}"
    raise ValueError(f"Unknown ticker: {ticker}")


def fetch_10k(ticker: str, out_dir: Path, years: int) -> int:
    cik = cik_for(ticker)
    r = httpx.get(
        f"{BASE}/submissions/CIK{cik}.json",
        headers={"User-Agent": UA},
        timeout=30,
    )
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    count = 0
    for form, acc, primary, fdate in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["filingDate"]
    ):
        if form != "10-K":
            continue
        if count >= years:
            break
        acc_clean = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary}"
        dest = out_dir / f"{ticker}_{fdate}_10K.html"
        if not dest.exists():
            doc = httpx.get(url, headers={"User-Agent": UA}, timeout=60)
            doc.raise_for_status()
            dest.write_bytes(doc.content)
            print(f"  saved {dest.name} ({len(doc.content)//1024} KB)")
            time.sleep(0.15)  # be polite, <10 req/s
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--out", default="data/docs/sec")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for t in args.tickers:
        print(f"== {t} ==")
        try:
            total += fetch_10k(t, out, args.years)
        except Exception as e:
            print(f"  skip {t}: {e}")
    print(f"\nDone. {total} filings under {out}")


if __name__ == "__main__":
    main()
