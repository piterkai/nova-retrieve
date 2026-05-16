"""Download recent 10-K filings for a list of tickers from SEC EDGAR.

Public API, rate-limited to 10 req/s. Set a real User-Agent (SEC requires it).

Usage:
    # tickers inline
    python -m scripts.fetch_sec_edgar AAPL MSFT NVDA --years 3 --out data/docs/sec

    # tickers from a file (one or more per line, blank lines and # comments ok)
    python -m scripts.fetch_sec_edgar --tickers-file data/sp100.txt --years 5

    # both sources are merged (de-duplicated, order preserved)
    python -m scripts.fetch_sec_edgar TSLA --tickers-file data/sp100.txt --years 5
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List

import httpx

UA = "nova-retrieve-research contact@example.com"  # SEC requires a real contact
BASE = "https://data.sec.gov"


def _norm(ticker: str) -> str:
    # SEC's company_tickers.json uses dashes for share classes (e.g. BRK-B);
    # accept the common dotted form (BRK.B) too.
    return ticker.strip().upper().replace(".", "-")


def load_ticker_map() -> Dict[str, str]:
    """Fetch SEC's ticker directory once and return {NORMALIZED_TICKER: cik10}."""
    r = httpx.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": UA},
        timeout=30,
    )
    r.raise_for_status()
    return {
        _norm(row["ticker"]): f"{int(row['cik_str']):010d}"
        for row in r.json().values()
    }


def read_tickers_file(path: Path) -> List[str]:
    """One or more tickers per line; blank lines and `#` comments ignored."""
    tickers: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tickers.extend(line.replace(",", " ").split())
    return tickers


def fetch_10k(ticker: str, cik: str, out_dir: Path, years: int) -> int:
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
    ap.add_argument("tickers", nargs="*", help="ticker symbols (optional if --tickers-file given)")
    ap.add_argument("--tickers-file", type=Path, help="file with tickers (one+ per line, # comments ok)")
    ap.add_argument("--years", type=int, default=3, help="how many most-recent 10-K filings per company")
    ap.add_argument("--out", default="data/docs/sec")
    args = ap.parse_args()

    # merge inline + file tickers, de-dupe, preserve order
    raw: List[str] = list(args.tickers)
    if args.tickers_file:
        raw += read_tickers_file(args.tickers_file)
    seen = set()
    tickers: List[str] = []
    for t in raw:
        n = _norm(t)
        if n and n not in seen:
            seen.add(n)
            tickers.append(t.strip())
    if not tickers:
        ap.error("no tickers given — pass them as arguments or via --tickers-file")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading SEC ticker directory ({len(tickers)} companies requested)…")
    ticker_map = load_ticker_map()

    total = 0
    for t in tickers:
        print(f"== {t} ==")
        cik = ticker_map.get(_norm(t))
        if not cik:
            print(f"  skip {t}: unknown ticker")
            continue
        try:
            total += fetch_10k(t, cik, out, args.years)
        except Exception as e:
            print(f"  skip {t}: {e}")
    print(f"\nDone. {total} filings under {out}")


if __name__ == "__main__":
    main()
