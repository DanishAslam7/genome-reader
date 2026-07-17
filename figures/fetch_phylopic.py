#!/usr/bin/env python3
"""
Fetch PhyloPic silhouettes for representative organisms in the cohort (Figure 1a).

API chain (v2):  /nodes?filter_name=NAME&page=0  ->  _links.items[].href
                 -> /nodes/<uuid>                ->  _links.primaryImage
                 -> /images/<uuid>               ->  _links.rasterFiles[].href

PhyloPic images carry CC licences (CC0 / CC-BY / CC-BY-SA). Every download's licence
and attribution is recorded in figures/phylopic/CREDITS.json so the caption can credit
them correctly. Run once; the figure build then needs no network.

  python figures/fetch_phylopic.py
"""
import json, sys, time, urllib.request, urllib.parse
from pathlib import Path

OUT = Path("figures/phylopic"); OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.phylopic.org"
UA = {"User-Agent": "genome-reader-fig1/1.0 (research figure; contact ird601344@iitd.ac.in)"}

# 2 per kingdom. Fall back to a broader clade where PhyloPic has no silhouette for the
# exact species (common for microbial eukaryotes).
WANT = [
    ("animalia", "human",         ["Homo sapiens"]),
    ("animalia", "fly",           ["Drosophila melanogaster", "Drosophila"]),
    ("plantae",  "arabidopsis",   ["Arabidopsis thaliana", "Arabidopsis", "Brassicaceae"]),
    ("plantae",  "maize",         ["Zea mays", "Zea", "Poaceae"]),
    ("fungi",    "yeast",         ["Saccharomyces cerevisiae", "Saccharomyces",
                                   "Saccharomycetaceae", "Ascomycota"]),
    ("fungi",    "neurospora",    ["Neurospora crassa", "Neurospora", "Sordariomycetes",
                                   "Sordariomyceta"]),
    ("protista", "plasmodium",    ["Plasmodium falciparum", "Plasmodium", "Apicomplexa",
                                   "Alveolata"]),
    ("protista", "chlamydomonas", ["Chlamydomonas reinhardtii", "Chlamydomonas",
                                   "Chlorophyta", "Chlorophyceae"]),
]


def get(url):
    if url.startswith("/"):
        url = API + url
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.loads(r.read().decode())


def node_for(name, build):
    # NB two API quirks: filter_name is case-SENSITIVE (must be lowercase), and the
    # API rejects "+" for spaces -> force %20 via quote_via=quote.
    q = urllib.parse.urlencode({"build": build, "filter_name": name.lower(), "page": 0},
                               quote_via=urllib.parse.quote)
    items = get(f"{API}/nodes?{q}").get("_links", {}).get("items") or []
    # prefer an exact title match over a parenthesised parent node
    for it in items:
        if it.get("title", "").lower() == name.lower():
            return it["href"]
    return items[0]["href"] if items else None


def main():
    build = get(f"{API}/")["build"]
    print(f"phylopic build {build}\n")
    credits = []
    for kingdom, slug, candidates in WANT:
        img = used = None
        for name in candidates:
            try:
                href = node_for(name, build)
                if not href:
                    continue
                node = get(href)
                pi = (node.get("_links", {}).get("primaryImage") or {}).get("href")
                if not pi:
                    continue
                img, used = get(pi), name
                break
            except Exception as e:
                print(f"   ! {name}: {type(e).__name__} {e}")
            time.sleep(0.2)
        if not img:
            print(f"{slug:14s} NO SILHOUETTE for {candidates}")
            continue

        L = img.get("_links", {})
        rasters = L.get("rasterFiles") or []
        if not rasters:
            print(f"{slug:14s} no raster files")
            continue
        pick = rasters[-1]                      # largest available
        src = pick["href"]
        if src.startswith("/"):
            src = "https://images.phylopic.org" + src
        dest = OUT / f"{slug}.png"
        with urllib.request.urlopen(urllib.request.Request(src, headers=UA), timeout=30) as r:
            dest.write_bytes(r.read())

        lic = (L.get("license") or {}).get("href", "?")
        attr = img.get("attribution") or "-"
        title = (L.get("self") or {}).get("title", "")
        credits.append({"slug": slug, "kingdom": kingdom, "resolved_from": used,
                        "phylopic_title": title, "license": lic, "attribution": attr,
                        "source": src, "sizes": pick.get("sizes")})
        short = lic.rstrip("/").split("/licenses/")[-1] if "/licenses/" in lic else lic.rstrip("/").split("/")[-1]
        print(f"{slug:14s} <- {used:26s} {short:12s} {attr[:34]:34s} {pick.get('sizes')}")

    json.dump(credits, open(OUT / "CREDITS.json", "w"), indent=2)
    print(f"\nwrote {len(credits)}/{len(WANT)} silhouettes + CREDITS.json")


if __name__ == "__main__":
    sys.exit(main())
