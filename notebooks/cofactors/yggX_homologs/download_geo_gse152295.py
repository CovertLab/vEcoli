"""
Download all 32 supplementary processed-data files from GEO accession GSE152295.

GSE152295: Stress-response transcriptomes for 32 human pathogenic bacteria
under 11 infection-related stress conditions (Liébana-García et al. 2020).

Output: data/gse152295/GSE152295_{ABBR}_processed.txt.gz  (inside this folder)
Run:    uv run python download_geo_gse152295.py
"""

from pathlib import Path
import requests

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152295/suppl/"

ORGANISMS = [
    "ACHX",
    "ACIB",
    "AGGA",
    "BBURG",
    "BURK",
    "Campy",
    "ENTFA",
    "EPEC",
    "ETEC",
    "FRAT",
    "HINF",
    "HPG27",
    "HPJ99",
    "KLEBS",
    "LEGIP",
    "Listeria",
    "MRSA",
    "MSSA",
    "MTB",
    "NGON",
    "NMEN",
    "PSEUDO",
    "SALMT",
    "SEPI",
    "SHIF",
    "SPYO",
    "SSUIS",
    "STAGA",
    "STRPN",
    "UPEC",
    "Vibrio",
    "YPSTB",
]

OUT_DIR = Path(__file__).parent / "data" / "gse152295"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_file(abbr):
    filename = f"GSE152295_{abbr}_processed.txt.gz"
    dest = OUT_DIR / filename
    if dest.exists():
        print(f"  skip {filename} (already downloaded)")
        return
    url = BASE_URL + filename
    print(f"  downloading {filename} ...", end=" ", flush=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
    size_kb = dest.stat().st_size // 1024
    print(f"done ({size_kb} KB)")


if __name__ == "__main__":
    print(f"Downloading {len(ORGANISMS)} files to {OUT_DIR}\n")
    errors = []
    for abbr in ORGANISMS:
        try:
            download_file(abbr)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append(abbr)

    print(f"\nDone. {len(ORGANISMS) - len(errors)}/{len(ORGANISMS)} files downloaded.")
    if errors:
        print(f"Failed: {errors}")
