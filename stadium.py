import requests
from bs4 import BeautifulSoup
from pathlib import Path

URL = "https://en.wikipedia.org/wiki/List_of_Major_League_Soccer_stadiums"
OUT = Path("data/mlsstadium.txt")
MODE = "raw_html"

def scrape_table(url=URL, mode=MODE) -> str:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (TextScraper/1.0)"}, timeout=20)
    r.raise_for_status()
    
    if mode == "raw_html":
        return r.text
    
    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        a.insert_after(f" [LINK:{a['href']}]")
    for img in soup.find_all("img"):
        img.insert_after(f" [IMG alt='{img.get('alt', '')}', src='{img.get('src', '')}']")

if __name__ == "__main__":
    blob = scrape_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(blob, encoding="utf-8")
    print(f"Wrote clean stadium table ({len(blob)} chars) to {OUT.resolve()}")
    
    #supa-secret-password