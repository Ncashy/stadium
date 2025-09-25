import json, time
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from bs4 import BeautifulSoup  # pip install beautifulsoup4

# --- Config ---
endpoint = "https://cdong1--azure-proxy-web-app.modal.run"
api_key = "supersecretkey"
deployment_name = "gpt-4o"
client = OpenAI(base_url=endpoint, api_key=api_key)

INFILE  = Path("data/mlsstadium.txt")                # raw HTML/text
OUTFILE = Path("data/mls_stadiums.json")

TARGET_FIELDS = [
    "image",
    "stadium",
    "team",
    "location",
    "first_mls_year_in_stadium",
    "capacity",
    "opened",
    "surface",
    "roof_type",
]

SYSTEM = (
  "You are a strict data normalizer. Return ONLY a JSON array of objects, "
  "where each object has EXACTLY these keys in this order: "
  '["image","stadium","team","location","first_mls_year_in_stadium","capacity","opened","surface","roof_type"].\n'
  "Rules:\n"
  "- For `image`, ALWAYS use the provided URL from input if not null; do not invent or drop URLs.\n"
  "- For `roof_type`, prefer the `roof_hint` if given; otherwise infer from row text.\n"
  "- Keep values tidy. Use null where unknown.\n"
  "- first_mls_year_in_stadium must be an integer >= 1996 or null.\n"
  "- capacity and opened should be integers if parseable, else null.\n"
  "- Do NOT include commentary or text outside the JSON."
)

def call_llm(batch_rows: List[Dict], max_retries=5) -> List[Dict]:
    """Send one batch of rows (with text + image + roof_hint) to the LLM and parse JSON output."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Extract structured stadium records from this data:\n\n{json.dumps(batch_rows, indent=2)}"}
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0,
                max_tokens=2000,
            )
            text = resp.choices[0].message.content.strip()
            first, last = text.find("["), text.rfind("]")
            if first == -1 or last == -1:
                raise ValueError("No JSON array found in output")
            return json.loads(text[first:last+1])
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"⚠️ Error {e}, retrying in {wait}s...")
            time.sleep(wait)

def main(batch_size=10, delay=3):
    blob = INFILE.read_text(encoding="utf-8")
    soup = BeautifulSoup(blob, "html.parser")

    trs = soup.select("table.wikitable tr")[1:]
    print(f"Found {len(trs)} rows")

    all_results: List[Dict] = []

    for i in range(0, len(trs), batch_size):
        chunk = trs[i:i+batch_size]

        batch_rows = []
        for tr in chunk:
            img = tr.find("img")
            img_url = f"https:{img['src']}" if img and img.has_attr("src") else None

            cols = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            roof_hint = cols[7] if len(cols) > 7 else None

            batch_rows.append({
                "text": " | ".join(cols),
                "image": img_url,
                "roof_hint": roof_hint
            })

        print(f"🔎 Processing rows {i+1}-{min(i+batch_size, len(trs))}...")
        results = call_llm(batch_rows)
        all_results.extend(results)

        time.sleep(delay)  # respect rate limits

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    OUTFILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Extracted {len(all_results)} stadiums → {OUTFILE.resolve()}")


import pandas as pd
from pathlib import Path
import json

INFILE  = Path("data/mls_stadiums.json")
OUTFILE = Path("data/mls_stadiums.csv")

# Load JSON
rows = json.loads(INFILE.read_text(encoding="utf-8"))

# Convert to DataFrame and save as CSV
df = pd.DataFrame(rows)
for col in ["first_mls_year_in_stadium", "capacity", "opened"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").dropna().astype("Int64")
df.to_csv(OUTFILE, index=False)

print(f"✅ Wrote CSV with {len(df)} rows → {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
