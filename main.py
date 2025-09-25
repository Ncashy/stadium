import json, time
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from bs4 import BeautifulSoup  # new dependency

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
  '["image","stadium","team","location","first_mls_year_in_stadium","capacity","opened","surface","roof_type"]. '
  "Rules:\n"
  "- Keep values tidy. Use null where unknown.\n"
  "- first_mls_year_in_stadium must be an integer >= 1996 or null.\n"
  "- capacity and opened should be integers if parseable, else null.\n"
  "- Preserve image as string (if present in HTML) else null.\n"
  "- Do NOT include commentary or extra text outside the JSON."
)

def call_llm(batch_text: str, max_retries=5) -> List[Dict]:
    """Send one batch of rows to the LLM and parse JSON output."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Extract structured stadium records from this raw table text:\n\n{batch_text}"}
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0,
                max_tokens=2000,   # smaller since we send fewer rows
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

def main(batch_size=5, delay=3):
    # Load raw HTML
    blob = INFILE.read_text(encoding="utf-8")
    soup = BeautifulSoup(blob, "html.parser")

    # Extract table rows (skip header row)
    trs = soup.select("table.wikitable tr")[1:]
    print(f"Found {len(trs)} rows")

    all_results: List[Dict] = []

    # Process in batches
    for i in range(0, len(trs), batch_size):
        chunk = trs[i:i+batch_size]
        # Join text of rows for this batch
        batch_text = "\n".join(tr.get_text(" ", strip=True) for tr in chunk)

        print(f"🔎 Processing rows {i+1}-{min(i+batch_size, len(trs))}...")
        results = call_llm(batch_text)
        all_results.extend(results)

        # Respect rate limits
        time.sleep(delay)

    # Save combined results
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    OUTFILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Extracted {len(all_results)} stadiums → {OUTFILE.resolve()}")

if __name__ == "__main__":
    main()