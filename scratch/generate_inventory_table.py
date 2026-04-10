import csv
from pathlib import Path

def get_char_counts():
    data_dir = Path("data")
    counts = {}
    for p in data_dir.glob("*"):
        if p.suffix.lower() in {".md", ".txt"}:
            try:
                content = p.read_text(encoding="utf-8")
                counts[p.stem] = len(content)
            except Exception:
                pass
    return counts

def generate_table():
    counts = get_char_counts()
    csv_path = Path("docs/document_metadata.csv")
    
    if not csv_path.exists():
        print("CSV file not found")
        return

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print("| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |")
        print("|---|--------------|-------|----------|-----------------|")
        for i, row in enumerate(reader, 1):
            file_name = row.get("file_name", "").strip()
            # Mapping .pdf in csv to .md in data folder
            stem = Path(file_name).stem
            count = counts.get(stem, "N/A")
            source = row.get("source", "").strip()
            category = row.get("category", "").strip()
            date = row.get("date", "").strip()
            
            # Shorten source link for readability
            short_source = (source[:30] + "..") if len(source) > 30 else source
            metadata = f"category: {category}, date: {date}"
            
            print(f"| {i} | {file_name} | {short_source} | {count} | {metadata} |")

if __name__ == "__main__":
    generate_table()
