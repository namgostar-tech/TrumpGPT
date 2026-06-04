import json
import re
import html

def clean_truths():
    input_file = "truth_archive.json"
    output_file = "clean_truths.txt"

    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_posts = len(data)
    kept_posts = 0
    filtered_empty = 0
    filtered_url = 0
    filtered_rt = 0

    with open(output_file, "w", encoding="utf-8") as f_out:
        for d in data:
            content = d.get("content", "")
            
            # skip empty
            if not content.strip():
                filtered_empty += 1
                continue
                
            # skip urls
            if "http://" in content or "https://" in content:
                filtered_url += 1
                continue
                
            # skip retweets
            if content.startswith("RT @"):
                filtered_rt += 1
                continue
                
            # clean text
            content = html.unescape(content)
            # force single line
            clean_content = re.sub(r'[\r\n\t]+', ' ', content)
            clean_content = clean_content.replace('\xa0', ' ')
            # fix spaces
            clean_content = re.sub(r' +', ' ', clean_content).strip()
            
            # write
            f_out.write(clean_content + " <|endoftext|>\n")
            kept_posts += 1

    print("\n--- Extraction Complete ---")
    print(f"Total posts processed: {total_posts}")
    print(f"Filtered out (Empty):  {filtered_empty}")
    print(f"Filtered out (URLs):   {filtered_url}")
    print(f"Filtered out (RTs):    {filtered_rt}")
    print(f"Total pure truths:     {kept_posts}")
    print(f"Saved cleanly to {output_file}")

if __name__ == "__main__":
    clean_truths()
