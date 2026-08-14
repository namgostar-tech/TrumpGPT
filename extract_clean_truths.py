import json
import re
import html
import argparse
import os

def clean_truths(input_file="truth_archive.json", output_file="clean_truths.txt"):
    """
    Extracts and normalizes Truth Social posts from a JSON archive.
    Filters out empty posts, links, and retweets, and appends an <|endoftext|> token.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

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
            
            # Skip empty posts
            if not content.strip():
                filtered_empty += 1
                continue
                
            # Skip posts containing external URLs
            if "http://" in content or "https://" in content:
                filtered_url += 1
                continue
                
            # Skip retweets
            if content.startswith("RT @"):
                filtered_rt += 1
                continue
                
            # Clean HTML entities
            content = html.unescape(content)
            # Force single line by removing linebreaks and tabs
            clean_content = re.sub(r'[\r\n\t]+', ' ', content)
            clean_content = clean_content.replace('\xa0', ' ')
            # Collapse multiple spaces
            clean_content = re.sub(r' +', ' ', clean_content).strip()
            
            # Write with end-of-text separator
            f_out.write(clean_content + " <|endoftext|>\n")
            kept_posts += 1

    print("\n--- Extraction Complete ---")
    print(f"Total posts processed: {total_posts}")
    print(f"Filtered out (Empty):  {filtered_empty}")
    print(f"Filtered out (URLs):   {filtered_url}")
    print(f"Filtered out (RTs):    {filtered_rt}")
    print(f"Total valid truths:    {kept_posts}")
    print(f"Saved cleanly to:      {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and normalize Truth Social post archives for TrumpGPT training.")
    parser.add_argument("--input", type=str, default="truth_archive.json", help="Path to raw JSON archive")
    parser.add_argument("--output", type=str, default="clean_truths.txt", help="Path to output clean text file")
    args = parser.parse_args()

    clean_truths(args.input, args.output)
