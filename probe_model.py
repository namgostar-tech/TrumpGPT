import os
import re
import random
import torch
from collections import Counter

# Import train components for generation
from train import BigramLanguageModel, encode, decode, DEVICE

# Files
DATASET_FILE = "clean_truths.txt"
GEN_FILE = "1000_generations.txt"

# Target groups
ALLIES = ['republican', 'republicans', 'vance', 'maga', 'patriots', 'trump', 'donald', 'conservatives', 'gop', 'cruz', 'desantis', 'america first', 'musk', 'tucker', 'supporters', 'right']
ENEMIES = ['democrat', 'democrats', 'biden', 'judge', 'judges', 'merchan', 'engoron', 'smith', 'jack', 'fake news', 'marxists', 'communists', 'left', 'liberals', 'pelosi', 'schumer', 'harris', 'kamala', 'obama', 'fbi', 'doj', 'media', 'swamp', 'deep state', 'rinos', 'socialists']

STOP_WORDS = set([
    'the', 'is', 'at', 'of', 'and', 'a', 'to', 'in', 'that', 'it', 'for', 'on', 'with', 
    'as', 'this', 'was', 'but', 'they', 'are', 'we', 'by', 'be', 'have', 'an', 'what',
    'not', 'all', 'were', 'from', 'when', 'so', 'if', 'or', 'you', 'he', 'his', 'him',
    'she', 'her', 'has', 'our', 'who', 'will', 'my', 'their', 'about', 'out', 'up',
    'very', 'just', 'can', 'do', 'would', 'there', 'which', 'one', 'been', 'no', 'out',
    'me', 'your', 'i', 'than', 'them', 'because', 'could', 'some', 'any', 'these',
    'its', 'now', 'how', 'did', 'am', 'then', 'only', 'also', 'those', 'other', 'into',
    'us', 'much', 'over', 'even', 'most', 'such', 'well', 'where', 'why', 'being',
    'like', 'too', 'many', 'get', 'fake', 'news', 'president', 'people', 'country', 'state',
    'has', 'had', 'time', 'more', 'new', 'no', 'make', 'great', 'again', 'america', 'american'
])

def generate_samples(num_samples=1000):
    print(f"Generating {num_samples} samples... This may take a while.")
    model = BigramLanguageModel()
    m = model.to(DEVICE)
    m.load_state_dict(torch.load('TrumpGPT-Base-v1.pth', map_location=DEVICE))
    m.eval()

    with open(GEN_FILE, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            if (i+1) % 50 == 0:
                print(f"Generating sample {i+1}/{num_samples}...")
            torch.seed()
            start_context = torch.tensor(encode('\n'), dtype=torch.long, device=DEVICE).unsqueeze(0)
            generated_tokens = m.generate(start_context, max_new_tokens=500)[0].tolist()
            text = decode(generated_tokens)
            f.write(f"========== GENERATION {i+1} ==========\n")
            f.write(text.strip() + "\n\n")
    print(f"Finished generating {num_samples} samples to {GEN_FILE}.")

def extract_sentences(text, keywords=None):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if not keywords:
        return [s.strip().replace('\n', ' ') for s in sentences if len(s.strip()) > 10]
    
    matches = []
    for s in sentences:
        s_lower = s.lower()
        if any(kw in s_lower for kw in keywords):
            matches.append(s.strip().replace('\n', ' '))
    return matches

def analyze_text(text, name):
    print(f"\n{'='*50}\n--- Analysis for {name} ---\n{'='*50}")
    
    # Strip generation artifacts
    text = re.sub(r'========== GENERATION \d+ ==========', '', text)
    text = text.replace('<|endoftext|>', '')
    
    words_raw = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    total_words = len(words_raw) if words_raw else 1
    
    caps_words = [w for w in words_raw if w.isupper()]
    caps_ratio = len(caps_words) / total_words * 100
    
    exclamation_count = text.count('!')
    exclamations_per_1k = (exclamation_count / total_words) * 1000
    
    print("\n--- Stylistic Transfer Metrics ---")
    print(f"Percentage of ALL CAPS words: {caps_ratio:.2f}%")
    print(f"Exclamation Marks per 1000 words: {exclamations_per_1k:.2f}")

    all_sentences = extract_sentences(text)
    all_sentences.sort(key=lambda s: len(s.split()), reverse=True)
    
    print("\n--- Semantic Logic Breakdown (Longest Sentences) ---")
    print("Notice how longer generations often devolve into structural gibberish while maintaining 'Trumpian' style:")
    for i, s in enumerate(all_sentences[:3]):
        print(f" {i+1}. {s}")

    print("\n--- Word Association Probes (PMI within sentences) ---")
    words_lower = [w.lower() for w in words_raw]
    
    import math
    N = len(all_sentences)
    
    def get_pmi(target_list, min_cooccur=3):
        target_words = set(target_list)
        word_doc_counts = Counter()
        cooccur_counts = Counter()
        target_doc_count = 0
        
        for s in all_sentences:
            words_in_s = [w.lower() for w in re.findall(r'\b[a-zA-Z]{2,}\b', s)]
            unique_words = set(words_in_s)
            
            has_target = any(tw in unique_words for tw in target_words)
            if not has_target and 'fake news' in target_list:
                if 'fake' in words_in_s:
                    idx = words_in_s.index('fake')
                    if idx + 1 < len(words_in_s) and words_in_s[idx+1] == 'news':
                        has_target = True

            if has_target:
                target_doc_count += 1
                
            valid_words = set(w for w in unique_words if w not in STOP_WORDS and w not in target_words)
            for w in valid_words:
                word_doc_counts[w] += 1
                if has_target:
                    cooccur_counts[w] += 1
                    
        pmi_scores = {}
        if target_doc_count == 0 or N == 0:
            return Counter()
            
        p_y = target_doc_count / N
        for w, co_count in cooccur_counts.items():
            if co_count >= min_cooccur:
                p_x = word_doc_counts[w] / N
                p_xy = co_count / N
                pmi = math.log2(p_xy / (p_x * p_y))
                pmi_scores[w] = pmi
                
        return Counter(pmi_scores)

    allies_co = get_pmi(ALLIES)
    enemies_co = get_pmi(ENEMIES)
    
    print(f"\nTop 10 words associated with Allies in {name} (PMI score):")
    for w, c in allies_co.most_common(10):
        print(f"  {w}: {c:.3f}")
        
    print(f"\nTop 10 words associated with Enemies in {name} (PMI score):")
    for w, c in enemies_co.most_common(10):
        print(f"  {w}: {c:.3f}")


def main():
    # 1. Generate new samples
    # generate_samples(1000)

    # 2. Analyze Dataset
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            dataset_text = f.read()
        analyze_text(dataset_text, "Original Dataset (clean_truths.txt)")
    except Exception as e:
        print(f"Could not read {DATASET_FILE}: {e}")

    # 3. Analyze Generations
    try:
        with open(GEN_FILE, 'r', encoding='utf-8') as f:
            gen_text = f.read()
        analyze_text(gen_text, "Generated Output (1000_generations.txt)")
    except Exception as e:
        print(f"Could not read {GEN_FILE}: {e}")

if __name__ == "__main__":
    main()
