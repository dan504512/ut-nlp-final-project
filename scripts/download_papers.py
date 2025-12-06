#!/usr/bin/env python3
"""
Script to download all papers referenced in the project.md file.
Papers are organized by year and saved with descriptive names.
"""

import os
import time
import requests
from typing import Dict, List, Tuple

# Create papers directory if it doesn't exist
PAPERS_DIR = "papers"
os.makedirs(PAPERS_DIR, exist_ok=True)

# Map of paper citations to their URLs (when available via arXiv or open access)
# Format: (citation_key, filename, url)
PAPERS = [
    # 2015
    ("Bowman et al.2015", "bowman2015_snli.pdf", 
     "https://arxiv.org/pdf/1508.05326.pdf"),
    
    # 2016
    ("Rajpurkar et al.2016", "rajpurkar2016_squad.pdf",
     "https://arxiv.org/pdf/1606.05250.pdf"),
    
    # 2017
    ("Jia and Liang2017", "jia2017_adversarial_squad.pdf",
     "https://arxiv.org/pdf/1707.07328.pdf"),
    
    # 2018
    ("Williams et al.2018", "williams2018_multinli.pdf",
     "https://arxiv.org/pdf/1704.05426.pdf"),
    ("Yang et al.2018", "yang2018_hotpotqa.pdf",
     "https://arxiv.org/pdf/1809.09600.pdf"),
    ("Poliak et al.2018", "poliak2018_hypothesis_only.pdf",
     "https://arxiv.org/pdf/1803.02324.pdf"),
    ("Glockner et al.2018", "glockner2018_breaking_nli.pdf",
     "https://arxiv.org/pdf/1810.09774.pdf"),
    ("Kaushik and Lipton2018", "kaushik2018_reading_comprehension.pdf",
     "https://arxiv.org/pdf/1808.04926.pdf"),
    
    # 2019
    ("Chen and Durrett2019", "chen2019_multihop_reasoning.pdf",
     "https://arxiv.org/pdf/1904.04686.pdf"),
    ("Clark et al.2019", "clark2019_dont_take_easy_way.pdf",
     "https://arxiv.org/pdf/1909.03683.pdf"),
    ("He et al.2019", "he2019_unlearn_dataset_bias.pdf",
     "https://arxiv.org/pdf/1909.03496.pdf"),
    ("Liu et al.2019", "liu2019_inoculation.pdf",
     "https://arxiv.org/pdf/1906.02390.pdf"),
    ("McCoy et al.2019", "mccoy2019_right_wrong_reasons.pdf",
     "https://arxiv.org/pdf/1902.01007.pdf"),
    ("Wallace et al.2019", "wallace2019_universal_triggers.pdf",
     "https://arxiv.org/pdf/1908.07125.pdf"),
    
    # 2020
    ("Clark et al.2020", "clark2020_electra.pdf",
     "https://arxiv.org/pdf/2003.10555.pdf"),
    ("Gardner et al.2020", "gardner2020_contrast_sets.pdf",
     "https://arxiv.org/pdf/2004.02709.pdf"),
    ("Ribeiro et al.2020", "ribeiro2020_checklist.pdf",
     "https://arxiv.org/pdf/2005.04118.pdf"),
    ("Swayamdipta et al.2020", "swayamdipta2020_dataset_cartography.pdf",
     "https://arxiv.org/pdf/2009.10795.pdf"),
    ("Zhou and Bansal2020", "zhou2020_robustifying_nli.pdf",
     "https://arxiv.org/pdf/2004.14648.pdf"),
    ("Utama et al.2020", "utama2020_debiasing_nlu.pdf",
     "https://arxiv.org/pdf/2009.12303.pdf"),
    ("Nie et al.2020", "nie2020_collective_opinions.pdf",
     "https://arxiv.org/pdf/2010.03532.pdf"),
    ("Morris et al.2020", "morris2020_textattack.pdf",
     "https://arxiv.org/pdf/2005.05909.pdf"),
    ("Bartolo et al.2020", "bartolo2020_beat_the_ai.pdf",
     "https://arxiv.org/pdf/2002.00293.pdf"),
    
    # 2021
    ("Gardner et al.2021", "gardner2021_competency_problems.pdf",
     "https://arxiv.org/pdf/2104.08646.pdf"),
    ("Dua et al.2021", "dua2021_instance_bundles.pdf",
     "https://arxiv.org/pdf/2112.08633.pdf"),
    ("Sanh et al.2021", "sanh2021_learning_from_mistakes.pdf",
     "https://arxiv.org/pdf/2012.01958.pdf"),
    ("Meissner et al.2021", "meissner2021_embracing_ambiguity.pdf",
     "https://arxiv.org/pdf/2105.14574.pdf"),
    ("Yaghoobzadeh et al.2021", "yaghoobzadeh2021_forgettable_examples.pdf",
     "https://arxiv.org/pdf/2104.04374.pdf"),
]


def download_paper(citation: str, filename: str, url: str) -> bool:
    """
    Download a single paper.
    
    Args:
        citation: Citation key for the paper
        filename: Filename to save as
        url: URL to download from
        
    Returns:
        True if successful, False otherwise
    """
    filepath = os.path.join(PAPERS_DIR, filename)
    
    # Skip if already downloaded
    if os.path.exists(filepath):
        print(f"✓ {citation} already downloaded")
        return True
    
    try:
        print(f"Downloading {citation}...", end=" ")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print("✓")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return False


def main():
    """Download all papers."""
    print(f"Downloading papers to {PAPERS_DIR}/")
    print("=" * 50)
    
    successful = 0
    failed = []
    
    for citation, filename, url in PAPERS:
        if download_paper(citation, filename, url):
            successful += 1
        else:
            failed.append(citation)
        
        # Be polite to servers
        time.sleep(0.5)
    
    print("=" * 50)
    print(f"\nDownload complete!")
    print(f"Successful: {successful}/{len(PAPERS)}")
    
    if failed:
        print(f"Failed downloads: {', '.join(failed)}")
        print("\nNote: Some papers may require manual download from:")
        print("- ACL Anthology: https://aclanthology.org/")
        print("- NeurIPS Proceedings: https://papers.nips.cc/")
        print("- ICLR OpenReview: https://openreview.net/")
    
    # Create a README for the papers directory
    readme_content = """# Papers Directory

This directory contains papers referenced in the CS388 NLP Final Project.

## Papers by Year

### 2015
- Bowman et al. - Stanford NLI dataset

### 2016
- Rajpurkar et al. - SQuAD dataset

### 2017
- Jia and Liang - Adversarial examples for reading comprehension

### 2018
- Williams et al. - MultiNLI
- Yang et al. - HotpotQA
- Poliak et al. - Hypothesis-only baselines in NLI
- Glockner et al. - Breaking NLI systems
- Kaushik and Lipton - How much reading does reading comprehension require?

### 2019
- Chen and Durrett - Understanding dataset design choices for multi-hop reasoning
- Clark et al. - Don't take the easy way out
- He et al. - Unlearn dataset bias
- Liu et al. - Inoculation by fine-tuning
- McCoy et al. - Right for the wrong reasons
- Wallace et al. - Universal adversarial triggers

### 2020
- Clark et al. - ELECTRA
- Gardner et al. - Contrast sets
- Ribeiro et al. - CheckList
- Swayamdipta et al. - Dataset cartography
- Zhou and Bansal - Robustifying NLI models
- Utama et al. - Debiasing NLU models
- Nie et al. - Collective human opinions
- Morris et al. - TextAttack
- Bartolo et al. - Beat the AI

### 2021
- Gardner et al. - Competency problems
- Dua et al. - Learning with instance bundles
- Sanh et al. - Learning from others' mistakes
- Meissner et al. - Embracing ambiguity
- Yaghoobzadeh et al. - Forgettable examples
"""
    
    with open(os.path.join(PAPERS_DIR, "README.md"), "w") as f:
        f.write(readme_content)


if __name__ == "__main__":
    main()