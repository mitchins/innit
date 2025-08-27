# Data Sources and Licensing

This document records the datasets used to train the innit model and their licensing terms.

## English Text Sources

### 1. Project Gutenberg Collections
- **Datasets**: 
  - `laion/Project-Gutenberg` (HuggingFace)
  - `imperial-cpg/project_gutenberg_extended` (HuggingFace)
- **Content**: Public domain books and texts
- **License Status**: 
  - **Text content**: Public domain in the United States
  - **Project Gutenberg trademark**: Applies to redistribution of their ebook files
  - **Usage for training**: ✅ Permitted - we train on the text content, not redistributing ebook files
- **Note**: We do not redistribute Project Gutenberg ebook files or use their trademark in our branding

### 2. Language Identification Dataset
- **Dataset**: `papluca/language-identification` (HuggingFace)
- **License**: Apache 2.0
- **Usage**: English samples (label=0) used as positive examples
- **Content**: Clean, labeled text samples in multiple languages

## Non-English Text Sources

### 1. Language Identification Dataset  
- **Dataset**: `papluca/language-identification` (HuggingFace)
- **License**: Apache 2.0
- **Usage**: Non-English samples (labels≠0) used as negative examples
- **Languages**: Multiple languages including Spanish, French, German, Italian, etc.

### 2. OSCAR (Optional)
- **Dataset**: `oscar-corpus/oscar` (HuggingFace)
- **License**: CommonCrawl terms (research-friendly)
- **Usage**: Non-English web text for additional negative examples
- **Note**: Used sparingly due to size and quality considerations

## Synthetic Data

### Fallback Samples
- **Source**: Hand-crafted examples in the training script
- **Purpose**: Ensure training works even if dataset downloads fail
- **Content**: Simple sentences in English and other languages
- **License**: Created by us, public domain

## Teacher Model (Optional)

### fastText Language Identification
- **Model**: `lid.176.bin` / `lid.176.ftz`
- **License**: Model trained on CC-BY-SA sources
- **Usage**: Optional teacher for distillation (not redistributed)
- **Note**: We use the model's predictions during training but don't ship the weights

## Legal Compliance

### What We Do ✅
- Train on text content that is public domain or permissively licensed
- Create original model weights through our training process
- Use datasets according to their intended research/academic use
- Document all data sources and their licenses
- Release our model under MIT license

### What We Don't Do ❌
- Redistribute raw Project Gutenberg ebook files
- Use the Project Gutenberg trademark in branding
- Include copyrighted text in our model distribution
- Violate any dataset license terms
- Log or store user text by default

## Dataset Usage Summary

| Dataset | License | Usage | Status |
|---------|---------|-------|---------|
| Project Gutenberg texts | Public Domain (US) | English positive samples | ✅ Permitted |
| papluca/language-identification | Apache 2.0 | Both English and non-English samples | ✅ Permitted |
| OSCAR (optional) | CommonCrawl terms | Non-English negative samples | ✅ Research use |
| fastText LID (optional teacher) | CC-BY-SA sources | Training guidance only | ✅ Not redistributed |

## Verification

All datasets are loaded from HuggingFace Hub with their license information clearly documented. Users can verify licensing terms by visiting the dataset pages:

- https://huggingface.co/datasets/laion/Project-Gutenberg
- https://huggingface.co/datasets/papluca/language-identification
- https://huggingface.co/datasets/oscar-corpus/oscar

Last updated: 2024