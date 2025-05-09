# ChronoNarrative

A Hindi text processing pipeline for time-based event recognition using Stanza.

## Overview

ChronoNarrative is a text processing system that analyzes Hindi text to identify and structure sentences, focusing on time-based events such as actions, locations, and time references. The system uses Stanza for linguistic analysis including tokenization, lemmatization, POS tagging, and dependency parsing.

## Features

- Text preprocessing and cleaning
- Tokenization of Hindi text
- Lemmatization
- Part of Speech (POS) tagging
- Dependency parsing
- Named Entity Recognition (NER)
- Structured output in JSON format

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ChronoNarrative.git
cd ChronoNarrative
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Hindi language model:
```python
import stanza
stanza.download('hi')
```

## Usage

```python
from chrono_narrative import HindiTextProcessor

# Initialize the processor
processor = HindiTextProcessor()

# Process Hindi text
text = '''
15 से 20 जनवरी 2024: वार्षिक मेला आयोजित किया गया।
हर सोमवार: योग कक्षा होती है।
1-5 मार्च 2023: परीक्षा आयोजित की गई।
हर महीने की पहली तारीख: वेतन वितरित किया जाता है।
'''

result = processor.process_text_to_json(text)
print(result)
```

## Output Example

[
  {
    "text": "15 से 20 जनवरी 2024: वार्षिक मेला आयोजित किया गया।",
    "date": null,
    "start_date": "2024-01-15T00:00:00",
    "end_date": "2024-01-20T00:00:00",
    "recurrence": null,
    "tokens": [...]
  },
  {
    "text": "हर सोमवार: योग कक्षा होती है।",
    "date": null,
    "start_date": null,
    "end_date": null,
    "recurrence": "weekly",
    "tokens": [...]
  }
]