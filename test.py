import stanza
from kaalkram import HindiTextProcessor

# Initialize the processor
processor = HindiTextProcessor()

# Test text with clear examples
text = '''
15 से 20 जनवरी 2024: वार्षिक मेला आयोजित किया गया।
हर सोमवार: योग कक्षा होती है।
1-5 मार्च 2023: परीक्षा आयोजित की गई।
हर महीने की पहली तारीख: वेतन वितरित किया जाता है।
'''

# Process the text
result = processor.process_text_to_json(text)
print("\nProcessed Result:")
print(result) 
