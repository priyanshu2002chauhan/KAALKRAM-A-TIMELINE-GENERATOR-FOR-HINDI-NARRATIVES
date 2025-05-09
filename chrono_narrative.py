import stanza
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.parser import ParserError
from dateutil.relativedelta import relativedelta

class HindiTextProcessor:
    def __init__(self):
        """Initialize the Hindi text processor with Stanza pipeline."""
        try:
            self.nlp = stanza.Pipeline('hi', processors='tokenize,pos,lemma,depparse')
        except Exception as e:
            print(f"Error initializing Stanza pipeline: {e}")
            print("Downloading Hindi model...")
            stanza.download('hi')
            self.nlp = stanza.Pipeline('hi', processors='tokenize,pos,lemma,depparse')
        
        # Common Hindi date patterns
        self.date_patterns = [
            # Full date with Hindi month and year
            r'(\d{1,2})\s+(जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+(\d{4})',
            # Full date with Hindi month (short form) and year
            r'(\d{1,2})\s+(जन|फर|मार्च|अप्रै|मई|जून|जुला|अग|सित|अक्टू|नवं|दिस)\s+(\d{4})',
            # Numeric formats
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            # Year only
            r'(\d{4})\s+में',
            # Month and year
            r'(जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+(\d{4})',
            # Day of week
            r'(सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)'
        ]
        
        # Hindi month mapping (full and short forms)
        self.hindi_months = {
            'जनवरी': '01', 'जन': '01',
            'फरवरी': '02', 'फर': '02',
            'मार्च': '03',
            'अप्रैल': '04', 'अप्रै': '04',
            'मई': '05',
            'जून': '06',
            'जुलाई': '07', 'जुला': '07',
            'अगस्त': '08', 'अग': '08',
            'सितंबर': '09', 'सित': '09',
            'अक्टूबर': '10', 'अक्टू': '10',
            'नवंबर': '11', 'नवं': '11',
            'दिसंबर': '12', 'दिस': '12'
        }

        # Relative date patterns
        self.relative_date_patterns = {
            # Today and yesterday
            r'आज': lambda: datetime.now(),
            r'कल': lambda: datetime.now() - timedelta(days=1),
            r'परसों': lambda: datetime.now() - timedelta(days=2),
            r'आज रात': lambda: datetime.now().replace(hour=20, minute=0, second=0),
            r'कल सुबह': lambda: (datetime.now() - timedelta(days=1)).replace(hour=8, minute=0, second=0),
            
            # This week
            r'इस हफ्ते': lambda: datetime.now() - timedelta(days=datetime.now().weekday()),
            r'पिछले हफ्ते': lambda: datetime.now() - timedelta(days=datetime.now().weekday() + 7),
            r'अगले हफ्ते': lambda: datetime.now() - timedelta(days=datetime.now().weekday() - 7),
            
            # This month
            r'इस महीने': lambda: datetime.now().replace(day=1),
            r'पिछले महीने': lambda: (datetime.now().replace(day=1) - relativedelta(months=1)),
            r'अगले महीने': lambda: (datetime.now().replace(day=1) + relativedelta(months=1)),
            
            # This year
            r'इस साल': lambda: datetime.now().replace(month=1, day=1),
            r'पिछले साल': lambda: datetime.now().replace(month=1, day=1) - relativedelta(years=1),
            r'अगले साल': lambda: datetime.now().replace(month=1, day=1) + relativedelta(years=1),
            
            # Time of day
            r'सुबह': lambda: datetime.now().replace(hour=8, minute=0, second=0),
            r'दोपहर': lambda: datetime.now().replace(hour=12, minute=0, second=0),
            r'शाम': lambda: datetime.now().replace(hour=16, minute=0, second=0),
            r'रात': lambda: datetime.now().replace(hour=20, minute=0, second=0)
        }

        # Time range pattern: e.g., 15 से 20 जनवरी 2024, 1-5 मार्च 2023
        self.range_patterns = [
            r'(\d{1,2})\s*से\s*(\d{1,2})\s+(जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+(\d{4})',
            r'(\d{1,2})-(\d{1,2})\s+(जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+(\d{4})'
        ]

        # Recurring pattern: e.g., हर सोमवार, हर महीने की पहली तारीख
        self.recurring_patterns = [
            (r'हर\s+(सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)', 'weekly'),
            (r'हर\s+महीने\s+की\s+पहली\s+तारीख', 'monthly_first'),
            (r'हर\s+महीने\s+की\s+आखिरी\s+तारीख', 'monthly_last')
        ]

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess the input text.
        
        Args:
            text (str): Input Hindi text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace but preserve single spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters except Hindi characters, numbers, and basic punctuation
        text = re.sub(r'[^\u0900-\u097F\s.,!?:0-9]', '', text)
        return text.strip()

    def extract_time_range(self, text: str) -> Tuple[Optional[datetime], Optional[datetime], str]:
        """
        Extract time range from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[Optional[datetime], Optional[datetime], str]: Start date, end date, and remaining text
        """
        for pattern in self.range_patterns:
            match = re.search(pattern, text)
            if match:
                start_day, end_day, month, year = match.groups()
                month_num = self.hindi_months.get(month, '01')
                try:
                    start_date = datetime(int(year), int(month_num), int(start_day))
                    end_date = datetime(int(year), int(month_num), int(end_day))
                    remaining_text = text.replace(match.group(0), '').strip()
                    return start_date, end_date, remaining_text
                except Exception:
                    continue
        return None, None, text

    def extract_recurring(self, text: str) -> Tuple[Optional[str], str]:
        for pattern, recurrence in self.recurring_patterns:
            match = re.search(pattern, text)
            if match:
                recurring_value = match.group(0)
                remaining_text = text.replace(recurring_value, '').strip()
                return recurrence, remaining_text
        return None, text

    def extract_relative_date(self, text: str) -> Tuple[Optional[datetime], str]:
        """
        Extract relative date from text and return datetime object and remaining text.
        
        Args:
            text (str): Input text containing relative date
            
        Returns:
            Tuple[datetime, str]: Date object and text without date
        """
        for pattern, date_func in self.relative_date_patterns.items():
            match = re.search(pattern, text)
            if match:
                date_str = match.group(0)
                try:
                    date_obj = date_func()
                    remaining_text = text.replace(date_str, '').strip()
                    return date_obj, remaining_text
                except Exception:
                    continue
        return None, text

    def extract_date(self, text: str) -> Tuple[Optional[datetime], str]:
        """
        Extract date from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[Optional[datetime], str]: Date object and remaining text
        """
        # First try relative dates
        date_obj, remaining_text = self.extract_relative_date(text)
        if date_obj:
            return date_obj, remaining_text

        # Then try absolute dates
        for pattern in self.date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if 'में' in pattern:  # Handle year-only format
                        year = int(match.group(1))
                        date_obj = datetime(year, 1, 1)
                    elif len(match.groups()) == 3:  # Day Month Year
                        day = int(match.group(1))
                        month = self.hindi_months.get(match.group(2), '01')
                        year = int(match.group(3))
                        date_obj = datetime(year, int(month), day)
                    elif len(match.groups()) == 2:  # Month Year
                        month = self.hindi_months.get(match.group(1), '01')
                        year = int(match.group(2))
                        date_obj = datetime(year, int(month), 1)
                    else:  # Day of week
                        continue  # Skip day of week as it's not a specific date
                    
                    # Remove the date from the text
                    remaining_text = text.replace(match.group(0), '').strip()
                    return date_obj, remaining_text
                except (ValueError, IndexError):
                    continue
        return None, text

    def split_into_points(self, text: str) -> List[str]:
        """
        Split text into individual points/statements.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of individual points
        """
        # Split by newlines and periods
        points = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Split by periods but keep the period
            sentences = re.split(r'(?<=।)', line)
            points.extend([s.strip() for s in sentences if s.strip()])
        return points

    def process_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Process Hindi text and return structured analysis with chronological ordering.
        
        Args:
            text (str): Input Hindi text
            
        Returns:
            List[Dict[str, Any]]: List of tokens with their linguistic features
        """
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        # Split into points
        points = self.split_into_points(cleaned_text)
        
        # Process each point
        result = []
        for point in points:
            # Check for time range
            start_date, end_date, temp_text = self.extract_time_range(point)
            # Check for recurring
            recurrence, temp_text = self.extract_recurring(temp_text)
            # If not a range, check for single date
            if not start_date and not end_date:
                date_obj, temp_text = self.extract_date(temp_text)
            else:
                date_obj = None
            
            # Process with Stanza
            doc = self.nlp(temp_text)
            
            # Structure the output
            point_info = {
                "text": point,
                "date": date_obj.isoformat() if date_obj else None,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "recurrence": recurrence,
                "tokens": []
            }
            
            # Process each sentence in the document
            for sentence in doc.sentences:
                for word in sentence.words:
                    token_info = {
                        "id": word.id,
                        "text": word.text,
                        "lemma": word.lemma,
                        "upos": word.upos,
                        "head": word.head,
                        "deprel": word.deprel,
                    }
                    point_info["tokens"].append(token_info)
            
            result.append(point_info)
        
        return result

    def process_text_to_json(self, text: str) -> str:
        """
        Process Hindi text and return JSON string.
        
        Args:
            text (str): Input Hindi text
            
        Returns:
            str: JSON string containing the analysis
        """
        result = self.process_text(text)
        # Sort statements chronologically
        sorted_result = sort_statements_chronologically(result)
        return json.dumps(sorted_result, ensure_ascii=False, indent=2)

def sort_statements_chronologically(statements):
    # Sort statements based on their date or recurrence
    def get_sort_key(statement):
        # First check for date ranges
        if statement.get('end_date'):
            return statement['end_date']
        # Then check for single dates
        elif statement.get('date'):
            return statement['date']
        # Then check for start dates
        elif statement.get('start_date'):
            return statement['start_date']
        # For recurring events, use a default date for sorting
        elif statement.get('recurrence'):
            return '9999-12-31T00:00:00'
        # Default for statements without any date information
        return '9999-12-31T00:00:00'

    return sorted(statements, key=get_sort_key)

def main():
    """Example usage of the HindiTextProcessor."""
    processor = HindiTextProcessor()
    
    # Example text with dates and multiple points
    text = """
    15 जनवरी 2024: प्रधानमंत्री ने नई योजना की घोषणा की।
    10 जनवरी 2024: मंत्रिमंडल ने बजट पर चर्चा की।
    कल: सरकार ने नई नीति जारी की।
    आज: विपक्ष ने प्रदर्शन किया।
    """
    
    # Process and print results
    result = processor.process_text_to_json(text)
    print(result)

    # Sample input text
    text = """
    15 से 20 जनवरी 2024 को वार्षिक मेला आयोजित किया गया। हर सोमवार को योग कक्षा होती है। 1-5 मार्च 2023 को परीक्षा आयोजित की गई। हर महीने की पहली तारीख को वेतन वितरित किया जाता है।
    पिछले हफ्ते, सरकार ने नई नीति जारी की। आज सुबह, विपक्ष ने प्रदर्शन किया। 2023 में, पुरानी योजना शुरू हुई। हर महीने की आखिरी तारीख को बैठक आयोजित की जाती है।
    इस महीने, नई परियोजना शुरू हुई। अगले हफ्ते, मंत्रिमंडल की बैठक होगी। हर सोमवार को योग कक्षा होती है। 15 जनवरी 2024 को प्रधानमंत्री ने नई योजना की घोषणा की।
    """
    
    # Process the text
    statements = processor.process_text(text)
    
    # Sort statements chronologically
    sorted_statements = sort_statements_chronologically(statements)
    
    # Print the sorted statements
    print(json.dumps(sorted_statements, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 