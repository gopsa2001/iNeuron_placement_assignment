'''
[ NLP ]

Question - 2:-

Take any pdf and your task is to extract the text from that pdf and store it in a
csv file and then you need to find the most repeated word in that pdf.


'''

## Answer-2 [NLP]:-


import PyPDF2
import pandas as pd
from collections import Counter

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Path to your PDF file
pdf_path = "path/to/your/pdf/file.pdf"

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Store text in a CSV file
csv_file = "output.csv"
with open(csv_file, "w") as file:
    file.write(pdf_text)

# Find the most repeated word
word_count = Counter(pdf_text.split())
most_common_word = word_count.most_common(1)[0][0]

print("Most repeated word:", most_common_word)

'''
## Thank You
'''
