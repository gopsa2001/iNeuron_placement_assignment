'''
[ NLP ]

Question 4 :-

Take any text file and now your task is to Text Summarization without using
hugging transformer library.


'''

## Answer - 4 [ NLP ]:-


import re
from collections import defaultdict

def preprocess_text(text):
    # Remove special characters and symbols
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

def calculate_sentence_scores(sentences):
    # Initialize word frequency dictionary
    word_freq = defaultdict(int)
    for sentence in sentences:
        # Split sentence into words
        words = sentence.split()
        for word in words:
            # Update word frequency
            word_freq[word] += 1

    sentence_scores = defaultdict(int)
    for sentence in sentences:
        # Split sentence into words
        words = sentence.split()
        for word in words:
            # Accumulate importance score based on word frequency
            sentence_scores[sentence] += word_freq[word]

    return sentence_scores

def extract_summary(text, num_sentences):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Split text into sentences
    sentences = preprocessed_text.split(". ")

    # Calculate sentence scores
    sentence_scores = calculate_sentence_scores(sentences)

    # Sort sentences based on scores in descending order
    sorted_sentences = sorted(
        sentence_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Select the top 'num_sentences' sentences for the summary
    summary_sentences = [sentence for sentence, _ in sorted_sentences[:num_sentences]]

    # Generate the summary
    summary = ". ".join(summary_sentences)

    return summary

# Example usage
input_text = """
The quick brown fox jumps over the lazy dog.
The dog barks at the cat.
The cat chases the mouse.
The mouse runs away.
"""

num_summary_sentences = 3
summary = extract_summary(input_text, num_summary_sentences)
print(summary)


## Thank You 

