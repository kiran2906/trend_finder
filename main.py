from bs4 import BeautifulSoup
import requests


import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

from collections import Counter

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt

from wordcloud import WordCloud


# List of urls to scrape
urls = [
    # "https://en.wikipedia.org/wiki/Natural_language_processing",
    # "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.kdnuggets.com/2025/02/nettresults/30-must-know-tools-for-python-development",
    # "https://www.wsj.com/",
]

collected_texts = []  # To store the texts collected from each page


# functin to parse text:
def parse_urls(urls):
    collected_texts = []  # To store the texts collected from each page
    for url in urls:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all paragraph texts
            paragraphs = [p.get_text() for p in soup.find_all({"p"})]
            page_txt = " ".join(paragraphs)
            collected_texts.append(page_txt.strip())

        else:
            print(f"Failed to retrieve: {url}")
    return collected_texts


def clean_text(collected_texts):
    stop_words = set(stopwords.words("english"))

    cleaned_texts = []
    for text in collected_texts:
        # Remove all non-alpha carharcters and set all the text to lower case
        text = re.sub(r"[^A-Za-z\s]", " ", text).lower()
        words = [w for w in text.split() if w not in stop_words]
        cleaned_texts.append(" ".join(words))
    return cleaned_texts


def top_ten_words(cleaned_texts):
    top_ten_words = []
    # Combine all texts into one to anlyze overall page content
    all_text = " ".join(cleaned_texts)
    word_counts = Counter(all_text.split())
    top_ten_words = word_counts.most_common(10)

    return top_ten_words


def print_doc_polarity(cleaned_texts):
    for i, text in enumerate(cleaned_texts, 1):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            sentiment = "Positive :)"
        elif polarity < -0.1:
            sentiment = "Negative :("
        else:
            sentiment = "Neutral :|"

        print(f"Document {i} Sentiment: {sentiment} (Polarity:{polarity:.2f})")


def print_word_count_vectorizer(cleaned_texts):
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(cleaned_texts)

    # Fit LDA to find topics (for instance, 3 topics)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()

    for idx, topic in enumerate(lda.components_):
        print(
            f"Topic {idx + 1}: ",
            [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]],
        )


def print_word_cloud(collected_texts):
    # Generate combined text
    combined_text = " ".join(collected_texts)
    # Generate the word cloud
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="viridis"
    ).generate(combined_text)

    # Display the word cloud
    plt.figure(figsize=(10, 6))  # <-- corrected numeric dimensions
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Scraped Text", fontsize=16)
    plt.show()


def main():
    # first collect the text from the urls provided
    collected_texts = parse_urls(urls)

    # Clean the collected texts
    cleaned_texts = clean_text(collected_texts)

    # top_ten worlds
    top_ten = top_ten_words(cleaned_texts)

    # print document polarity
    print_doc_polarity(cleaned_texts)

    # print word count vectorizer
    print_word_count_vectorizer(collected_texts)

    # print word cloud
    print_word_cloud(collected_texts)


if __name__ == "__main__":
    main()
