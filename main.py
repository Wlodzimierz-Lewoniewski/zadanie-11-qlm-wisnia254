from collections import Counter
import math
import re


def tokenize(text):
    """Tokenizuje tekst na listę słów, usuwając interpunkcję."""
    text = text.lower()  # Zamiana na małe litery
    text = re.sub(r"[^\w\s]", "", text)  # Usunięcie znaków interpunkcyjnych
    return text.split()


def calculate_term_frequencies(documents):
    """Oblicza częstotliwości terminów w dokumentach i całym korpusie."""
    term_freqs = []
    corpus_freq = Counter()
    for idx, doc in enumerate(documents):
        term_freq = Counter(tokenize(doc))
        term_freqs.append(term_freq)
        corpus_freq.update(term_freq)
    return term_freqs, corpus_freq


def calculate_query_likelihood(query, documents, term_freqs, corpus_freq, lambda_=0.5):
    """Oblicza prawdopodobieństwo wygenerowania zapytania przez każdy dokument."""
    query_tokens = tokenize(query)
    corpus_size = sum(corpus_freq.values())
    scores = []

    for doc_idx, term_freq in enumerate(term_freqs):
        doc_size = sum(term_freq.values())
        score = 0
        for term in query_tokens:
            # Prawdopodobieństwa z wygładzaniem
            p_td = term_freq[term] / doc_size if doc_size > 0 else 0
            p_tc = corpus_freq[term] / corpus_size if corpus_size > 0 else 0
            p_t = lambda_ * p_td + (1 - lambda_) * p_tc
            if p_t > 0:
                score += math.log(p_t)
            else:
                score += float('-inf')  # Unikamy log(0) poprzez dodanie -inf
        scores.append((doc_idx, score))
    return scores


def rank_documents(query, documents, lambda_=0.5):
    """Sortuje dokumenty według prawdopodobieństwa generowania zapytania."""
    term_freqs, corpus_freq = calculate_term_frequencies(documents)
    scores = calculate_query_likelihood(query, documents, term_freqs, corpus_freq, lambda_)
    # Sortowanie: najpierw po score malejąco, a potem po indeksie rosnąco
    scores = sorted(scores, key=lambda x: (-x[1], x[0]))

    return [doc_idx for doc_idx, _ in scores]


# Przykład użycia
if __name__ == "__main__":
    # Pobieranie liczby dokumentów
    n = int(input())
    documents = []

    # Wprowadzanie dokumentów
    for i in range(n):
        doc = input()
        documents.append(doc)

    # Wprowadzenie zapytania
    query = input()

    # Ranking dokumentów
    result = rank_documents(query, documents)
    print(result)

