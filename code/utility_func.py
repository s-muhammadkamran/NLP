from nltk.tokenize import word_tokenize as w_tokenizer

def readfile(file_path, encoding='utf-8'):
    try:
        with open(file_path, encoding=encoding) as file:
            corpus = file.read()
    except Exception as e:
        corpus = ''
        print(f"Error reading corpus: {e}")
    return corpus

def tokenize_words(text):
    if not isinstance(text, str) or not text.strip():
        print("Invalid or empty input to words_tokenize.")
        return []
    return w_tokenizer(text)