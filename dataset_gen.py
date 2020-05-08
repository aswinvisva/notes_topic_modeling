import numpy as np
import sklearn as sk
import os
import textract
from sklearn.feature_extraction.text import CountVectorizer

class Data_Generator:

    def __init__(self):
        self.X_data = []

    def get_notes_pdf(self):

        data = []

        for root, dirs, files in os.walk("/home/aswinvisva/Notability", topdown=False):
            for name in files:
                path = os.path.join(root, name)
                if path.endswith(".pdf"):
                    try:
                        text = textract.process(path, encoding="utf8", errors='ignore')
                        data = data + text.split()
                        self.X_data.append(data)
                    except UnicodeDecodeError:
                        print("Passed")

        print(data)
        vectorizer = CountVectorizer()
        vectorizer.fit(data)
        X = []

        for x in self.X_data:
            X.append(vectorizer.transform(x).toarray())

        self.X_data = X

if __name__ == '__main__':
    gen = Data_Generator()
    gen.get_notes_pdf()
    print(gen.X_data)

