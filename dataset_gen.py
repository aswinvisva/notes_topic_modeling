import numpy as np
import sklearn as sk
import os
import textract
from sklearn.feature_extraction.text import CountVectorizer

class Data_Generator:

    def __init__(self):
        self.X_data = []
        self.paths = []

    def get_notes_pdf(self):

        data = []

        for root, dirs, files in os.walk("/home/aswinvisva/Notability", topdown=False):
            for name in files:
                path = os.path.join(root, name)
                if path.endswith(".pdf"):
                    try:
                        text = textract.process(path, encoding="utf8", errors='ignore')
                        data.append(text)
                        self.X_data.append(text)
                        self.paths.append(os.path.basename(path))
                    except UnicodeDecodeError:
                        pass

        vectorizer = CountVectorizer()
        vectorizer.fit(data)
        X = vectorizer.transform(self.X_data).toarray()

        print(len(self.X_data))
        print(len(X))
        self.X_data = X

        print(len(self.X_data[0]))

