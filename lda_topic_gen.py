from sklearn.decomposition import LatentDirichletAllocation
import numpy

class Model:

    def fit_transform(self, X_data, topics=5):
        lda = LatentDirichletAllocation(n_components=topics, random_state = 0)
        labels = lda.fit_transform(X_data)
        topic_labels = [numpy.where(arr == numpy.amax(arr))[0][0] for arr in labels]

        return topic_labels