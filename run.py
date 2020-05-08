from dataset_gen import Data_Generator
from lda_topic_gen import Model
import numpy as np

if __name__ == '__main__':
    gen = Data_Generator()
    gen.get_notes_pdf()

    model = Model()

    labels = model.fit_transform((gen.X_data))

    x0 = [gen.paths[i] for i in np.where(np.array(labels) == 0)[0]]
    x1 = [gen.paths[i] for i in np.where(np.array(labels) == 1)[0]]
    x2 = [gen.paths[i] for i in np.where(np.array(labels) == 2)[0]]
    x3 = [gen.paths[i] for i in np.where(np.array(labels) == 3)[0]]
    x4 = [gen.paths[i] for i in np.where(np.array(labels) == 4)[0]]

    print(x0)
    print(x1)
    print(x2)
    print(x3)
    print(x4)