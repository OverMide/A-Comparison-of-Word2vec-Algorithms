## Downloading Required Files

### Pre-trained Models

1. **Word2Vec (Google News)**:
   - Download the pre-trained Word2Vec model [here](https://code.google.com/archive/p/word2vec/).
   - Save the file in the `models` directory and rename it to `GoogleNews-vectors-negative300.bin`.

2. **FastText (Wiki)**:
   - The FastText model will be automatically downloaded by the Gensim library during code execution.

3. **GloVe**:
   - The GloVe model will also be automatically downloaded via Gensim, but you can manually download it [here](https://nlp.stanford.edu/projects/glove/).

### Datasets

1. **WordSim-353 Relatedness**:
   - Download the dataset from [here](http://www.gabrilovich.com/resources/data/wordsim353/).
   - Place it in the `data/` folder and ensure the file is named `wordsim_relatedness_goldstandard.txt`.
2. **WordSim-353 similarity**:
   - Download the dataset from [here](http://www.gabrilovich.com/resources/data/wordsim353/).
   - Place it in the `data/` folder and ensure the file is named `wordsim_similarity_goldstandard.txt`.