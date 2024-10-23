import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import gensim.downloader as api
from gensim.models import KeyedVectors


# ===============================================
# 1. Load Pre-trained Models and Measure Time
# ===============================================
def load_models():
    # Measure time for loading Word2Vec (Google News)
    start_time = time.time()
    google_news_path = './models/GoogleNews-vectors-negative300.bin'
    word2vec_model = KeyedVectors.load_word2vec_format(google_news_path, binary=True)
    w2v_load_time = time.time() - start_time

    # Measure time for loading FastText (Wiki)
    start_time = time.time()
    fasttext_model = api.load("fasttext-wiki-news-subwords-300")
    ft_load_time = time.time() - start_time

    # Measure time for loading GloVe
    start_time = time.time()
    glove_vectors = api.load("glove-wiki-gigaword-100")
    glove_load_time = time.time() - start_time

    print('Model load time:')
    print(f"Word2Vec loaded in: {w2v_load_time:.2f} seconds")
    print(f"FastText loaded in: {ft_load_time:.2f} seconds")
    print(f"GloVe loaded in: {glove_load_time:.2f} seconds")

    return word2vec_model, fasttext_model, glove_vectors


# ===============================================
# 2. Measure Similarity Calculation Time
# ===============================================
def similarity(word1, word2, word2vec_model, fasttext_model, glove_vectors):
    # Word2Vec similarity
    start_time = time.time()
    w2v_similarity = cosine_similarity([word2vec_model[word1]], [word2vec_model[word2]])[0][0]
    w2v_time = time.time() - start_time

    # FastText similarity
    start_time = time.time()
    ft_similarity = cosine_similarity([fasttext_model[word1]], [fasttext_model[word2]])[0][0]
    ft_time = time.time() - start_time

    # GloVe similarity
    if word1 in glove_vectors and word2 in glove_vectors:
        start_time = time.time()
        glove_similarity = cosine_similarity([glove_vectors[word1]], [glove_vectors[word2]])[0][0]
        glove_time = time.time() - start_time
    else:
        glove_similarity = "N/A"
        glove_time = 0

    print(f"Word2Vec similarity: {w2v_similarity}")
    print(f"FastText similarity: {ft_similarity}")
    print(f"GloVe similarity: {glove_similarity}")

    print('')
    print(f"Word2Vec similarity calculation time: {w2v_time:.6f} seconds")
    print(f"FastText similarity calculation time: {ft_time:.6f} seconds")
    print(f"GloVe similarity calculation time: {glove_time:.6f} seconds")
    return w2v_similarity, ft_similarity, glove_similarity


# ===============================================
# 3. Function to Check Similarity Accuracy for a Specific Word Pair
# ===============================================
def check_similarity_accuracy_with_human(model, word1, word2, data):
    # Find the human similarity from the dataset
    human_similarity_row = data[(data['word1'] == word1) & (data['word2'] == word2) |
                                (data['word1'] == word2) & (data['word2'] == word1)]

    if human_similarity_row.empty:
        print(f"No human similarity rating found for the word pair: '{word1}' and '{word2}'")
        return None, None

    # Normalize human similarity to be between 0 and 1
    human_similarity = human_similarity_row['similarity'].values[0] / 10.0  # Normalize to 0-1 scale

    # Check if both words are in the model
    if word1 in model and word2 in model:
        # Calculate the predicted similarity using cosine similarity
        predicted_similarity = cosine_similarity([model[word1]], [model[word2]])[0][0]
        # print(f"Predicted similarity between '{word1}' and '{word2}': {predicted_similarity:.4f}")
        # print(f"Human-rated similarity (normalized): {human_similarity:.4f}")

        # Calculate the error (difference between predicted and human similarity)
        error = abs(predicted_similarity - human_similarity)
        print(f"Error (difference between predicted and human similarity): {error:.4f}")

        return predicted_similarity, error
    else:
        print(f"Words '{word1}' or '{word2}' not found in the model.")
        return None, None


# ===============================================
# 4. Evaluate Accuracy using WordSim-353 Dataset
# ===============================================
def evaluate_model(model, data, model_name):
    predicted_similarities = []
    human_similarities = []

    for _, row in data.iterrows():
        word1, word2 = row['word1'], row['word2']
        human_sim = row['similarity']

        if word1 in model and word2 in model:
            pred_sim = cosine_similarity([model[word1]], [model[word2]])[0][0]
            predicted_similarities.append(pred_sim)
            human_similarities.append(human_sim / 10.0)  # Normalize human similarity

    spearman_corr, _ = spearmanr(predicted_similarities, human_similarities)
    print(f"{model_name} Spearman correlation: {spearman_corr}")
    return spearman_corr


# ===============================================
# 5. Main Function
# ===============================================
def main():
    print("\nSelect a type of similarity to use:")
    print("1. Semantic similarity")
    print("2. Relational similarity")
    choice = input("Enter the number of the similarity you want to use: ")

    if choice == '1':
        # Load WordSim-353 similarity dataset
        similarity_data = pd.read_csv('./data/wordsim_similarity_goldstandard.txt', sep='\t',
                                      names=["word1", "word2", "similarity"])
        words = [('tiger', 'cat'), ('planet', 'sun'), ('stock', 'phone'), ('five', 'month'),
                 ('drink', 'ear')]
    elif choice == '2':
        # Load SimLex-999 similarity dataset
        similarity_data = pd.read_csv('./data/wordsim_relatedness_goldstandard.txt', sep='\t',
                                      names=["word1", "word2", "similarity"])
        words = [('computer', 'keyboard'), ('glass', 'magician'), ('stock', 'phone'), ('money', 'operation'),
                 ('car', 'flight')]
    else:
        print("Invalid choice. Exiting.")
        return


    # Load pre-trained models and measure loading time
    word2vec_model, fasttext_model, glove_vectors = load_models()

    for word1, word2 in words:
        print(f"\nSimilarity and calculation time for '{word1}' and '{word2}':")
        w2v_similarity, ft_similarity, glove_similarity = similarity(word1, word2, word2vec_model, fasttext_model, glove_vectors)

        # Load WordSim-353 similarity dataset
        #similarity_data = pd.read_csv('./data/wordsim_relatedness_goldstandard.txt', sep='\t',
        #                              names=["word1", "word2", "similarity"])

        # Check similarity accuracy for a specific word pair
        print(f"\nChecking accuracy for the word pair '{word1}' and '{word2}':")
        check_similarity_accuracy_with_human(word2vec_model, word1, word2, similarity_data)
        check_similarity_accuracy_with_human(fasttext_model, word1, word2, similarity_data)
        check_similarity_accuracy_with_human(glove_vectors, word1, word2, similarity_data)

        # Measure accuracy (Spearman correlation) for Word2Vec, FastText, and GloVe
    print("\nEvaluating accuracy on dataset:")
    w2v_accuracy = evaluate_model(word2vec_model, similarity_data, "Word2Vec")
    ft_accuracy = evaluate_model(fasttext_model, similarity_data, "FastText")
    glove_accuracy = evaluate_model(glove_vectors, similarity_data, "GloVe")


# ===============================================
# Execute the Main Function
# ===============================================
if __name__ == "__main__":
    main()
