

import sys
import os
import glob
import html2text
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords


args = sys.argv
path = args[1]
current_path = os.getcwd()
filepath = current_path+"/"+path+"/*.html"
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#path = '/Users/LibUser/Desktop/MoraisChelseaHW4/input'
documents = []
#filepath = path+"/*.html"
files=glob.glob(filepath)
for filename in files:
    f = open(filename, 'rb')
    text = html2text.html2text(f.read().decode(errors='replace'))
    text = text.replace("\n","").lower()
    text = re.sub(pattern="[^\w\s]", repl="", string=text)   # Remove all punctuations
    text = re.sub(pattern="[0-9]*", repl="", string=text)    # Remove all numbers
    text = re.sub(pattern="_*", repl="", string=text)
    text_tokens = text.split()
    filtered_text = [word for word in text_tokens if (word not in stop_words) and (len(word) > 1)]
    documents.append(" ".join(filtered_text))

# Obtaining the TF-IDF scores
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Calculate pairwise similarity matrix using cosine similarity
similarity_matrix = 1 - np.dot(X, X.T).toarray() # Conversion to dense matrix

# Initialization of each document as a separate cluster
clusters = [[i] for i in range(len(documents))]

# Agglomerative clustering using single link method
while True:
    # Find two clusters with highest similarity
    max_sim = -1
    max_i, max_j = -1, -1
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            sim = np.min(similarity_matrix[clusters[i], :][:, clusters[j]])
            if sim > max_sim:
                max_sim = sim
                max_i, max_j = i, j
    
    # Stop clustering if similarity is below threshold
    if max_sim < 0.4:
        break
    
    # Merge clusters and update similarity matrix
    clusters[max_i] += clusters[max_j]
    del clusters[max_j]
    for i in range(len(clusters)):
        if i != max_i:
            min_sim = np.min(similarity_matrix[clusters[max_i], :][:, clusters[i]])
            similarity_matrix[clusters[max_i], :][:, clusters[i]] = min_sim
            similarity_matrix[clusters[i], :][:, clusters[max_i]] = min_sim
    
    # Indicate which objects were merged
    if len(clusters) > max_i and len(clusters) > max_j:
        print(f"Merged {clusters[max_j]} into {clusters[max_i]} (similarity={max_sim:.4f})")

    # Set similarity scores of merged cluster to 1.0 to avoid merging with other clusters
    similarity_matrix[clusters[max_i], :][:, clusters[max_i]] = 1.0

print("\n\n\n")
print("Number of clusters:",len(clusters))



min_similarity  = 1
min_i, min_j = -1, -1
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        s = similarity_matrix[i, j]
        if s < min_similarity:
            min_similarity = s
            min_i, min_j = i, j

print(f"Most dissimilar pair: Document {min_i+1} and Document {min_j+1} (similarity={min_similarity:.4f})")


max_similarity = -1
max_i, max_j = -1, -1
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        s = similarity_matrix[i, j]
        if s > max_similarity:
            max_similarity = s
            max_i, max_j = i, j

print(f"Most similar pair: Document {max_i+1} and Document {max_j+1} (similarity={max_similarity :.4f})")



centroid = np.mean(X, axis=0)
distances = np.sum(np.square(X - centroid), axis=1)
closest = np.argmin(distances)
print(f"Closest document to corpus centroid: Document {closest+1}")





