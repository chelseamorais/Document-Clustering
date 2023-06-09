INTRODUCTION

This project implements agglomerative clustering using the single-link method to the collection of HTML documents based on their textual content. It employs text preprocessing techniques including lowercasing, removing punctuation and numbers, and removing stop words. Following that, the documents' Term Frequency-Inverse Document Frequency (TF-IDF) representation is calculated, and a pairwise similarity matrix is produced using cosine similarity. The most similar clusters are iteratively combined until a certain threshold is attained. In addition, it lists the total number of clusters, the most different and similar document pairs, and the document that is the closest to the collection's centroid.

Input : python3 Clusters.py input 

Output : Print statements

I. Preprocessing + Tf-idf scores + Similarity matrix

1. The first preprocessing step is converting the html document to text using html2text(). This removes the html tags. Next the text is lowercased and line breaks, punctuations , numbers, underscores as well as all the stopwords are removed.

2. After the preprocessing step is completed the Tf Idf values are obtained. The collection of html documents is transformed into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) properties using this vectorizer.
The TF-IDF statistic measures the prominence of a phrase inside a document or group of documents. It is frequently employed in text mining and information retrieval activities.
The document's variable is then passed to the TfidfVectorizer object's fit_transform() method. 3.This process converts the provided documents into a matrix representation to be converted into a similarity matrix. The similarity matrix is calculated using the calculated tf-idf scores.

II. Algorithm : Agglomerative Clustering with single link method

In Agglomerative clustering documents are clustered based on their similarity. In each iteration, the algorithm finds the pair of clusters (max_i and max_j) with the highest similarity (calculated as the minimum similarity between any two points, one from each cluster). The algorithm mergers the clusters and updates the similarity matrix and then sets the similarity scores of the merged clusters to 1.0. The process continues until the highest similarity falls below the specified threshold.

Step1: Each html document is initially placed in its own cluster.

Step2: We iterate over each cluster and compare each pair of clusters to find the pair with the highest similarity. Then Calculate the similarity between two clusters as the minimum similarity between any two points, one from each cluster.

Step3: The clusters are merged if the threshold of 0.4 is not crossed and their similarity scores of the merged clusters are updated.

Step4: Step 2 and 3 is repeated until the threshold of 0.4 is not crossed.

III. Results

Which pair of HTML documents is the most similar? Documents 1 and Documents 30

Which pair of documents is the most dissimilar? Document 140 and Document 387

Which document is the closest to the corpus centroid? Document 30

Number of obtained clusters : 44
