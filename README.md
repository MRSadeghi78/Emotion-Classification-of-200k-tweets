# Emotion Classification of 600k Tweets
## Project Overview
This project involves processing, analyzing, and classifying emotions from a large dataset of Twitter posts labeled with six distinct emotion categories: anger, surprise, fear, joy, sadness, and love. The dataset is available on Kaggle, containing over 600K records. The goal is to explore various natural language processing (NLP) and machine learning techniques to analyze and classify emotional states.

## Dataset
The dataset consists of Twitter posts labeled with one of six emotions. Each post provides information that can help in understanding the characteristics of various emotional expressions.

## Project Structure and Key Tasks
### 1. Data Processing and Analysis
- **Construct Emotion DataFrames:** A separate DataFrame was created for each emotion category in the dataset.
- **Vocabulary Analysis:** For each category, we identified the vocabulary set, calculated the proportion of unique tokens, average tokens per post (including standard deviation), average pronouns per post, and average out-of-vocabulary tokens (no entry in WordNet).
- **Summary Table:** Key characteristics for each emotion were summarized in a table.
### 2. Visualizations and Frequency Analysis
- **WordCloud Visualizations:** Word clouds were generated to illustrate the frequent words in each emotion category.
- **Top 20 Frequent Tokens:** The top 20 frequent tokens (excluding stopwords) were identified for each category.
- **Vocabulary Correlation:** A vocabulary was constructed using the top 20 tokens, and a Pearson correlation matrix was generated for all pairs of emotions based on the frequency of these tokens.
### 3. Emotion Discrimination Using Empath Categories
- **Empath Embeddings:** Emotion-specific vectors were generated using the Empath library.
- **Correlation Analysis:** Pearson correlation was computed for Empath-generated embeddings across all emotion pairs. An 8x8 matrix representation was created to illustrate the correlation between emotion states.
### 4. Dimensionality Reduction with Word2Vec
- **Word2Vec Embeddings:** Word2Vec embeddings were generated for each emotion state, and PCA was applied to reduce these embeddings to 2D space.
- **Visualization:** A 2D plot was created to visualize the emotion states as points. Analysis was done on the proximity between emotions and its alignment with psychological models.
### 5. Vocabulary Overlap Analysis with Word2Vec
- **Proximity Evaluation:** For each emotion, the top 30 closest words were generated using Word2Vec embeddings.
- **Jaccard Distance:** The Jaccard distance was used to quantify proximity based on overlapping vocabulary between emotions. The results were compared to the PCA proximity analysis to determine consistency.
### 6. Embedding Comparison with FastText, GloVe, and BERT (RoBERTa)
- **Alternative Embeddings:** The Word2Vec analysis was repeated using FastText, GloVe, and BERT (RoBERTa) embeddings.
- **Comparative Analysis:** The results were compared across embeddings to assess the most effective model for emotion discrimination.
### 7. Fine-Tuning Models on Local Examples
- **Fine-Tuning:** Each of the embedding models (Word2Vec, FastText, GloVe, and RoBERTa) was fine-tuned on the local dataset to increase classification accuracy.
- **Visualization:** After fine-tuning, the 2D projections were recreated to examine any changes in proximity and representation of emotion states.
### 8. Machine Learning for Emotion Prediction
- **Classification with SVM:** An SVM classifier was trained on tf-idf features to predict emotions. Precision, recall, and F1-score were reported.
- **Stopword Removal & Vectorizer Comparison:** Experiments were conducted with and without stopword removal, and tf-idf was compared with count-vectorizer to observe changes in performance.
### 9. DeepMoji Emotion Recognition
- **DeepMoji Implementation:** The state-of-the-art DeepMoji model was used to predict emotions on the test set, and the F1-score was calculated.
- **Model Retraining:** If time permitted, DeepMoji was retrained on the local dataset to further enhance performance.
### 10. Exploration of New Deep Learning Models
- **Kaggle Model Research:** State-of-the-art implementations on Kaggle were explored, and a new deep learning model was proposed to improve prediction accuracy. Results were reported in terms of precision, recall, and F1-score.
