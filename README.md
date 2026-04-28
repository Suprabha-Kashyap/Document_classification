# BBC News Document Classification

This project implements a classical Machine Learning pipeline to categorize BBC news articles into five distinct categories: **Business, Entertainment, Politics, Sport, and Tech**. It focuses on Bag-of-Words techniques and comparative analysis of ensemble vs. probabilistic models.

## 📊 Dataset
The project uses the **bbc_text_cls.csv** dataset, which consists of:
- **Text:** Full-length news articles from the BBC.
- **Labels:** The category assigned to each article.

## 🛠 Project Workflow

### 1. Data Exploration
- Loading and inspecting the distribution of news categories using histograms to check for class balance.

### 2. Advanced Preprocessing
- **Lemmatization:** Implementing a custom tokenizer using NLTK's `WordNetLemmatizer` combined with **POS (Part-of-Speech) tagging**. This ensures that words are reduced to their correct base form (e.g., "running" becomes "run" as a verb, but remains "running" if used as a noun in context).

### 3. Vectorization
- **CountVectorizer:** Converting raw text into a matrix of token counts (Bag-of-Words representation).

### 4. Machine Learning Models
The project compares two powerful classifiers:
- **Multinomial Naive Bayes:** A probabilistic model highly effective for text data with discrete features (achieved ~97% test accuracy).
- **Random Forest Classifier:** An ensemble learning method using multiple decision trees (achieved ~95% test accuracy).

### 5. Performance Evaluation
Comprehensive evaluation using:
- **Accuracy Score:** Overall model correctness.
- **Confusion Matrix:** Identifying specific misclassifications between categories (e.g., Tech vs. Business).
- **Classification Report:** Detailed Precision, Recall, and F1-Score per category.

## ⚙️ Setup & Requirements

1. **Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib
   ```

2. **NLTK Resources:**
   ```python
   import nltk
   nltk.download(['wordnet', 'punkt', 'averaged_perceptron_tagger', 'omw-1.4'])
   ```
