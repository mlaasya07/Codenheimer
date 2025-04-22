# Codenheimer
---

## 📚 Introduction

This document outlines the functionality and implementation of the Recommendation System, a Python-based application designed to provide personalized reward recommendations. Utilizing the `synthetic_user_dataset_10000.xlsx` dataset, the system employs TF-IDF vectorization, cosine similarity, and natural language processing (NLP) to match users with rewards such as zoo tickets, concert passes, or gardening kits. This documentation details system requirements, dataset structure, operational mechanics, usage instructions, performance metrics, and maintenance guidelines.

## 📜 System Overview

The Recommendation System processes a dataset of 10,000 users to generate tailored reward recommendations based on age group, interests, activity score, and past rewards. It integrates collaborative filtering (user similarity) and content-based filtering (interest matching) to ensure relevance. Key functionalities include:

- NLP preprocessing for text standardization and interest mapping
- TF-IDF vectorization and cosine similarity for user matching
- Tiered recommendations (basic: <50, standard: 50-74, premium: ≥75) based on activity score `(Stretch Goal)`
- Evaluation accuracy of 99.4% on a 500-user subset
- Caching and sparse matrices for optimized performance

## 🛠️ System Requirements

To operate the system, the following prerequisites must be met:

- **Python Libraries**:
    - Required: `pandas`, `numpy`, `scikit-learn`, `nltk`, `scipy`
    - Installation: `pip install pandas numpy scikit-learn nltk scipy`
- **NLTK Data**:
    - Execute:
        
        ```python
        import nltk
        nltk.download(['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'])
        
        ```
        
- **Dataset**:
    - File: `synthetic_user_dataset_10000.xlsx` (10,000 records)
    
    [synthetic_user_dataset_10000.xlsx](attachment:144609ef-b0d2-4cec-bfba-3a192bb118c6:synthetic_user_dataset_10000.xlsx)
    
- **Hardware**:
    - Minimum: CPU with 4GB RAM
    - Storage: ~1-2GB for caching similarity matrix

## 🗂️ Dataset Structure

The `synthetic_user_dataset_10000.xlsx` dataset contains 10,000 user records with the following schema:

| Column | Description | Example Value |
| --- | --- | --- |
| `user_id` | Unique user identifier | `usr_01001` |
| `age_group` | User’s age range (18-24, 25-34, 35-44, 45-54, 55+) | `25-34` |
| `interests` | Comma-separated user interests | `animals,gaming,DIY` |
| `activity_score` | Numeric engagement score (0-100) | `55` |
| `past_rewards` | Comma-separated previously received rewards | `tool_set,concert_pass` |

**Observations**:

- **Interests**: ~20 mapped categories (e.g., `movies`, `gaming`, `cooking`, `pets`, `crafts`), derived from 100+ specific terms via `interest_mappings`.
- **Age Groups**: Five categories, with `25-34` and `35-44` prevalent.
- **Activity Score**: Ranges from 10 to 100, used for tiering (basic: <50, standard: 50-74, premium: ≥75).
- **Past Rewards**: ~50 unique rewards (e.g., `tool_set`, `zoo_ticket`, `movie_ticket`), standardized via `term_mappings`.
- **Data Handling**: Missing values imputed (`activity_score` with mean, others with “unknown”).

## 🧑‍💻 Operational Mechanics

The system generates recommendations through the following workflow:

1. **Data Loading**:
    - Imports dataset into a Pandas `DataFrame`.
2. **Preprocessing**:
    - Standardizes multi-word terms (e.g., `tool set` to `tool_set`) using `term_mappings`.
    - Maps specific interests to broad categories (e.g., `dog` to `pets`) via `interest_mappings`.
    - Applies NLP (lowercasing, removing `stopwords`/punctuation/numbers, lemmatizing) using NLTK.
    - Normalizes `activity_score` to 0-1 with `MinMaxScaler`.
    - Combines features (`age_group`, `interests` [3x boosted], `activity_score_norm`, `past_rewards`).
3. **Vectorization**:
    - Converts features to TF-IDF vectors (`TfidfVectorizer`, n-grams: 1-3, max features: 3000, `max_df=0.8`).
    - Computes sparse cosine similarity matrix, cached as `similarity_matrix.pkl`.
4. **Reward Mapping**:
    - Maps ~50 rewards to interests (e.g., `movie_ticket` to `movies`) using `reward_interest_map`.
    - Builds a user-reward matrix, boosting new rewards (e.g., `pet_training` +0.1).
5. **Recommendation Generation**:
    - **New Users**:
        - Processes input interests, maps to categories, and creates TF-IDF vector.
        - Scores rewards from similar users, ensuring diversity by covering input interests.
        - Defaults to “`standard`” tier.
6. **Evaluation**:
    - Validates recommendations against user interests or past rewards.
    - Achieves 98.2% accuracy on a 500-user subset.

## 🚀 Usage Instructions

### System Execution

1. **Environment Setup**:
    - Install libraries and NLTK data (see **System Requirements**).
    - Place `synthetic_user_dataset_10000.xlsx` in the working directory.
2. **Running the System**:
    - Execute the Jupyter Notebook (`RecommendationSystem (3).ipynb`).
    - The system will:
        - Load and preprocess the dataset.
        - Cache the similarity matrix.
        - Prompt for new user input (three interests from 20+ categories, e.g., `movies`, `music`, `yoga`).
3. **Example Interaction**:
    - **Input**: `yoga,movies,music`
    - **Output**:
        
        ```
        Processed interests: yoga movies music
        Selected rewards: ['DVD', 'movie_ticket', 'yoga_session']
        Your Recommendations:
        Interests: yoga movies music
        Rewards: ['DVD', 'movie_ticket', 'yoga_session']
        Tier: standard
        
        ```
        
4. **Existing Users**:
    - Use `recommend_rewards(user_idx, num_recommendations=3)`, where `user_idx` is 0 to 9999.
    - Example: `recommend_rewards(0)`.
5. **New Users**:
    - Use `recommend_for_new_user(user_input, num_recommendations=3)`, with comma-separated interests.
    - Example: `recommend_for_new_user("yoga,movies,music")`.

## 🎁 Available Rewards

The system supports ~50 unique rewards, mapped to 20+ interest categories:

| Reward | Associated Interests |
| --- | --- |
| `tool_set` | `crafts` |
| `concert_pass` | `music` |
| `zoo_ticket` | `pets` |
| `movie_ticket` | `movies` |
| `e-book` | `books` |
| `free_meal` | `cooking` |
| `travel_voucher` | `travel` |
| `sports_ticket` | `sports` |
| `yoga_session` | `yoga` |
| `board_game` | `board_games` |
| `etc…` |  |

**Note**: Rewards are standardized (e.g., `movie ticket` to `movie_ticket`) and extensible via `reward_interest_map`.

## 📊 Performance Metrics

- **Preprocessing**: ~4.58 seconds for 10,000 users
- **TF-IDF Vectorization**: ~0.26 seconds
- **Similarity Matrix**: ~7.61 seconds (initial) or ~seconds (cached)
- **Recommendation**: ~seconds for new user (varies with input)
- **Accuracy**: 98.2% on 500-user subset

**Optimization Strategies**:

- Cache `similarity_matrix.pkl` to skip recomputation.
- Use sparse matrices (`scipy.sparse`) for memory efficiency.
- Limit `max_features` and `max_df` in `TfidfVectorizer`.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| NLTK data not found | Run `nltk.download()` commands in **System Requirements**. |
| Dataset file not found | Ensure `synthetic_user_dataset_10000.xlsx` is in `/content/`. |
| Invalid interest input | Use interests from mapped categories (e.g., `movies`, `yoga`). |
| Memory errors | Reduce `max_features` or use a higher-RAM system. |


---

## 🔍 References

- Blog: https://appinventiv.com/blog/recommendation-system-machine-learning/
- Youtube: [**🚀 Data Cleaning/Data Preprocessing Before Building a Model - A Comprehensive Guide**](https://youtu.be/GP-2634exqA?si=3qop1iAfk2z7IiTw)
- Youtube: [Movie Recommendation System using Machine Learning with Python](https://youtu.be/7rEagFH9tQg?si=N3Qmpd-2rtzfyv20)
- ChatGPT, Deepseek and Grok AI (for error checking and optimizing)
  
## 📝 Conclusion

- Developed by **Team Codenheimer** for the **‘Rayv Code-o-Tron 3000 Virtual Hackathon’**.

**Members**:

- **Anirudh**
- **Laasya**
- **Lahari**
- **Tasneem**

the Recommendation System excels in delivering accurate, diverse reward recommendations. Its optimized preprocessing, vectorization, and recommendation logic ensure high performance and scalability, making it a robust solution for personalized reward matching.

---
