# Codenheimer
---

## 📚 Introduction

This document outlines the functionality and implementation of the Recommendation System, a Python-based application designed to provide personalized reward recommendations. Utilizing the `synthetic_user_dataset_10000.xlsx` dataset, the system employs TF-IDF vectorization, cosine similarity, and natural language processing (NLP) to match users with rewards such as zoo tickets, concert passes, or gardening kits. A Flask-based web interface enhances user interaction, allowing users to select interests and receive tailored recommendations through a responsive frontend. This documentation details system requirements, dataset structure, operational mechanics, frontend implementation, usage instructions, performance metrics, and maintenance guidelines.

## 📜 System Overview

The Recommendation System processes a dataset of 10,000 users to generate tailored reward recommendations based on age group, interests, activity score, and past rewards. It integrates collaborative filtering (user similarity) and content-based filtering (interest matching) to ensure relevance. A user-friendly web interface, built with HTML, CSS, JavaScript, and Bootstrap, allows users to select up to three interests and view recommendations dynamically. Key functionalities include:

- NLP preprocessing for text standardization and interest mapping
- TF-IDF vectorization and cosine similarity for user matching
- Tiered recommendations (basic: <50, standard: 50-74, premium: ≥75) based on activity score
- Evaluation accuracy of 99.4% on a 500-user subset
- Caching and sparse matrices for optimized backend performance
- Responsive frontend with dynamic interest selection and error handling

## 🛠️ System Requirements

To operate the system, the following prerequisites must be met:

- **Python Libraries**:
  - Required: `pandas`, `numpy`, `scikit-learn`, `nltk`, `scipy`, `flask`
  - Installation: `pip install pandas numpy scikit-learn nltk scipy flask`
- **NLTK Data**:
  - Execute:
    ```python
    import nltk
    nltk.download(['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'])
    ```
- **Frontend Dependencies**:
  - Bootstrap 5.3.0 (loaded via CDN)
  - Google Fonts (Lora) (loaded via CDN)
- **Dataset**:
  - File: `synthetic_user_dataset_10000.xlsx` (10,000 records)
- **Hardware**:
  - Minimum: CPU with 4GB RAM
  - Storage: ~1-2GB for caching similarity matrix
- **Software**:
  - Web browser (Chrome, Firefox, or equivalent) for frontend access
  - Python 3.8+ for Flask backend

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
   - Combines features (`age_group`, `interests` [5x boosted], `activity_score_norm`, `past_rewards` [3x boosted]).
3. **Vectorization**:
   - Converts features to TF-IDF vectors (`TfidfVectorizer`, n-grams: 1-4, max features: 5000, `max_df=0.85`).
   - Computes sparse cosine similarity matrix, cached as `similarity_matrix_fixed.pkl`.
4. **Reward Mapping**:
   - Maps ~50 rewards to interests (e.g., `movie_ticket` to `movies`) using `reward_interest_map`.
   - Builds a user-reward matrix, boosting new rewards (e.g., `pet_training` +0.15).
5. **Recommendation Generation**:
   - **New Users**:
     - Processes input interests, maps to categories, and creates TF-IDF vector.
     - Scores rewards from similar users, ensuring diversity by covering input interests.
     - Defaults to “`standard`” tier.
6. **Frontend Interaction**:
   - Users select up to three interests via clickable buttons.
   - Submits selections to Flask backend via POST request.
   - Displays recommendations, processed interests, and tier in a responsive card.
7. **Evaluation**:
   - Validates recommendations against user interests or past rewards.
   - Achieves 99.4% accuracy on a 500-user subset.

## 🌐 Frontend Implementation

The frontend is a Flask-integrated web interface built with:

- **HTML**: `index.html` provides the structure, including a form for interest selection, a results section, and error alerts.
- **CSS**: `styles.css` customizes the UI with a gradient background, Lora font, and responsive card design. Interest buttons are styled with hover and selected states, and a scrollable container supports dynamic interest lists.
- **JavaScript**: `script.js` handles dynamic button creation, click events for interest selection (max 3), form submission, and error handling. It fetches interests from `/interests` and submits selections to `/recommend` via AJAX.
- **Bootstrap 5.3.0**: Ensures responsive layout and consistent styling.
- **Flask**: Serves the frontend (`index.html`) and handles API endpoints (`/interests`, `/recommend`).

**Key Features**:
- **Dynamic Interest Buttons**: Populated from the backend, allowing users to toggle up to three interests.
- **Error Handling**: Displays alerts for invalid selections (e.g., none or more than three interests).
- **Responsive Design**: Adapts to various screen sizes with a centered card layout.
- **Visual Feedback**: Buttons change color when selected, and results are displayed in a clean, formatted section.

## 🚀 Usage Instructions

### System Execution

1. **Environment Setup**:
   - Install Python libraries and NLTK data (see **System Requirements**).
   - Place `synthetic_user_dataset_10000.xlsx` in the working directory.
   - Ensure project structure:
     ```
     .
     ├── app.py
     ├── static/
     │   ├── css/styles.css
     │   ├── js/script.js
     ├── templates/
     │   ├── index.html
     ├── synthetic_user_dataset_10000.xlsx
     ├── similarity_matrix_fixed.pkl (generated)
     ```
2. **Running the Web Application**:
   - Run the Flask app:
     ```bash
     python app.py
     ```
   - Open a browser and navigate to `http://localhost:5000`.
3. **Using the Web Interface**:
   - **Select Interests**: Click up to three interest buttons (e.g., `movies`, `music`, `yoga`).
   - **Submit**: Click “Get Recommendations” to view results.
   - **View Results**: See processed interests, recommended rewards, and tier (e.g., `standard`).
   - **Example**:
     - **Selection**: `yoga`, `movies`, `music`
     - **Output**:
       ```
       Your Recommendations:
       Interests: yoga, movies, music
       Rewards: movie_ticket, yoga_session, concert_ticket
       Tier: standard
       ```
   - **Error Handling**: Alerts appear if no interests or more than three are selected.
4. **Running the Notebook (Alternative)**:
   - Execute `Codenheimer(2).ipynb` in Jupyter.
   - Follow prompts to input three interests (e.g., `yoga,movies,music`).
   - Example output:
     ```
     Processed interests: yoga movies music
     Selected rewards: ['movie_ticket', 'yoga_session', 'concert_ticket']
     Your Recommendations:
     Interests: yoga movies music
     Rewards: ['movie_ticket', 'yoga_session', 'concert_ticket']
     Tier: standard
     ```
5. **Existing Users (Notebook)**:
   - Use `recommend_rewards(user_idx, num_recommendations=3)`, where `user_idx` is 0 to 9999.
   - Example: `recommend_rewards(0)`.
6. **New Users (Notebook)**:
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
- **Frontend Load**: ~1-2 seconds for interest population and rendering
- **Accuracy**: 99.4% on 500-user subset

**Optimization Strategies**:
- Cache `similarity_matrix_fixed.pkl` to skip recomputation.
- Use sparse matrices (`scipy.sparse`) for memory efficiency.
- Limit `max_features` and `max_df` in `TfidfVectorizer`.
- Leverage CDN for Bootstrap and Google Fonts to reduce load time.

## 🔧 Troubleshooting

| Issue | Resolution |
| --- | --- |
| NLTK data not found | Run `nltk.download()` commands in **System Requirements**. |
| Dataset file not found | Ensure `synthetic_user_dataset_10000.xlsx` is in the working directory. |
| Invalid interest input | Select interests from displayed buttons or use mapped categories in notebook (e.g., `movies`, `yoga`). |
| Memory errors | Reduce `max_features` or use a higher-RAM system. |
| Flask server not running | Run `python app.py` and check `http://localhost:5000`. |
| Frontend not loading | Verify `static/` and `templates/` folders are correctly structured. |
| Interest buttons not appearing | Ensure `/interests` endpoint is accessible and dataset is loaded. |

## 🔍 References

- Blog: https://appinventiv.com/blog/recommendation-system-machine-learning/
- YouTube: [**🚀 Data Cleaning/Data Preprocessing Before Building a Model - A Comprehensive Guide**](https://youtu.be/GP-2634exqA?si=3qop1iAfk2z7IiTw)
- YouTube: [Movie Recommendation System using Machine Learning with Python](https://youtu.be/7rEagFH9tQg?si=N3Qmpd-2rtzfyv20)
- ChatGPT, Deepseek, and Grok AI (for error checking, optimizing and help with code logic and partially, also implementation)

## 📝 Conclusion

Developed by **Team Codenheimer** for the **‘Rayv Code-o-Tron 3000 Virtual Hackathon’**.

**Members**:
- **Anirudh**
- **Laasya**
- **Lahari**
- **Tasneem**

The Recommendation System excels in delivering accurate, diverse reward recommendations through a robust backend and an intuitive Flask-based web interface. Its optimized preprocessing, vectorization, and responsive frontend ensure high performance, scalability, and user satisfaction, making it an effective solution for personalized reward matching.

---
