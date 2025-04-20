import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import time
import random
from scipy import sparse

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

FILE_PATH = "synthetic_user_dataset_10000.xlsx"
df = pd.read_excel(FILE_PATH)

term_mappings = {
    'tool set': 'tool_set', 'tool kit': 'tool_kit', 'game credit': 'game_credit', 'game DLC': 'game_DLC',
    'concert pass': 'concert_pass', 'zoo ticket': 'zoo_ticket', 'free meal': 'free_meal', 'flight deal': 'flight_deal',
    'sports equipment': 'sports_equipment', 'cooking class': 'cooking_class', 'dance lesson': 'dance_lesson',
    'music festival': 'music_festival', 'movie premiere': 'movie_premiere', 'pet accessory': 'pet_accessory',
    'travel guide': 'travel_guide', 'news subscription': 'news_subscription', 'gaming console': 'gaming_console',
    'diy workshop': 'diy_workshop', 'vip pass': 'vip_pass', 'streaming voucher': 'streaming_voucher',
    'book club': 'book_club', 'food tour': 'food_tour', 'pet training': 'pet_training', 'craft kit': 'craft_kit',
    'chef workshop': 'chef_workshop', 'movie ticket': 'movie_ticket', 'concert ticket': 'concert_ticket',
    'sports ticket': 'sports_ticket', 'food coupon': 'food_coupon', 'news coupon': 'news_coupon',
    'travel voucher': 'travel_voucher', 'music workshop': 'music_workshop',
    'art class': 'art_class', 'photo workshop': 'photo_workshop', 'writing course': 'writing_course',
    'gardening kit': 'gardening_kit', 'tech gadget': 'tech_gadget', 'fashion voucher': 'fashion_voucher',
    'museum pass': 'museum_pass', 'yoga session': 'yoga_session', 'board game': 'board_game'
}

def standardize_terms(text):
    if not isinstance(text, str):
        return text
    for term, replacement in term_mappings.items():
        text = text.replace(term, replacement)
    return text

df['interests'] = df['interests'].apply(lambda x: ' '.join(str(x).split(',')) if pd.notnull(x) else '')
df['past_rewards'] = df['past_rewards'].apply(lambda x: ' '.join(str(x).split(',')) if pd.notnull(x) else '')
df['interests'] = df['interests'].apply(standardize_terms)
df['past_rewards'] = df['past_rewards'].apply(standardize_terms)
df['interests_original'] = df['interests']
df['past_rewards_original'] = df['past_rewards']

interest_mappings = {
    'movies': 'movies', 'movie': 'movies', 'gaming': 'gaming', 'game': 'gaming', 'music': 'music',
    'books': 'books', 'book': 'books', 'reading': 'books',
    'diy': 'crafts', 'crafting': 'crafts', 'woodworking': 'crafts', 'knitting': 'crafts',
    'animals': 'pets', 'animal': 'pets', 'dog': 'pets', 'cat': 'pets', 'pet': 'pets',
    'news': 'current_events', 'current events': 'current_events', 'politics': 'current_events',
    'dance': 'dancing', 'dancing': 'dancing', 'ballet': 'dancing', 'salsa': 'dancing',
    'travel': 'travel', 'travelling': 'travel', 'vacation': 'travel',
    'food': 'cooking', 'cooking': 'cooking', 'baking': 'cooking', 'eating': 'cooking',
    'sports': 'sports', 'sport': 'sports', 'fitness': 'sports', 'exercise': 'sports',
    'jogging': 'sports', 'jog': 'sports', 'running': 'sports', 'hiking': 'sports',
    'swimming': 'sports', 'swim': 'sports', 'working out': 'sports', 'work out': 'sports',
    'workout': 'sports', 'weightlifting': 'sports', 'gym': 'sports', 'sprinting': 'sports', 'sprint': 'sports',
    'art': 'art', 'painting': 'art', 'drawing': 'art', 'sculpting': 'art',
    'photography': 'photography', 'photo': 'photography', 'camera': 'photography',
    'writing': 'writing', 'creative writing': 'writing', 'journaling': 'writing',
    'gardening': 'gardening', 'plants': 'gardening', 'landscaping': 'gardening',
    'tech': 'tech', 'technology': 'tech', 'coding': 'tech', 'gadgets': 'tech',
    'fashion': 'fashion', 'clothing': 'fashion', 'style': 'fashion', 'design': 'fashion',
    'history': 'history', 'historical': 'history', 'archaeology': 'history',
    'yoga': 'yoga', 'meditation': 'yoga', 'wellness': 'yoga',
    'board games': 'board_games', 'tabletop': 'board_games', 'strategy games': 'board_games'
}

def map_interests(text):
    tokens = text.lower().split()
    mapped = [interest_mappings.get(token, token) for token in tokens]
    return ' '.join(mapped)

df['interests_mapped'] = df['interests_original'].apply(map_interests)

features_selected = ['age_group', 'interests', 'activity_score', 'past_rewards']
for feature in features_selected:
    if feature == 'activity_score':
        df[feature] = df[feature].fillna(df[feature].mean())
    else:
        df[feature] = df[feature].fillna('unknown')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, preserve_interests=False):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'\b(?:and|or|&|\|)\b', ' ', text)
    text = re.sub(r'[,]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    if preserve_interests:
        tokens = [interest_mappings.get(word, word) for word in tokens]
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

def preprocess_column(col):
    col = col.str.lower() \
             .replace(r'\b(?:and|or|&|\|)\b', ' ', regex=True) \
             .replace(r'[,]', ' ', regex=True) \
             .replace(r'\d+', ' ', regex=True) \
             .replace(f"[{re.escape(string.punctuation)}]", " ", regex=True)
    return col.apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(x) if w not in stop_words]) if isinstance(x, str) else '')

df['age_group'] = preprocess_column(df['age_group'])
df['interests'] = preprocess_column(df['interests'])
df['past_rewards'] = preprocess_column(df['past_rewards'])
df['interests'] = df['interests'].apply(lambda x: ' '.join([interest_mappings.get(w, w) for w in x.split()]) if x else '')

scaler = MinMaxScaler()
df['activity_score_norm'] = scaler.fit_transform(df[['activity_score']])

combined_features = df['age_group'] + ' ' + (df['interests'] + ' ') * 5 + df['activity_score_norm'].astype(str) + ' ' + (df['past_rewards'] + ' ') * 3

CACHE_PATH = "similarity_matrix_fixed.pkl"

vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_df=0.85, max_features=5000)
feature_vectors = vectorizer.fit_transform(combined_features)

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'rb') as f:
        user_similarity = pickle.load(f).toarray()
else:
    user_similarity = cosine_similarity(feature_vectors)
    sparse_similarity = sparse.csr_matrix(user_similarity)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(sparse_similarity, f)

reward_interest_map = {
    'tool_set': ['crafts'], 'tool_kit': ['crafts'], 'craft_kit': ['crafts'], 'diy_workshop': ['crafts'],
    'concert_pass': ['music'], 'concert_ticket': ['music'], 'music_festival': ['music'], 'music_workshop': ['music'],
    'zoo_ticket': ['pets'], 'pet_accessory': ['pets'], 'pet_grooming': ['pets'], 'pet_training': ['pets'],
    'game_credit': ['gaming'], 'game_DLC': ['gaming'], 'gaming_console': ['gaming'],
    'DVD': ['movies'], 'popcorn': ['movies'], 'movie_premiere': ['movies'], 'movie_ticket': ['movies'],
    'e-book': ['books'], 'audiobook': ['books'], 'book_club': ['books'], 'subscription': ['books'],
    'free_meal': ['cooking'], 'cooking_class': ['cooking'], 'food_tour': ['cooking'], 'recipe_book': ['cooking'],
    'chef_workshop': ['cooking'], 'food_coupon': ['cooking'],
    'news_coupon': ['current_events'], 'news_subscription': ['current_events'], 'discount': ['current_events'],
    'flight_deal': ['travel'], 'travel_guide': ['travel'], 'travel_package': ['travel'], 'travel_voucher': ['travel'],
    'sports_equipment': ['sports'], 'jersey': ['sports'], 'fitness_tracker': ['sports'], 'sports_ticket': ['sports'],
    'dance_lesson': ['dancing'], 'dance_festival': ['dancing'],
    'streaming_voucher': ['movies', 'music'], 'vip_pass': ['music', 'movies'],
    'art_class': ['art'], 'art_supplies': ['art'], 'gallery_pass': ['art'],
    'photo_workshop': ['photography'], 'camera_accessory': ['photography'], 'photo_book': ['photography'],
    'writing_course': ['writing'], 'journal': ['writing'], 'writing_retreat': ['writing'],
    'gardening_kit': ['gardening'], 'plant_subscription': ['gardening'], 'garden_tour': ['gardening'],
    'tech_gadget': ['tech'], 'coding_course': ['tech'], 'tech_magazine': ['tech'],
    'fashion_voucher': ['fashion'], 'style_workshop': ['fashion'], 'clothing_subscription': ['fashion'],
    'museum_pass': ['history'], 'history_book': ['history'], 'historical_tour': ['history'],
    'yoga_session': ['yoga'], 'meditation_app': ['yoga'], 'wellness_retreat': ['yoga'],
    'board_game': ['board_games'], 'game_night_pass': ['board_games'], 'strategy_game': ['board_games']
}

all_rewards = set()
for rewards in df['past_rewards_original']:
    if pd.notnull(rewards):
        reward_list = [r.strip() for r in re.split(r'[,\s]+', rewards) if r.strip()]
        cleaned_rewards = []
        i = 0
        while i < len(reward_list):
            if i + 1 < len(reward_list) and f"{reward_list[i]} {reward_list[i+1]}" in term_mappings.values():
                cleaned_rewards.append(f"{reward_list[i]}_{reward_list[i+1]}")
                i += 2
            else:
                cleaned_rewards.append(reward_list[i])
                i += 1
        all_rewards.update(cleaned_rewards)
all_rewards.update(reward_interest_map.keys())
all_rewards = sorted([r for r in all_rewards if r in reward_interest_map])

user_reward_matrix = np.zeros((len(df), len(all_rewards)))
reward_to_idx = {reward: idx for idx, reward in enumerate(all_rewards)}
for idx, rewards in enumerate(df['past_rewards_original']):
    if pd.notnull(rewards):
        reward_list = [r.strip() for r in re.split(r'[,\s]+', rewards) if r.strip()]
        cleaned_rewards = []
        i = 0
        while i < len(reward_list):
            if i + 1 < len(reward_list) and f"{reward_list[i]} {reward_list[i+1]}" in term_mappings.values():
                cleaned_rewards.append(f"{reward_list[i]}_{reward_list[i+1]}")
                i += 2
            else:
                cleaned_rewards.append(reward_list[i])
                i += 1
        for reward in cleaned_rewards:
            if reward in reward_to_idx:
                user_reward_matrix[idx, reward_to_idx[reward]] = 1

new_rewards = ['pet_training', 'craft_kit', 'chef_workshop', 'movie_ticket', 'concert_ticket',
               'sports_ticket', 'food_coupon', 'news_coupon', 'travel_voucher', 'music_workshop',
               'art_class', 'photo_workshop', 'writing_course', 'gardening_kit', 'tech_gadget',
               'fashion_voucher', 'museum_pass', 'yoga_session', 'board_game']
for reward in new_rewards:
    if reward in reward_to_idx:
        user_reward_matrix[:, reward_to_idx[reward]] += 0.15

def recommend_rewards(user_idx, num_recommendations=3):
    sim_scores = user_similarity[user_idx]
    threshold = 0.3
    filtered_users = [i for i, score in enumerate(sim_scores) if score > threshold and i != user_idx]
    if not filtered_users:
        filtered_users = np.argsort(sim_scores)[::-1][1:11]
    else:
        filtered_users = sorted(filtered_users, key=lambda x: sim_scores[x], reverse=True)[:10]

    reward_scores = np.zeros(len(all_rewards))
    for sim_user in filtered_users:
        reward_scores += user_reward_matrix[sim_user] * sim_scores[sim_user]

    user_interests = df.iloc[user_idx]['interests_mapped'].split()
    for reward, idx in reward_to_idx.items():
        reward_interests = reward_interest_map.get(reward, [])
        for interest in reward_interests:
            if interest in user_interests:
                reward_scores[idx] *= 300.0

    if reward_scores.max() < 0.5:
        for reward, idx in reward_to_idx.items():
            reward_interests = reward_interest_map.get(reward, [])
            for interest in reward_interests:
                if interest in user_interests:
                    reward_scores[idx] += 300.0

    top_indices = np.argsort(reward_scores)[::-1][:num_recommendations]
    recommended_rewards = [all_rewards[idx] for idx in top_indices]

    activity = df.iloc[user_idx]['activity_score']
    tier = 'premium' if activity >= 75 else 'standard' if activity >= 50 else 'basic'
    return recommended_rewards, tier

def simulate_accuracy(user_idx, recommended_rewards):
    user_interests = set(df.iloc[user_idx]['interests_mapped'].split())
    user_past_rewards = set([r.strip() for r in re.split(r'[,\s]+', df.iloc[user_idx]['past_rewards_original']) if r.strip()])
    for reward in recommended_rewards:
        reward_interests = set(reward_interest_map.get(reward, []))
        if reward_interests & user_interests or reward in user_past_rewards:
            return True
    return False

Accuracy_scores = []
eval_subset = df.sample(n=500, random_state=42).index
for user_idx in eval_subset:
    recommendations, tier = recommend_rewards(user_idx)
    satisfied = simulate_accuracy(user_idx, recommendations)
    Accuracy_scores.append(satisfied)
accuracy_rate = np.mean(Accuracy_scores) * 100
print(f"Model Accuracy: {accuracy_rate:.2f}%")

recommendations_list = []
for user_idx in eval_subset:
    recommendations, tier = recommend_rewards(user_idx)
    satisfied = simulate_accuracy(user_idx, recommendations)

def recommend_for_new_user(user_input, num_recommendations=3):
    if not user_input.strip():
        print("No interests entered, can’t recommend anything!")
        return [], 'standard', 'no interests provided'

    user_input_processed = preprocess_text(user_input, preserve_interests=True)
    user_interests = [interest_mappings.get(word, word) for word in user_input_processed.split()][:3]
    user_input_processed = ' '.join(user_interests)
    print(f"Processed interests: {user_input_processed}")

    new_user_features = f"unknown {user_input_processed} {user_input_processed} {user_input_processed} {df['activity_score_norm'].mean()}"
    new_user_vector = vectorizer.transform([new_user_features])
    sim_scores = cosine_similarity(new_user_vector, feature_vectors)[0]
    similar_users = np.argsort(sim_scores)[::-1][:10]

    reward_scores = np.zeros(len(all_rewards))
    for sim_user in similar_users:
        reward_scores += user_reward_matrix[sim_user] * sim_scores[sim_user]

    interest_counts = {interest: user_interests.count(interest) for interest in user_interests}
    for reward, idx in reward_to_idx.items():
        reward_interests = reward_interest_map.get(reward, [])
        for interest in reward_interests:
            if interest in user_interests:
                reward_scores[idx] *= 200.0 * max(1, interest_counts.get(interest, 1))

    if reward_scores.max() < 0.5:
        for reward, idx in reward_to_idx.items():
            reward_interests = reward_interest_map.get(reward, [])
            for interest in reward_interests:
                if interest in user_interests:
                    reward_scores[idx] += 200.0

    top_indices = np.argsort(reward_scores)[::-1][:num_recommendations * 2]
    selected_rewards = []
    covered_interests = set()
    candidate_rewards = [all_rewards[idx] for idx in top_indices]
    for reward in candidate_rewards:
        reward_interests = set(reward_interest_map.get(reward, []))
        if reward_interests & set(user_interests) and reward not in selected_rewards:
            selected_rewards.append(reward)
            covered_interests.update(reward_interests & set(user_interests))
            if len(selected_rewards) >= num_recommendations or covered_interests == set(user_interests):
                break

    while len(selected_rewards) < num_recommendations:
        for reward in candidate_rewards:
            reward_interests = set(reward_interest_map.get(reward, []))
            if reward not in selected_rewards and reward_interests & set(user_interests):
                selected_rewards.append(reward)
                break
        else:
            fallback_rewards = {
                'pets': ['zoo_ticket', 'pet_accessory', 'pet_grooming', 'pet_training'],
                'crafts': ['tool_set', 'tool_kit', 'craft_kit', 'diy_workshop'],
                'cooking': ['free_meal', 'cooking_class', 'food_tour', 'recipe_book', 'chef_workshop', 'food_coupon'],
                'movies': ['movie_ticket', 'DVD', 'popcorn', 'movie_premiere', 'streaming_voucher'],
                'music': ['concert_ticket', 'concert_pass', 'music_festival', 'music_workshop', 'streaming_voucher'],
                'books': ['e-book', 'audiobook', 'book_club', 'subscription'],
                'gaming': ['game_credit', 'game_DLC', 'gaming_console'],
                'dancing': ['dance_lesson', 'dance_festival'],
                'travel': ['flight_deal', 'travel_guide', 'travel_package', 'travel_voucher'],
                'current_events': ['news_subscription', 'discount', 'news_coupon'],
                'art': ['art_class', 'art_supplies', 'gallery_pass'],
                'photography': ['photo_workshop', 'camera_accessory', 'photo_book'],
                'writing': ['writing_course', 'journal', 'writing_retreat'],
                'gardening': ['gardening_kit', 'plant_subscription', 'garden_tour'],
                'tech': ['tech_gadget', 'coding_course', 'tech_magazine'],
                'fashion': ['fashion_voucher', 'style_workshop', 'clothing_subscription'],
                'history': ['museum_pass', 'history_book', 'historical_tour'],
                'yoga': ['yoga_session', 'meditation_app', 'wellness_retreat'],
                'board_games': ['board_game', 'game_night_pass', 'strategy_game']
            }
            for interest in user_interests:
                for reward in fallback_rewards.get(interest, []):
                    if reward in all_rewards and reward not in selected_rewards:
                        selected_rewards.append(reward)
                        break
                if len(selected_rewards) >= num_recommendations:
                    break
            else:
                selected_rewards.append(random.choice(all_rewards))

    print(f"Selected rewards: {selected_rewards}")
    return selected_rewards[:num_recommendations], 'standard', user_input_processed

start_time = time.time()
mapped_interests = sorted(set(interest_mappings.values()))
print("\nAvailable Interests:")
for i, interest in enumerate(mapped_interests, 1):
    print(f"{i}. {interest}")
user_input = input("\nPick three interests from the list above: ")
recommendations, tier, processed_input = recommend_for_new_user(user_input)
print(f"Recommendations generated in {time.time() - start_time:.2f} seconds")

if recommendations:
    print(f"\nYour Recommendations:")
    print(f"Interests: {processed_input}")
    print(f"Rewards: {recommendations}")
    print(f"Tier: {tier}")
else:
    print("Oops, couldn’t generate recommendations.")

accuracy_rate = np.mean([simulate_accuracy(user_idx, recommend_rewards(user_idx)[0]) for user_idx in df.sample(n=500, random_state=42).index]) * 100
print(f"Model Accuracy: {accuracy_rate:.2f}%")
