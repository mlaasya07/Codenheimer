import random
from flask import Flask, request, jsonify, render_template, send_from_directory
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
from scipy import sparse

app = Flask(__name__, static_folder='static')

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/interests', methods=['GET'])
def get_interests():
    unique_interests = sorted(set(interest_mappings.values()))
    return jsonify({'interests': unique_interests})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_input = data.get('interests', '')
    if not user_input:
        return jsonify({'error': 'No interests provided', 'recommendations': [], 'tier': 'standard', 'processed_input': 'no interests provided'}), 400
    
    recommendations, tier, processed_input = recommend_for_new_user(user_input)
    return jsonify({
        'recommendations': recommendations,
        'tier': tier,
        'processed_input': processed_input
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

