import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
from functools import wraps
import pymongo
from bson import ObjectId
import os

# -----------------------------
# Setup Flask
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')   # Change this to a secure secret key
CORS(app)

# -----------------------------
# MongoDB Setup
# -----------------------------
# Replace this with your MongoDB connection string
MONGO_URI = os.environ.get('MONGO_URI')  # Gets URI from environment variable# Change this to your MongoDB URI
try:
    client = pymongo.MongoClient(MONGO_URI)
    db = client.anime_recommender
    users_collection = db.users
    lists_collection = db.user_lists
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None

# -----------------------------
# Load dataset and models
# -----------------------------
df = pd.read_csv("anime_cleaned_data.csv").reset_index(drop=True)
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
sparse_similarities = joblib.load("sparse_similarities.pkl")

print(f"✅ Dataset loaded with {len(df)} anime")

# -----------------------------
# Auth Decorators
# -----------------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = users_collection.find_one({'_id': ObjectId(data['user_id'])})
            if not current_user:
                return jsonify({'error': 'User not found'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token is invalid'}), 401
        except Exception as e:
            return jsonify({'error': 'Token validation failed'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated

# -----------------------------
# Helper functions
# -----------------------------
def clean_for_json(data):
    """Clean pandas data for JSON serialization by handling NaN values"""
    if isinstance(data, list):
        return [clean_for_json(item) for item in data]
    elif isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                cleaned[key] = None
            elif isinstance(value, np.integer):
                cleaned[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned[key] = float(value) if not np.isnan(value) else None
            else:
                cleaned[key] = value
        return cleaned
    else:
        return data

def get_recommendations(title, n=10):
    title_lower = title.strip().lower()
    df_lower = df['title_romaji'].str.lower()
    try:
        idx = df_lower[df_lower == title_lower].index[0]
    except IndexError:
        return []

    sim_scores = sparse_similarities.get(idx, [])
    anime_indices = [i[0] for i in sim_scores[:n]]
    recommendations = df.iloc[anime_indices].to_dict(orient='records')
    return clean_for_json(recommendations)

def search_titles(query, top_n=10):
    query_lower = query.strip().lower()
    matches = df[df['title_romaji'].str.lower().str.contains(query_lower, na=False)]
    results = matches.head(top_n).to_dict(orient='records')
    return clean_for_json(results)

def filter_by_genre(genres):
    if not genres:
        popular_anime = df.sort_values('popularity', ascending=False).head(20)
        return clean_for_json(popular_anime.to_dict(orient='records'))
    
    filtered = df.copy()
    for genre in genres:
        # Keep only rows that contain this genre
        filtered = filtered[filtered['genres'].str.contains(genre, case=False, na=False)]
    
    return clean_for_json(filtered.to_dict(orient='records'))


# -----------------------------
# Auth Routes
# -----------------------------
@app.route("/auth/register", methods=['POST'])
def register():
    if db is None:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        # Check if user exists
        if users_collection.find_one({'username': username}):
            return jsonify({'error': 'Username already exists'}), 409
        
        # Create user
        user_id = users_collection.insert_one({
            'username': username,
            'password': generate_password_hash(password),
            'created_at': datetime.utcnow()
        }).inserted_id
        
        # Generate token
        token = jwt.encode({
            'user_id': str(user_id),
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'user': {
                'id': str(user_id),
                'username': username,
                'token': token
            }
        })
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route("/auth/login", methods=['POST'])
def login():
    if db is None:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Find user
        user = users_collection.find_one({'username': username})
        if not user or not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate token
        token = jwt.encode({
            'user_id': str(user['_id']),
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'user': {
                'id': str(user['_id']),
                'username': user['username'],
                'token': token
            }
        })
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# -----------------------------
# User Lists Routes
# -----------------------------
@app.route("/user/lists", methods=['GET'])
@token_required
def get_user_lists(current_user):
    try:
        user_id = str(current_user['_id'])
        
        # Get all user lists
        lists = {
            'favorites': [],
            'watching': [],
            'completed': [],
            'planned': []
        }
        
        for list_type in lists.keys():
            cursor = lists_collection.find({
                'user_id': user_id,
                'list_type': list_type
            }).sort('added_date', -1)
            
            lists[list_type] = []
            for item in cursor:
                lists[list_type].append({
                    'anime_id': item['anime_id'],
                    'anime_title': item['anime_title'],
                    'cover_image': item.get('cover_image'),
                    'score': item.get('score', 0),
                    'added_date': item['added_date'].isoformat() if item.get('added_date') else None
                })
        
        return jsonify(lists)
        
    except Exception as e:
        print(f"Get lists error: {e}")
        return jsonify({'error': 'Failed to fetch lists'}), 500

@app.route("/user/lists/<list_type>", methods=['POST'])
@token_required
def add_to_list(current_user, list_type):
    if list_type not in ['favorites', 'watching', 'completed', 'planned']:
        return jsonify({'error': 'Invalid list type'}), 400
    
    try:
        data = request.get_json()
        user_id = str(current_user['_id'])
        anime_id = data.get('anime_id')
        anime_title = data.get('anime_title')
        cover_image = data.get('cover_image')
        score = data.get('score', 0)
        
        if not anime_id or not anime_title:
            return jsonify({'error': 'Anime ID and title required'}), 400
        
        # Check if already in this list
        existing = lists_collection.find_one({
            'user_id': user_id,
            'anime_id': anime_id,
            'list_type': list_type
        })
        
        if existing:
            return jsonify({'error': f'Anime already in {list_type} list'}), 409
        
        # Remove from other lists (anime can only be in one status list, but can be in favorites + status)
        if list_type != 'favorites':
            lists_collection.delete_many({
                'user_id': user_id,
                'anime_id': anime_id,
                'list_type': {'$in': ['watching', 'completed', 'planned']}
            })
        
        # Add to list
        lists_collection.insert_one({
            'user_id': user_id,
            'anime_id': anime_id,
            'anime_title': anime_title,
            'cover_image': cover_image,
            'list_type': list_type,
            'score': score,
            'added_date': datetime.utcnow()
        })
        
        return jsonify({'message': f'Added to {list_type} list'})
        
    except Exception as e:
        print(f"Add to list error: {e}")
        return jsonify({'error': 'Failed to add to list'}), 500

@app.route("/user/lists/<list_type>/<int:anime_id>", methods=['DELETE'])
@token_required
def remove_from_list(current_user, list_type, anime_id):
    try:
        user_id = str(current_user['_id'])
        
        result = lists_collection.delete_one({
            'user_id': user_id,
            'anime_id': anime_id,
            'list_type': list_type
        })
        
        if result.deleted_count > 0:
            return jsonify({'message': f'Removed from {list_type} list'})
        else:
            return jsonify({'error': 'Item not found in list'}), 404
        
    except Exception as e:
        print(f"Remove from list error: {e}")
        return jsonify({'error': 'Failed to remove from list'}), 500

@app.route("/user/lists/<list_type>/<int:anime_id>/score", methods=['PUT'])
@token_required
def update_score(current_user, list_type, anime_id):
    try:
        data = request.get_json()
        score = data.get('score', 0)
        user_id = str(current_user['_id'])
        
        if not (1 <= score <= 10):
            return jsonify({'error': 'Score must be between 1 and 10'}), 400
        
        result = lists_collection.update_one(
            {
                'user_id': user_id,
                'anime_id': anime_id,
                'list_type': list_type
            },
            {'$set': {'score': score, 'updated_date': datetime.utcnow()}}
        )
        
        if result.modified_count > 0:
            return jsonify({'message': 'Score updated'})
        else:
            return jsonify({'error': 'Item not found in list'}), 404
        
    except Exception as e:
        print(f"Update score error: {e}")
        return jsonify({'error': 'Failed to update score'}), 500

# -----------------------------
# Anime Routes (existing + enhanced)
# -----------------------------
@app.route("/home")
def home():
    try:
        limit = int(request.args.get("limit", 20))
        top_anime = df.sort_values("popularity", ascending=False).head(limit)
        result = top_anime.to_dict(orient="records")
        cleaned_result = clean_for_json(result)
        return jsonify(cleaned_result)
    except Exception as e:
        print(f"Error in home endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/search")
def search():
    try:
        query = request.args.get("query", "")
        if not query:
            return jsonify([])
        results = search_titles(query)
        return jsonify(results)
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/filter")
def filter_anime():
    try:
        genres = request.args.get("genres", "")
        genre_list = [g.strip() for g in genres.split(',')] if genres else []
        results = filter_by_genre(genre_list)
        return jsonify(results)
    except Exception as e:
        print(f"Error in filter endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/anime/<int:anime_id>")
def anime_detail(anime_id):
    try:
        anime = df[df['id'] == anime_id]
        if anime.empty:
            return jsonify({"error": "Anime not found"}), 404
        result = anime.iloc[0].to_dict()
        cleaned_result = clean_for_json(result)
        return jsonify(cleaned_result)
    except Exception as e:
        print(f"Error in detail endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/recommend")
def recommend():
    try:
        title = request.args.get("title", "")
        if not title:
            return jsonify([])
        results = get_recommendations(title)
        return jsonify(results)
    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "dataset_size": len(df),
        "mongodb_connected": db is not None,
        "sample_anime": df.iloc[0]['title_romaji'] if len(df) > 0 else None
    })

# -----------------------------
# User Stats Route
# -----------------------------
@app.route("/user/stats")
@token_required
def user_stats(current_user):
    try:
        user_id = str(current_user['_id'])
        
        stats = {}
        for list_type in ['favorites', 'watching', 'completed', 'planned']:
            count = lists_collection.count_documents({
                'user_id': user_id,
                'list_type': list_type
            })
            stats[list_type] = count
        
        # Calculate average score
        completed_with_scores = list(lists_collection.find({
            'user_id': user_id,
            'list_type': 'completed',
            'score': {'$gt': 0}
        }))
        
        if completed_with_scores:
            avg_score = sum(item['score'] for item in completed_with_scores) / len(completed_with_scores)
            stats['average_score'] = round(avg_score, 1)
        else:
            stats['average_score'] = 0
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"User stats error: {e}")
        return jsonify({'error': 'Failed to fetch user stats'}), 500

# -----------------------------
# Error Handlers
# -----------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    print("=== Enhanced Anime Recommender Backend ===")
    print(f"📊 Dataset loaded with {len(df)} anime")
    print(f"🗄️  MongoDB: {'Connected' if db is not None else 'Not connected'}")

    print("🚀 Server running on http://127.0.0.1:5000")
    print("\n📋 Available Endpoints:")
    print("   Auth:")
    print("     POST /auth/register")
    print("     POST /auth/login")
    print("   Anime:")
    print("     GET  /home?limit=20")
    print("     GET  /search?query=cowboy")
    print("     GET  /filter?genres=Action,Adventure")
    print("     GET  /anime/<id>")
    print("     GET  /recommend?title=Cowboy Bebop")
    print("   User Lists (requires auth):")
    print("     GET  /user/lists")
    print("     POST /user/lists/<list_type>")
    print("     DELETE /user/lists/<list_type>/<anime_id>")
    print("     PUT  /user/lists/<list_type>/<anime_id>/score")
    print("     GET  /user/stats")
    print("   Health:")
    print("     GET  /health")
    print(f"\n💡 Make sure to update MONGO_URI to your MongoDB connection string!")
    print(f"   Current: {MONGO_URI}")
    
    app.run(debug=True)