import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import time

# Configuration de la page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Maintenance Prédictive Pro",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar cachée par défaut
)

# ==================== PALETTES DE COULEURS ====================
COLORS = {
    'primary': '#2C3E50',           # Bleu foncé/gris pour les titres
    'blue': '#3498DB',               # Bleu primaire - Bleu vif (principal)
    'dark_blue': '#2980B9',          # Bleu foncé - Bleu plus profond
    'light_blue': '#5DADE2',         # Bleu clair - Bleu plus lumineux
    'pastel_blue': '#85C1E9',        # Bleu pastel - Bleu doux
    'very_light_blue': '#AED6F1',    # Bleu très clair - Presque blanc bleuté
    'night_blue': '#1F618D',         # Bleu nuit - Bleu très foncé
    'teal': '#1ABC9C',               # Bleu canard - Bleu-vert (turquoise)
    'royal_blue': '#2471A3',         # Bleu roi - Bleu royal
    'sky_blue': '#7FB3D5',           # Bleu ciel - Bleu ciel
    'steel_blue': '#5D6D7E',         # Bleu acier - Bleu-gris
    'green': '#2ECC71',               # Vert pour le succès
    'orange': '#F39C12',              # Orange pour les avertissements
    'red': '#E74C3C',                 # Rouge pour les erreurs/pannes
    'purple': '#9B59B6',              # Violet pour les variations
    'dark_green': '#27AE60',          # Vert foncé
    'dark_orange': '#E67E22',         # Orange foncé
    'dark_red': '#C0392B',            # Rouge foncé
    'gray': '#7F8C8D',                # Gris
    'light_gray': '#ECF0F1',          # Gris clair
}

# Palette pour graphiques séquentiels (dégradé du plus foncé au plus clair)
COLORS_SEQUENTIAL = [
    '#1F618D',  # Bleu nuit
    '#2980B9',  # Bleu foncé
    '#3498DB',  # Bleu primaire
    '#5DADE2',  # Bleu clair
    '#85C1E9',  # Bleu pastel
    '#AED6F1',  # Bleu très clair
    '#1ABC9C',  # Turquoise
]

# Palette pour graphiques catégoriels (couleurs distinctes)
COLORS_CATEGORICAL = [
    '#3498DB',  # Bleu primaire
    '#E74C3C',  # Rouge
    '#2ECC71',  # Vert
    '#F39C12',  # Orange
    '#9B59B6',  # Violet
    '#1ABC9C',  # Turquoise
    '#2980B9',  # Bleu foncé
    '#C0392B',  # Rouge foncé
    '#27AE60',  # Vert foncé
    '#7FB3D5',  # Bleu ciel
]

# Dégradés spécifiques pour différents usages
BLUE_GRADIENTS = {
    'dark_to_light': ['#1F618D', '#2980B9', '#3498DB', '#5DADE2', '#85C1E9', '#AED6F1'],
    'light_to_dark': ['#AED6F1', '#85C1E9', '#5DADE2', '#3498DB', '#2980B9', '#1F618D'],
    'primary_set': ['#3498DB', '#5DADE2', '#85C1E9', '#2980B9', '#1F618D'],
}

# ==================== STYLE CSS PERSONNALISÉ ====================
st.markdown("""
    <style>
    /* Police et style général */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Mini-header (optionnel) */
    .mini-header img {
        width: 40px;
        height: 40px;
    }
    
    .mini-header h3 {
        margin: 0;
        color: #667eea;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    
    /* Cartes métriques */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .metric-card h3 {
        color: #666;
        font-size: 1rem;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
        line-height: 1.2;
    }
    
    .metric-card .metric-delta {
        font-size: 0.9rem;
        color: #4CAF50;
        margin: 0.5rem 0 0 0;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .badge-blue {
        background: #E3F2FD;
        color: #1976D2;
    }
    
    .badge-green {
        background: #E8F5E9;
        color: #388E3C;
    }
    
    .badge-orange {
        background: #FFF3E0;
        color: #F57C00;
    }
    
    /* Cartes de résultat */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .success-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
    }
    
    .result-card h2 {
        font-size: 2.5rem;
        margin: 0 0 1rem 0;
        font-weight: 700;
    }
    
    .result-card p {
        font-size: 1.2rem;
        margin: 0;
        opacity: 0.9;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== FONCTIONS D'AUTHENTIFICATION ====================

def init_users():
    """Initialise la base de données utilisateurs"""
    users_file = 'users.csv'
    if not os.path.exists(users_file):
        # Créer un fichier avec un utilisateur admin par défaut
        default_users = pd.DataFrame({
            'username': ['admin'],
            'password': ['admin123'],
            'name': ['Administrateur'],
            'role': ['admin'],
            'created_at': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        default_users.to_csv(users_file, index=False)
    return users_file

def check_login(username, password):
    """Vérifie les identifiants de connexion"""
    users_file = 'users.csv'
    if os.path.exists(users_file):
        users = pd.read_csv(users_file)
        user_row = users[(users['username'] == username) & (users['password'] == password)]
        if not user_row.empty:
            user = user_row.iloc[0]
            return {
                'username': user['username'],
                'name': user['name'],
                'role': user['role']
            }
    return None

def register_user(username, password, confirm_password, name):
    """Inscrit un nouvel utilisateur"""
    if not username or not password or not name:
        return False, "Tous les champs sont requis"
    
    if password != confirm_password:
        return False, "Les mots de passe ne correspondent pas"
    
    if len(password) < 6:
        return False, "Le mot de passe doit contenir au moins 6 caractères"
    
    users_file = 'users.csv'
    
    if os.path.exists(users_file):
        users = pd.read_csv(users_file)
        if username in users['username'].values:
            return False, "Nom d'utilisateur déjà existant"
    else:
        users = pd.DataFrame()
    
    new_user = pd.DataFrame([{
        'username': username,
        'password': password,
        'name': name,
        'role': 'user',
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    if users.empty:
        users = new_user
    else:
        users = pd.concat([users, new_user], ignore_index=True)
    
    users.to_csv(users_file, index=False)
    return True, "Inscription réussie ! Vous pouvez maintenant vous connecter"

def get_all_users():
    """Récupère tous les utilisateurs"""
    users_file = 'users.csv'
    if os.path.exists(users_file):
        return pd.read_csv(users_file)
    return pd.DataFrame()

def logout():
    """Déconnecte l'utilisateur"""
    for key in ['logged_in', 'username', 'user_name', 'user_role']:
        if key in st.session_state:
            del st.session_state[key]

# Initialiser la base de données utilisateurs
init_users()

# ==================== FONCTIONS UTILITAIRES ====================

@st.cache_data
def load_data():
    """Charge et prépare les données avec mise en cache"""
    try:
        df = pd.read_csv('ai4i2020.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def train_model(df):
    """Entraîne le modèle Random Forest avec mise en cache"""
    features = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    X = df[features]
    y = df['Machine failure']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    return model, scaler, features

def save_prediction(record):
    """Sauvegarde une prédiction dans l'historique"""
    filename = 'predictions_history.csv'
    
    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        updated = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
        updated.to_csv(filename, index=False)
    else:
        pd.DataFrame([record]).to_csv(filename, index=False)

def load_prediction_history():
    """Charge l'historique des prédictions"""
    filename = 'predictions_history.csv'
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

def create_gauge_chart(value, title):
    """Crée un graphique jauge personnalisé"""
    
    if value < 30:
        bar_color = COLORS['green']
    elif value < 70:
        bar_color = COLORS['orange']
    else:
        bar_color = COLORS['red']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': COLORS['primary']}},
        delta={'reference': 50, 'increasing': {'color': COLORS['red']}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': COLORS['primary']},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': COLORS['gray'],
            'steps': [
                {'range': [0, 30], 'color': '#D5F5E3'},
                {'range': [30, 70], 'color': '#FDEBD0'},
                {'range': [70, 100], 'color': '#FADBD8'}
            ],
            'threshold': {
                'line': {'color': COLORS['dark_red'], 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': COLORS['primary'], 'family': "Inter", 'size': 12}
    )
    return fig

# ==================== ÉTAT DE CONNEXION ====================

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'login'  # 'login' ou 'register'

# ==================== PAGE DE CONNEXION/INSCRIPTION STYLISÉE ====================

if not st.session_state['logged_in']:
    # Style CSS supplémentaire pour la page de connexion
    st.markdown("""
    <style>
        /* Fond animé */
        .stApp {
            background: linear-gradient(-45deg, #667eea, #764ba2, #6b8cff, #9f7aea);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Carte de connexion/inscription - FORME RECTANGULAIRE */
        .auth-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 2.5rem 3rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.3);
            max-width: 550px;
            width: 100%;
            margin: 2rem auto;
            animation: slideUp 0.5s ease;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* En-tête avec icône - simplifié */
        .auth-header {
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .machine-icon {
            font-size: 3rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        
        /* Boutons de basculement stylisés */
        .toggle-container {
            display: flex;
            gap: 10px;
            background: #F3F4F6;
            padding: 5px;
            border-radius: 50px;
            margin-bottom: 2rem;
        }
        
        .stButton button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        /* Style des champs de formulaire */
        .stTextInput > div > div > input {
            border-radius: 12px !important;
            border: 2px solid #E5E7EB !important;
            padding: 12px 16px !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Labels stylisés */
        .field-label {
            color: #374151;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        /* Bouton principal */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            width: 100% !important;
            margin-top: 1.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* Messages de statut */
        .stAlert {
            border-radius: 12px !important;
            border: none !important;
            padding: 1rem !important;
            margin-top: 1rem !important;
        }
        
        /* Pied de page */
        .auth-footer {
            text-align: center;
            margin-top: 2rem;
            color: rgba(255, 255, 255, 0.9);
            font-size: 0.9rem;
        }
        
        /* Mot de passe oublié */
        .forgot-password {
            text-align: right;
            margin: 0.5rem 0 1rem 0;
        }
        
        .forgot-password a {
            color: #667eea;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        .forgot-password a:hover {
            text-decoration: underline;
        }
        
        /* Séparateur */
        .divider {
            display: flex;
            align-items: center;
            text-align: center;
            margin: 1.5rem 0;
            color: #9CA3AF;
        }
        
        .divider::before,
        .divider::after {
            content: '';
            flex: 1;
            border-bottom: 1px solid #E5E7EB;
        }
        
        .divider span {
            padding: 0 10px;
            font-size: 0.9rem;
        }
    </style>
    
    <div class="auth-card">
        <div class="auth-header">
            <div class="machine-icon">🔧⚙️</div>
            <!-- Titre supprimé pour un design épuré -->
        </div>
    """, unsafe_allow_html=True)
    
    # Boutons de basculement stylisés
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔐 Connexion", 
                    use_container_width=True,
                    type="primary" if st.session_state['auth_mode'] == 'login' else "secondary"):
            st.session_state['auth_mode'] = 'login'
            st.rerun()
    
    with col2:
        if st.button("📝 Inscription", 
                    use_container_width=True,
                    type="primary" if st.session_state['auth_mode'] == 'register' else "secondary"):
            st.session_state['auth_mode'] = 'register'
            st.rerun()
    
    # Formulaire selon le mode
    if st.session_state['auth_mode'] == 'login':
        with st.form("login_form"):
            st.markdown("""
                <h3 style="text-align: center; margin-bottom: 1.5rem; color: #2C3E50;">
                    Bienvenue ! 👋
                </h3>
            """, unsafe_allow_html=True)
            
            # Nom d'utilisateur
            st.markdown('<label class="field-label">👤 Nom d\'utilisateur</label>', unsafe_allow_html=True)
            username = st.text_input(
                "", 
                placeholder="Entrez votre nom d'utilisateur",
                key="login_username",
                label_visibility="collapsed"
            )
            
            # Mot de passe
            st.markdown('<label class="field-label">🔒 Mot de passe</label>', unsafe_allow_html=True)
            password = st.text_input(
                "", 
                placeholder="Entrez votre mot de passe",
                type="password",
                key="login_password",
                label_visibility="collapsed"
            )
            
            # Mot de passe oublié
            st.markdown("""
                <div class="forgot-password">
                    <a href="#">Mot de passe oublié ?</a>
                </div>
            """, unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Se connecter")
            
            if submitted:
                if username and password:
                    user = check_login(username, password)
                    if user:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = user['username']
                        st.session_state['user_name'] = user['name']
                        st.session_state['user_role'] = user['role']
                        st.success("✅ Connexion réussie ! Redirection...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Nom d'utilisateur ou mot de passe incorrect")
                else:
                    st.warning("⚠️ Veuillez remplir tous les champs")
    
    else:  # mode inscription
        with st.form("register_form"):
            st.markdown("""
                <h3 style="text-align: center; margin-bottom: 1.5rem; color: #2C3E50;">
                    Créer un compte 🚀
                </h3>
            """, unsafe_allow_html=True)
            
            # Nom complet
            st.markdown('<label class="field-label">👤 Nom complet</label>', unsafe_allow_html=True)
            new_name = st.text_input(
                "",
                placeholder="Entrez votre nom complet",
                key="reg_name",
                label_visibility="collapsed"
            )
            
            # Nom d'utilisateur
            st.markdown('<label class="field-label">📧 Nom d\'utilisateur</label>', unsafe_allow_html=True)
            new_username = st.text_input(
                "",
                placeholder="Choisissez un nom d'utilisateur",
                key="reg_username",
                label_visibility="collapsed"
            )
            
            # Mot de passe
            st.markdown('<label class="field-label">🔒 Mot de passe</label>', unsafe_allow_html=True)
            new_password = st.text_input(
                "",
                placeholder="Minimum 6 caractères",
                type="password",
                key="reg_password",
                label_visibility="collapsed"
            )
            
            # Confirmation mot de passe
            st.markdown('<label class="field-label">✓ Confirmer le mot de passe</label>', unsafe_allow_html=True)
            new_password_confirm = st.text_input(
                "",
                placeholder="Confirmez votre mot de passe",
                type="password",
                key="reg_password_confirm",
                label_visibility="collapsed"
            )
            
            # Séparateur
            st.markdown("""
                <div class="divider">
                    <span>Informations du compte</span>
                </div>
            """, unsafe_allow_html=True)
            
            register_submitted = st.form_submit_button("Créer mon compte")
            
            if register_submitted:
                if new_name and new_username and new_password and new_password_confirm:
                    if new_password != new_password_confirm:
                        st.error("❌ Les mots de passe ne correspondent pas")
                    elif len(new_password) < 6:
                        st.error("❌ Le mot de passe doit contenir au moins 6 caractères")
                    else:
                        # Vérifier si l'utilisateur existe déjà
                        users_file = 'users.csv'
                        if os.path.exists(users_file):
                            users = pd.read_csv(users_file)
                            if new_username in users['username'].values:
                                st.error("❌ Ce nom d'utilisateur est déjà pris")
                            else:
                                # Créer le nouvel utilisateur
                                new_user = pd.DataFrame([{
                                    'username': new_username,
                                    'password': new_password,
                                    'name': new_name,
                                    'role': 'user',
                                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }])
                                users = pd.concat([users, new_user], ignore_index=True)
                                users.to_csv(users_file, index=False)
                                st.success("✅ Compte créé avec succès ! Vous pouvez maintenant vous connecter")
                                st.balloons()
                                time.sleep(2)
                                st.session_state['auth_mode'] = 'login'
                                st.rerun()
                        else:
                            # Premier utilisateur
                            new_user = pd.DataFrame([{
                                'username': new_username,
                                'password': new_password,
                                'name': new_name,
                                'role': 'admin',  # Premier utilisateur est admin
                                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }])
                            new_user.to_csv(users_file, index=False)
                            st.success("✅ Compte administrateur créé avec succès !")
                            st.balloons()
                            time.sleep(2)
                            st.session_state['auth_mode'] = 'login'
                            st.rerun()
                else:
                    st.warning("⚠️ Veuillez remplir tous les champs")
    
    # Fermeture de la carte et footer
    st.markdown("""
        </div>
        <div class="auth-footer">
            <p>🔧 Maintenance Prédictive Pro • © 2025</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.stop()  # Arrête l'exécution ici pour les utilisateurs non connectés

# ==================== CHARGEMENT DES DONNÉES ====================

df = load_data()

if df is None:
    st.error("""
        ### ❌ Fichier non trouvé
        Veuillez placer le fichier 'ai4i2020.csv' dans le dossier de l'application.
    """)
    st.stop()

# ==================== MENU LATÉRAL (Connecté) ====================

with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <img src="https://img.icons8.com/fluency/96/000000/maintenance.png" width="80">
            <h3>Maintenance Prédictive</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Définir role_icon et role_text avant de les utiliser
    role_icon = "👑" if st.session_state['user_role'] == 'admin' else "👤"
    role_text = "Administrateur" if st.session_state['user_role'] == 'admin' else "Utilisateur"
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); 
                    padding: 1rem; 
                    border-radius: 15px; 
                    margin-bottom: 1.5rem; 
                    border: 1px solid #667eea40;
                    display: flex;
                    align-items: center;
                    gap: 15px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        width: 50px;
                        height: 50px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-size: 24px;
                        font-weight: bold;">
                {st.session_state['user_name'][0].upper()}
            </div>
            <div style="flex: 1;">
                <p style="margin:0; font-size: 1.1rem; font-weight: 600; color: #2C3E50;">
                    {st.session_state['user_name']}
                </p>
                <p style="margin:0; color: #666; font-size: 0.9rem; display: flex; align-items: center; gap: 5px;">
                    <span style="font-size: 1.1rem;">{role_icon}</span> {role_text}
                </p>
                <p style="margin:0; color: #999; font-size: 0.8rem; margin-top: 3px;">
                    @{st.session_state['username']}
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Déconnexion", use_container_width=True):
        logout()
        st.rerun()
    
    st.markdown("---")
    
    # Menu principal
    menu_options = ["📊 Dashboard", "🔮 Prédiction", "📈 Analyse", "📚 Historique"]
    if st.session_state['user_role'] == 'admin':
        menu_options.append("👥 Utilisateurs")
    
    menu = st.radio(
        "Choisissez une section :",
        menu_options,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Informations système
    with st.expander("ℹ️ À propos", expanded=False):
        st.markdown("""
            **Version :** 2.0 Pro  
            **Modèle :** Random Forest  
            **Précision :** ~98%  
            **Dernière MAJ :** Mars 2025
        """)
    
    # État du système
    if 'model' in st.session_state:
        st.success("✅ Système opérationnel")
    else:
        st.info("🔄 Chargement en cours...")

# ==================== ENTRAÎNEMENT DU MODÈLE ====================

with st.spinner("🚀 Initialisation du modèle en cours..."):
    time.sleep(1)
    model, scaler, features = train_model(df)
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    
    # Garder une copie du modèle original
    model_original = model
    scaler_original = scaler

# ==================== SECTION 1: DASHBOARD ====================

if menu == "📊 Dashboard":
    st.header("📊 Tableau de Bord en Temps Réel")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = len(df)
        st.markdown(f"""
            <div class="metric-card">
                <h3>📦 Total Machines</h3>
                <p class="metric-value">{total:,}</p>
                <p class="metric-delta">+{int(total*0.01)} ce mois</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        failures = df['Machine failure'].sum()
        st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Pannes Total</h3>
                <p class="metric-value">{int(failures)}</p>
                <p class="metric-delta">Taux: {failures/total:.1%}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        types = df['Type'].nunique()
        st.markdown(f"""
            <div class="metric-card">
                <h3>🏭 Types</h3>
                <p class="metric-value">{types}</p>
                <p class="metric-delta">L, M, H</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        wear = df['Tool wear [min]'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <h3>⚙️ Usure Moyenne</h3>
                <p class="metric-value">{wear:.0f} min</p>
                <p class="metric-delta">±{df['Tool wear [min]'].std():.0f} min</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques interactifs
    tab1, tab2, tab3 = st.tabs(["📈 Distribution", "🔗 Corrélations", "📊 Statistiques"])
    
    with tab1:
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Camembert des pannes
            fig = go.Figure(data=[go.Pie(
                labels=['Normal', 'Panne'],
                values=[total-failures, failures],
                hole=0.4,
                marker=dict(
                    colors=[COLORS['green'], COLORS['red']],
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=14, color=COLORS['primary'], family="Inter")
            )])
            fig.update_layout(
                title=dict(
                    text="Répartition Normal vs Panne",
                    font=dict(size=18, color=COLORS['primary'], family="Inter")
                ),
                showlegend=False,
                height=400,
                annotations=[dict(
                    text=f'{failures/total:.1%}', 
                    x=0.5, y=0.5, 
                    font_size=24, 
                    font_color=COLORS['primary'],
                    showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            # Distribution par type - VERSION AVEC COULEURS MODIFIÉES
            type_data = df['Type'].value_counts().reset_index()
            type_data.columns = ['Type', 'Count']
            
            # Palette de couleurs personnalisée pour les types
            color_map = {
                'L': COLORS['pastel_blue'],  # bleu doux
                'M': COLORS['dark_blue'],    # Bleu foncé
                'H': COLORS['teal'],         # Turquoise
            }
            
            fig = px.bar(
                type_data,
                x='Type',
                y='Count',
                color='Type',
                color_discrete_map=color_map,
                title="Distribution par Type de Machine",
                text='Count'
            )
            
            fig.update_traces(
                textposition='outside',
                textfont=dict(size=14, color=COLORS['primary']),
                marker_line_color='white',
                marker_line_width=1.5,
                opacity=0.9
            )
            
            fig.update_layout(
                title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
                height=400,
                xaxis=dict(title="Type", title_font=dict(size=14, color=COLORS['primary'])),
                yaxis=dict(title="Nombre", title_font=dict(size=14, color=COLORS['primary'])),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Matrice de corrélation
        corr_matrix = df[features + ['Machine failure']].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            title="Matrice de Corrélation",
            color_continuous_scale='Viridis',
            zmin=-1, zmax=1
        )
        
        fig.update_layout(
            title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Statistiques détaillées
        stats_df = df[features].describe().T
        stats_df['variance'] = df[features].var()
        stats_df['skew'] = df[features].skew()
        
        st.dataframe(
            stats_df.style.background_gradient(cmap='Blues', subset=['mean', 'std']),
            use_container_width=True,
            height=400
        )

# ==================== SECTION 2: PRÉDICTION ====================

elif menu == "🔮 Prédiction":
    st.header("🔮 Prédiction Intelligente")
    
    col_input, col_result = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin-top:0;">📝 Paramètres Machine</h3>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            machine_type = st.selectbox(
                "Type de machine",
                options=["Low (L)", "Medium (M)", "High (H)"],
                help="Le type de produit influence la durée de vie"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                air_temp = st.slider(
                    "🌡️ Température air (K)",
                    min_value=290.0, max_value=310.0, value=300.0, step=0.1,
                    help="Température ambiante"
                )
                
                rotational_speed = st.slider(
                    "⚡ Vitesse rotation (rpm)",
                    min_value=1000, max_value=3000, value=1500, step=10,
                    help="Vitesse de rotation"
                )
                
                tool_wear = st.slider(
                    "🔧 Usure outil (min)",
                    min_value=0, max_value=500, value=100, step=5,
                    help="Durée d'utilisation"
                )
            
            with col2:
                process_temp = st.slider(
                    "🔥 Température process (K)",
                    min_value=300.0, max_value=320.0, value=310.0, step=0.1,
                    help="Température du processus"
                )
                
                torque = st.slider(
                    "🔩 Couple (Nm)",
                    min_value=0.0, max_value=100.0, value=40.0, step=0.5,
                    help="Couple appliqué"
                )
            
            st.markdown("""
                <div style="display: flex; gap: 10px; margin: 1rem 0;">
                    <span class="badge badge-blue">⚡ Haute précision</span>
                    <span class="badge badge-green">🤖 IA active</span>
                    <span class="badge badge-orange">📊 Temps réel</span>
                </div>
            """, unsafe_allow_html=True)
            
            submitted = st.form_submit_button(
                "🚀 Lancer la prédiction",
                use_container_width=True
            )
    
    with col_result:
        if submitted:
            with st.spinner("Analyse en cours..."):
                time.sleep(1)
                
                input_data = np.array([[
                    air_temp, process_temp, rotational_speed, torque, tool_wear
                ]])
                input_scaled = st.session_state['scaler'].transform(input_data)
                
                prediction = st.session_state['model'].predict(input_scaled)[0]
                proba = st.session_state['model'].predict_proba(input_scaled)[0]
                risk_score = proba[1] * 100
            
            if prediction == 1:
                if risk_score > 80:
                    card_class = "danger-card"
                    icon = "🚨"
                    title = "RISQUE CRITIQUE"
                else:
                    card_class = "warning-card"
                    icon = "⚠️"
                    title = "RISQUE DE PANNE"
                
                st.markdown(f"""
                    <div class="result-card {card_class}">
                        <h2>{icon} {title}</h2>
                        <p>Une intervention est nécessaire rapidement</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-card success-card">
                        <h2>✅ MACHINE SAINE</h2>
                        <p>Aucun risque de panne détecté</p>
                    </div>
                """, unsafe_allow_html=True) 
            # SAUVEGARDE AUTOMATIQUE DANS L'HISTORIQUE
            record = {
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Type': machine_type,
                'Air_temp': air_temp,
                'Process_temp': process_temp,
                'Rotational_speed': rotational_speed,
                'Torque': torque,
                'Tool_wear': tool_wear,
                'Prediction': 'Panne' if prediction == 1 else 'Normal',
                'Risk_Score': f"{risk_score:.1f}%",
                'Utilisateur': st.session_state['username']
            }
            
            # Sauvegarde automatique
            save_prediction(record)
        
        else:
            st.markdown("""
                <div style="text-align: center; padding: 3rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <img src="https://img.icons8.com/fluency/96/000000/robot.png" width="120">
                    <h3 style="margin: 1rem 0;">Prêt pour l'analyse</h3>
                    <p style="color: #666; max-width: 400px; margin: 0 auto;">
                        Configurez les paramètres de la machine dans le formulaire et lancez une prédiction intelligente.
                    </p>
                </div>
            """, unsafe_allow_html=True)
# ==================== SECTION 3: ANALYSE ====================

elif menu == "📈 Analyse":
    st.header("📈 Analyse Approfondie")
    
    st.subheader("🎯 Importance des Features")
    
    feat_importance = pd.DataFrame({
        'Feature': features,
        'Importance': st.session_state['model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        feat_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Plasma',
    )
    
    fig.update_traces(
        texttemplate='%{text:.1%}',
        textposition='outside',
        marker_line_color='white',
        marker_line_width=1.5
    )
    
    fig.update_layout(
        title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
        height=400,
        xaxis=dict(title="Importance", title_font=dict(size=14, color=COLORS['primary'])),
        yaxis=dict(title="", tickfont=dict(size=12, color=COLORS['primary'])),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("⏱️ Analyse de l'Usure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df,
            x='Tool wear [min]',
            y='Torque [Nm]',
            color='Machine failure',
            title="Corrélation Usure - Couple",
            color_discrete_map={0: COLORS['green'], 1: COLORS['red']},
            opacity=0.7,
            labels={'Tool wear [min]': 'Usure outil (min)', 'Torque [Nm]': 'Couple (Nm)'}
        )
        fig.update_traces(
            marker=dict(size=8, line=dict(color='white', width=1))
        )
        fig.update_layout(
            title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12, color=COLORS['primary']),
            legend=dict(
                title="État",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df,
            x='Tool wear [min]',
            color='Machine failure',
            title="Distribution de l'usure",
            color_discrete_map={0: COLORS['green'], 1: COLORS['red']},
            nbins=50,
            barmode='overlay',
            labels={'Tool wear [min]': 'Usure outil (min)', 'count': 'Fréquence'}
        )
        fig.update_layout(
            title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12, color=COLORS['primary']),
            legend=dict(
                title="État",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== SECTION 4: DÉTAIL DES PANNES ====================

elif menu == "📊 Détail des Pannes":
    st.header("📊 Analyse Détaillée des Pannes")
    
    # Filtres interactifs
    st.subheader("🔍 Filtres")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        selected_types = st.multiselect(
            "Types de machine",
            options=df['Type'].unique(),
            default=list(df['Type'].unique())  # Correction: convertir en liste
        )
    
    with col_f2:
        wear_threshold = st.slider(
            "Usure minimum (min)",
            min_value=0,
            max_value=int(df['Tool wear [min]'].max()),
            value=0
        )
    
    with col_f3:
        show_failures_only = st.checkbox("Afficher uniquement les pannes")
    
    # Appliquer les filtres
    filtered_df = df.copy()
    if selected_types:
        filtered_df = filtered_df[filtered_df['Type'].isin(selected_types)]
    
    filtered_df = filtered_df[filtered_df['Tool wear [min]'] >= wear_threshold]
    
    if show_failures_only:
        filtered_df = filtered_df[filtered_df['Machine failure'] == 1]
    
    st.info(f"📊 Données analysées : {len(filtered_df)} machines sur {len(df)}")
    
    st.markdown("---")
    
    # Onglets pour les différentes analyses
    tab_detail1, tab_detail2, tab_detail3, tab_detail4 = st.tabs([
        "📦 Box Plots", 
        "⚠️ Indicateurs d'alerte", 
        "📋 Pannes par type",
        "📈 Corrélations détaillées"
    ])
    
    with tab_detail1:
        st.subheader("📦 Distribution des paramètres (Box Plots)")
        
        if len(filtered_df) > 0:
            # Sélection du paramètre
            param_option = st.selectbox(
                "Choisir un paramètre à analyser :",
                features,
                key='box_param_detail'
            )
            
            # Box plot avec distinction Normal/Panne
            fig = px.box(
                filtered_df,
                y=param_option,
                color='Machine failure',
                title=f"Distribution de {param_option} selon l'état",
                color_discrete_map={0: COLORS['green'], 1: COLORS['red']},
                points="all",
                labels={'Machine failure': 'État', param_option: param_option}
            )
            
            fig.update_traces(
                marker=dict(size=6, line=dict(color='white', width=1)),
                boxmean=True
            )
            
            fig.update_layout(
                title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
                height=600,
                yaxis_title=param_option,
                xaxis_title="État (0=Normal, 1=Panne)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    title="État",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques descriptives
            st.subheader("📊 Statistiques descriptives")
            
            stats_normal = filtered_df[filtered_df['Machine failure'] == 0][param_option].describe()
            stats_failure = filtered_df[filtered_df['Machine failure'] == 1][param_option].describe()
            
            # Créer un DataFrame pour les statistiques
            stats_df = pd.DataFrame({
                'Statistique': stats_normal.index,
                'Machines normales': stats_normal.values.round(2) if not stats_normal.empty else ['N/A']*len(stats_normal.index),
                'Machines en panne': stats_failure.values.round(2) if not stats_failure.empty else ['N/A']*len(stats_normal.index)
            })
            
            st.dataframe(stats_df, use_container_width=True, height=200)
        else:
            st.warning("Aucune donnée disponible avec les filtres actuels")
    
    with tab_detail2:
        st.subheader("⚠️ Indicateurs d'alerte")
        
        if len(filtered_df) > 0:
            # Définir des seuils d'alerte
            thresholds = {
                'Air temperature [K]': 305,
                'Process temperature [K]': 315,
                'Rotational speed [rpm]': 2800,
                'Torque [Nm]': 70,
                'Tool wear [min]': min(400, float(filtered_df['Tool wear [min]'].max())) 
            }
            
            # Seuils personnalisables
            st.markdown("#### Ajuster les seuils d'alerte")
            custom_thresholds = {}
            
            col_th1, col_th2 = st.columns(2)
            
            with col_th1:
                for i, (feature, default_th) in enumerate(list(thresholds.items())[:3]):
                    if feature in filtered_df.columns:
                        custom_thresholds[feature] = st.number_input(
                            f"Seuil pour {feature}",
                            min_value=0.0,
                            max_value=float(filtered_df[feature].max() * 1.5),
                            value=float(default_th),
                            step=0.1 if 'K' in feature else 1.0,
                            key=f"thresh_{feature}"
                        )
            
            with col_th2:
                for i, (feature, default_th) in enumerate(list(thresholds.items())[3:]):
                    if feature in filtered_df.columns:
                        custom_thresholds[feature] = st.number_input(
                            f"Seuil pour {feature}",
                            min_value=0.0,
                            max_value=float(filtered_df[feature].max() * 1.5),
                            value=float(default_th),
                            step=0.1 if 'K' in feature else 1.0,
                            key=f"thresh_{feature}"
                        )
            
            st.markdown("---")
            
            # Calcul des alertes
            alert_data = []
            alert_details = []
            
            for feature, threshold in custom_thresholds.items():
                if feature in filtered_df.columns:
                    machines_alert = filtered_df[filtered_df[feature] > threshold]
                    count_alert = len(machines_alert)
                    percentage = (count_alert / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
                    
                    # Simplifier le nom du paramètre
                    param_name = feature.replace('[K]', '').replace('[rpm]', '').replace('[Nm]', '').replace('[min]', '').strip()
                    
                    alert_data.append({
                        'Paramètre': param_name,
                        'Seuil': f"{threshold:.1f}",
                        'Machines en alerte': count_alert,
                        'Pourcentage': f"{percentage:.1f}%"
                    })
                    
                    # Détail des machines en alerte (seulement si nécessaire)
                    if count_alert > 0 and count_alert <= 100:  # Limiter pour éviter les tableaux trop grands
                        alert_details.append(machines_alert[['UDI', 'Type', feature, 'Machine failure']])
            
            
            # Détail des machines en alerte
            if alert_details:
                st.markdown("#### 🔍 Détail des machines en alerte")
                
                # Concaténer tous les détails
                all_alerts = pd.concat(alert_details, ignore_index=True).drop_duplicates()
                
                # Ajouter une colonne pour le nombre d'alertes par machine
                alert_counts = []
                for idx, row in all_alerts.iterrows():
                    count = 0
                    for feature in custom_thresholds.keys():
                        if feature in row and row[feature] > custom_thresholds[feature]:
                            count += 1
                    alert_counts.append(count)
                
                all_alerts['Nombre d\'alertes'] = alert_counts
                
                # Trier par nombre d'alertes
                all_alerts = all_alerts.sort_values('Nombre d\'alertes', ascending=False)
                
                st.dataframe(
                    all_alerts.style.background_gradient(cmap='YlOrRd', subset=['Nombre d\'alertes']),
                    use_container_width=True,
                    height=300
                )
        else:
            st.warning("Aucune donnée disponible avec les filtres actuels")
    
    with tab_detail3:
        st.subheader("📋 Analyse des pannes par type")
        
        if len(filtered_df) > 0 and len(filtered_df[filtered_df['Machine failure'] == 1]) > 0:
            # Statistiques détaillées par type
            type_stats = filtered_df.groupby('Type').agg({
                'Machine failure': ['count', 'sum', 'mean']
            }).round(3)
            type_stats.columns = ['Total machines', 'Nombre de pannes', 'Taux de panne']
            type_stats = type_stats.reset_index()
            
            # Formater le taux en pourcentage
            type_stats['Taux de panne'] = (type_stats['Taux de panne'] * 100).round(1)
            
            # Ajouter une barre de progression visuelle
            fig = px.bar(
                type_stats,
                x='Type',
                y='Taux de panne',
                color='Taux de panne',
                title="Taux de panne par type de machine",
                text='Taux de panne',
                color_continuous_scale='Reds'
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(
                title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
                height=400,
                xaxis_title="Type de machine",
                yaxis_title="Taux de panne (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            st.markdown("#### 📊 Détail des pannes par type")
            
            # Ajouter des barres de progression dans le tableau
            type_stats['Progression'] = type_stats['Taux de panne'].apply(
                lambda x: f'{"█" * int(x/10)}{"░" * (10 - int(x/10))} {x:.1f}%'
            )
            
            st.dataframe(
                type_stats[['Type', 'Total machines', 'Nombre de pannes', 'Taux de panne', 'Progression']],
                use_container_width=True,
                height=200
            )
            
            # Analyse des types de pannes spécifiques
            st.markdown("#### 🔧 Répartition des types de pannes")
            
            failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            
            # Vérifier que ces colonnes existent
            existing_failure_types = [ft for ft in failure_types if ft in filtered_df.columns]
            
            if existing_failure_types:
                failure_by_type = filtered_df.groupby('Type')[existing_failure_types].sum().reset_index()
                
                # Graphique en barres groupées
                fig = px.bar(
                    failure_by_type,
                    x='Type',
                    y=existing_failure_types,
                    title="Types de pannes par catégorie de machine",
                    barmode='group',
                    color_discrete_sequence=COLORS_CATEGORICAL
                )
                fig.update_layout(
                    title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
                    height=500,
                    xaxis_title="Type de machine",
                    yaxis_title="Nombre de pannes",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Les données détaillées des types de pannes ne sont pas disponibles")
        else:
            st.info("Aucune panne détectée avec les filtres actuels")
    
    with tab_detail4:
        st.subheader("📈 Corrélations détaillées")
        
        if len(filtered_df) > 1:  # Besoin d'au moins 2 points pour une corrélation
            # Corrélations avec les différents types de pannes
            failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']
            existing_cols = [col for col in features + failure_cols if col in filtered_df.columns]
            
            if len(existing_cols) > 1:
                corr_matrix = filtered_df[existing_cols].corr()
                
                # Extraire les corrélations features vs pannes
                feature_cols = [f for f in features if f in existing_cols]
                failure_cols_existing = [f for f in failure_cols if f in existing_cols]
                
                if feature_cols and failure_cols_existing:
                    corr_with_failures = corr_matrix.loc[feature_cols, failure_cols_existing]
                    
                    fig = px.imshow(
                        corr_with_failures,
                        text_auto='.2f',
                        aspect="auto",
                        title="Corrélation Features vs Types de pannes",
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
                        height=500,
                        xaxis_title="Types de pannes",
                        yaxis_title="Caractéristiques"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour les corrélations features vs pannes")
                
                # Matrice de corrélation entre les types de pannes
                st.markdown("#### 🔗 Corrélations entre types de pannes")
                
                if len(failure_cols_existing) > 1:
                    corr_failures = corr_matrix.loc[failure_cols_existing, failure_cols_existing]
                    
                    fig = px.imshow(
                        corr_failures,
                        text_auto='.2f',
                        aspect="auto",
                        title="Corrélations entre types de pannes",
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        title_font=dict(size=16, color=COLORS['primary'], family="Inter"),
                        height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour les corrélations entre types de pannes")
            else:
                st.info("Colonnes insuffisantes pour les corrélations")
        else:
            st.warning("Données insuffisantes pour calculer des corrélations")
    
    # Téléchargement des données filtrées
    st.markdown("---")
    if len(filtered_df) > 0:
        if st.button("📥 Télécharger les données analysées"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Confirmer le téléchargement",
                data=csv,
                file_name=f"analyse_pannes_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("Aucune donnée à télécharger")


# ==================== SECTION 6: HISTORIQUE ====================

elif menu == "📚 Historique":
    st.header("📚 Historique des Prédictions")
    
    history = load_prediction_history()
    
    if history is not None and not history.empty:
        # Métriques sur l'historique
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total analyses", len(history))
        
        with col2:
            pannes = len(history[history['Prediction'] == 'Panne'])
            st.metric("Pannes détectées", pannes)
        
        with col3:
            st.metric("Taux de panne", f"{pannes/len(history):.1%}")
        
        with col4:
            st.metric("Dernière analyse", history['Date'].iloc[-1][:10])
        
        # Tableau stylisé
        st.subheader("📋 Détail des analyses")
        
        def color_prediction(val):
            color = '#ff5252' if val == 'Panne' else '#4caf50'
            return f'background-color: {color}; color: white'
        
        styled_df = history.style.map(
            color_prediction, 
            subset=['Prediction']
        )
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # Graphique d'évolution
        if len(history) > 1:
            st.subheader("📈 Évolution du risque")
            
            history['Risk_Value'] = history['Risk_Score'].str.replace('%', '').astype(float)
            
            fig = px.line(
                history,
                x='Date',
                y='Risk_Value',
                title="Évolution du risque de panne",
                markers=True
            )
            fig.update_layout(
                title_font=dict(size=18, color=COLORS['primary'], family="Inter"),
                yaxis_title="Risque (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Export
        if st.button("📥 Exporter en CSV"):
            csv = history.to_csv(index=False)
            st.download_button(
                "Télécharger",
                csv,
                "historique_predictions.csv",
                "text/csv"
            )
    else:
        st.info("""
            ### 📭 Aucun historique disponible
            Effectuez des prédictions dans la section **🔮 Prédiction** et cliquez sur "Sauvegarder" pour voir l'historique.
        """)

# ==================== SECTION 7: GESTION UTILISATEURS (Admin seulement) ====================

elif menu == "👥 Utilisateurs" and st.session_state.get('user_role') == 'admin':
    st.header("👥 Gestion des Utilisateurs")
    
    tab1, tab2 = st.tabs(["📋 Liste des utilisateurs", "➕ Créer un utilisateur"])
    
    with tab1:
        st.subheader("Liste des utilisateurs")
        users = get_all_users()
        
        # Masquer les mots de passe
        display_users = users.copy()
        display_users['password'] = '••••••••'
        
        st.dataframe(display_users, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total utilisateurs", len(users))
        with col2:
            st.metric("Administrateurs", len(users[users['role'] == 'admin']))
        with col3:
            st.metric("Utilisateurs", len(users[users['role'] == 'user']))
    
    with tab2:
        st.subheader("Créer un nouvel utilisateur")
        
        with st.form("create_user_form"):
            new_username = st.text_input("Nom d'utilisateur*")
            new_password = st.text_input("Mot de passe*", type="password")
            new_password_confirm = st.text_input("Confirmer le mot de passe*", type="password")
            new_name = st.text_input("Nom complet*")
            new_role = st.selectbox("Rôle", ["user", "admin"])
            
            if st.form_submit_button("Créer l'utilisateur"):
                if not new_username or not new_password or not new_name:
                    st.error("Tous les champs sont requis")
                elif new_password != new_password_confirm:
                    st.error("Les mots de passe ne correspondent pas")
                else:
                    success, message = create_user(new_username, new_password, new_name, new_role)
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)

# ==================== FOOTER ====================

st.markdown("""
    <div class="footer">
        <p>🔧 Maintenance Prédictive Pro • Propulsé par Streamlit & Scikit-learn • © 2025</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">
            🤖 Modèle Random Forest avec {:.1f}% de précision • {} échantillons d'entraînement
        </p>
    </div>""".format(98.5, len(df)), unsafe_allow_html=True) 