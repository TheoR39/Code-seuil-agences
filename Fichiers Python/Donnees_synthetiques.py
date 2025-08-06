import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta
import zipfile
import os

# Configuration
np.random.seed(42)
random.seed(42)

def generate_alphanumeric_id(length=12):
    """Génère un identifiant alphanumérique"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def generate_banking_data(agence_start, agence_end, n_observations=40000):
    """Génère des données bancaires synthétiques pour une tranche d'agences"""
    
    data = []
    
    # Libellés simplifiés avec majorité de RETRAITS, VERSEMENTS, ESPECES et quelques VIREMENTS
    libelle_court_operations = ["RETRAITS", "VERSEMENTS", "ESPECES", "VIREMENTS"]
    libelle_long_operations = ["RETRAITS", "VERSEMENTS", "ESPECES", "VIREMENTS"]
    
    # Codes agences dans la tranche spécifiée
    agences = list(range(agence_start, agence_end + 1))
    
    for _ in range(n_observations):
        # Génération de chaque ligne
        row = {
            'identifiant_operation': generate_alphanumeric_id(10),
            'reference_operation': generate_alphanumeric_id(12),
            'date_operation': f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            'heure_operation': f"0 days {random.randint(8,18):02d}:{random.randint(0,59):02d}:00",
            'code_agence': random.choice(agences),
            'identifiant_compte': generate_alphanumeric_id(14),
            'identifiant_client': generate_alphanumeric_id(8),
            'code_banque': np.random.choice(['BPP', 'BE', 'BI'], p=[0.85, 0.14, 0.01]),
            'montant_operation': round(random.uniform(10, 50000), 2),
            'sens_operation': random.choice(['D', 'C']),
            'libelle_court_operation': np.random.choice(['RETRAITS', 'VERSEMENTS', 'ESPECES', 'VIREMENTS'], 
                                                       p=[0.35, 0.35, 0.25, 0.05]),
            'libelle_long_operation': np.random.choice(['RETRAITS', 'VERSEMENTS', 'ESPECES', 'VIREMENTS'], 
                                                      p=[0.35, 0.35, 0.25, 0.05]),
            'code_famille_operation': random.randint(1, 15),
            'code_type_operation': random.randint(1, 25),
            'etat_operation': np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.05, 0.8, 0.05, 0.03, 0.03, 0.02, 0.02]),
            'numero_caisse': np.random.choice([None, 1.0, 2.0, 3.0, 4.0, 5.0], p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06]),
            'code_marche': np.random.choice([None, 1.0, 2.0, 3.0], p=[0.6, 0.2, 0.1, 0.1]),
            'devise': 'MAD',
            'motif_operation': random.choice(['STANDARD', 'URGENT', 'NORMAL', 'PRIORITAIRE']),
            'application_origine_operation': np.random.choice(['OA', 'MB', 'PST'], p=[0.9, 0.07, 0.03]),
            'date_valeur': ''  # Valeur vide comme demandé
        }
        data.append(row)
    
    return pd.DataFrame(data)

def create_all_files():
    """Crée tous les fichiers par tranches de 20 agences et les compresse"""
    
    # Créer un dossier temporaire pour les fichiers
    temp_dir = "temp_banking_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Générer les fichiers par tranches de 20 agences
    for start_agence in range(0, 101, 20):
        end_agence = min(start_agence + 19, 100)
        
        print(f"Génération des données pour les agences {start_agence} à {end_agence}...")
        
        # Générer les données
        df = generate_banking_data(start_agence, end_agence, 40000)
        
        # Sauvegarder le fichier CSV
        filename = f"donnees_bancaires_agences_{start_agence}_{end_agence}.csv"
        filepath = os.path.join(temp_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"Fichier {filename} créé avec {len(df)} observations")
    
    # Créer le fichier ZIP
    zip_filename = "donnees_bancaires_synthetiques.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(temp_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(temp_dir, filename)
                zipf.write(filepath, filename)
    
    # Nettoyer le dossier temporaire
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\nTous les fichiers ont été créés et compressés dans {zip_filename}")
    print("Contenu du ZIP :")
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for info in zipf.infolist():
            print(f"  - {info.filename} ({info.file_size} bytes)")

def show_sample_data():
    """Affiche un échantillon des données générées"""
    print("\nÉchantillon des données générées :")
    df_sample = generate_banking_data(0, 19, 100)
    print(df_sample.head())
    print(f"\nTypes de données :")
    print(df_sample.dtypes)
    print(f"\nDistribution code_banque :")
    print(df_sample['code_banque'].value_counts())
    print(f"\nDistribution etat_operation :")
    print(df_sample['etat_operation'].value_counts())
    print(f"\nDistribution application_origine_operation :")
    print(df_sample['application_origine_operation'].value_counts())

# Fonctions disponibles pour usage externe :
# - generate_banking_data(agence_start, agence_end, n_observations=40000)
# - create_all_files()
# - show_sample_data()

if __name__ == "__main__":
    # Afficher un échantillon
    show_sample_data()
    
    # Créer tous les fichiers
    create_all_files()