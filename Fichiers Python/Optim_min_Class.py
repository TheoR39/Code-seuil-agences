import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
import scipy.optimize as optimize
from DataCharger_Class import DataCharger
from BasicStats_Class import BasicStats
from scipy import stats
from skopt import gp_minimize
import os

# Pour obtenir les données: 
# - On a besoin des valeurs de flux historique par agence.
# - On a besoin d'accéder aux quantiles
# - On a besoin d'accéder aux fréquences annuelles pour chaque agence
# - On pourrait donc créer une classe intermédiaire qui récupère des données statistiques précises
# pour toutes les agences (fréquence de flux négatifs, récupération des seuils à proba de rupture
# fixée, stockage des quantiles pour comparaison, sans forcément tout afficher).


# Pour le moment, on crée donc la donnée suivante, avec pour chaque agence:
# Une colonne 'flux_net' qui contient un dictionnaire type (date, flux_net) pour chaque journée
# Une colonne 'nb_j_ouvres' qui renseigne le nombre de jours ouvrés pour l'agence en question sur l'année dernière
# Une colonne 'code_agence' pour savoir de quelle agence on parle (quand même)
# Une colonne 'freq_pos' pour connaître à quel degré l'agence était créditrice / débitrice l'année passée
# Deux colonnes de 'seuil' pour avoir une base de comparaison



class Optim_min_threshold:


    def __init__(self, filepath : str, filepath_optim : str,  c_trans : float = 150,
                c_rupt : float = 1000, t_int = 0.02):
        '''Constructeur de la classe d'optimisation du seuil min'''

        self.data = None  # Données pour l'optimisation
        self.random_state = None  # Graine pour le bootstrap
        self.threshold_order = None  # Valeur seuil pour la commande dans le modèle
        self.solution = {}   # Dictionnaire vide pour contenir les solutions (par agence)
        self.bootstrap_ratio = 0.85  # On va se baser à la fois sur du bootstrap et sur de l'estimation

        # Vérification des arguments du constructeur: 
        if not all(isinstance(x, float) for x in (c_trans, c_rupt)):
            raise TypeError("Tous les coûts rentrés doivent être des flottants")
        if not isinstance(t_int, int):
            raise TypeError("Le taux d'intérêt (annuel) doit être un entier")
        self.t_int = t_int  # Stockage du taux d'intérêt annuel (fixé par défaut à 2%)
        # En particulier, il faut penser à le convertir en taux d'intérêt journalier
        if not (50 <= c_trans <= 2000):
            raise ValueError("Le coût de transport doit être compris entre 50 et 2000 MAD")
        self.c_trans = c_trans   # Stockage du coût de transport
        if not c_rupt > c_trans:
            raise ValueError("Le coût de rupture doit nécessairement être supérieur au coût de transport")
        self.lambd = c_rupt  # Stockage du coût associé à la rupture (avec c_rupt > c_trans)

        # Vérification des fichiers:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Le fichier '{filepath}' n'existe pas ou est introuvable")
        if not filepath.lower().endswith('.csv'):
            raise ValueError(f"Le fichier '{filepath}' n'est pas un CSV.")
        self.filepath = filepath
        if not os.path.isfile(filepath_optim):
            raise FileNotFoundError(f"Le fichier '{filepath_optim}' n'existe pas ou est introuvable")
        if not filepath.lower().endswith('.csv'):
            raise ValueError(f"Le fichier '{filepath_optim}' n'est pas un CSV.")
        self.filepath_optim = filepath_optim

    
    def remplissage_data_optim(self, year: Optional[int] = 2024):
        '''Même logique que pour le clustering: on récupère la donnée d'intérêt
        et on initialise les clés du dictionnaire des solutions'''

        if not hasattr(self,'filepath') or not isinstance(self.filepath, str):
            raise AttributeError("L'attribut 'filepath' n'existe pas ou est incorrect")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Aucun fichier trouvé à l'adresse {self.filepath}")
        dataset = DataCharger(self.filepath) # Chargement des données complètes
        dataset = dataset.preparer_donnees()[0] # Données nettoyées chargées
        if dataset is None or dataset.empty:
            raise ValueError("Le dataset chargé est vide")
        if dataset.index is None or dataset.index.empty:
            raise ValueError("Le dataset ne contient pas d'index")
        if 'code_agence' not in dataset.columns:
            raise ValueError("'code_agence' n'est pas présent dans les colonnes du dataset")
        liste_agences = sorted(dataset['code_agence'].dropna().unique().tolist())
        lignes  = []
        for agence in liste_agences:
            try:
                class_data = DataCharger(filepath = self.filepath, code = agence, annee = year)
                class_data.assignation_simple()
                # print(f"Agence {agence} — class_data type: {type(class_data)}")  # Débogage
                # if hasattr(class_data, 'data'):
                #     print(f"Agence {agence} — class_data.data type: {type(class_data.data)}")
                data_agence = BasicStats(class_data)
                print(f"Agence {agence} — data_agence.data type: {type(data_agence.data)}")
                lignes.append(data_agence.data_retrieval_optim())
            except Exception as e:
                print(f"Erreur lors du chargement des données de l'agence {agence} : {e}")
                continue
        if not lignes:
            raise ValueError("Aucune donnée n'a pu être traitée")
        data_freq = pd.DataFrame(lignes)
        if 'code_agence' not in data_freq:
            raise ValueError("Impossible de créer l'index: 'code'agence' n'existe pas")
        return data_freq
    

    def choice_quantile(self, quantile: Optional[float] = 0.7):
        '''Méthode pour choisir le quantile pour réapprovisionnement'''

        possible_choices = np.arange(0.5,0.95, 0.05)  # On va de la médiane au quantile 0.90
        if not isinstance(quantile, float):
            raise TypeError("Le quantile doit être un float")
        if quantile not in possible_choices:
            raise ValueError (f"Le quantile sélectionné doit être dans la liste suivante: {possible_choices}")
        self.threshold_order = quantile   # Permet de sélectionner le quantile voulu pour la commande
        # (c'est-à-dire le réapprovisionnement de l'agence)


    def set_random_seed(self, seed: int):
        if not hasattr(self,'random_state'):
            raise AttributeError("self n'a pas d'attribut 'random_state'")
        else:
            if not isinstance(seed, int):
                raise TypeError("La graine doit obligatoirement être un entier")
            self.random_state = seed


