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

# Pour le moment, on crée donc la donnée suivante, avec pour chaque agence:
# Une colonne 'flux_net' qui contient un dictionnaire type (date, flux_net) pour chaque journée
# Une colonne 'nb_j_ouvres' qui renseigne le nombre de jours ouvrés pour l'agence en question sur l'année dernière
# Une colonne 'code_agence' pour savoir de quelle agence on parle (quand même)
# Une colonne 'freq_pos' pour connaître à quel degré l'agence était créditrice / débitrice l'année passée
# Deux colonnes de 'seuil' pour avoir une base de comparaison
# Une autre colonne qui contient le dataset complet pour les seuils dont les intervalles de confiance (au cas où)

# On suppose maintenant la donnée nécessaire créée. 
# On doit donc maintenant passer à l'étape de simulation.
# Le modèle est le suivant: au début de la simulation, on commence en mettant l'agence au seuil min
# voulu. A partir des valeurs de flux net historiques, on effectue du bootstrap, et on simule des journées
# sur plusieurs années. A chaque fin de journée (simulée par flux + seuil), on regarde la chose suivante:
# - si on passe en-dessous du quantile choisi, on compte un coût de transport et on remet au seuil
# - si on passe en-dessous de 0, on compte un coût de rupture et on remet au seuil
# - si on passe au-dessus de la barre du seuil économique, on évacue jusqu'à seuil min.

class Optim_min_threshold:


    def __init__(self, filepath : str, filepath_optim : str,  c_trans : float = 150,
                c_rupt : float = 1500, t_int : float = 0.02):
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
        self.taux_journalier = (1 + self.t_int)**(1/365) - 1  # Stockage du taux d'intérêt journalier
        if not (50 <= c_trans <= 2000):
            raise ValueError("Le coût de transport doit être compris entre 50 et 2000 MAD")
        self.c_trans = c_trans   # Stockage du coût de transport
        if not c_rupt > c_trans:
            raise ValueError("Le coût de rupture doit nécessairement être supérieur au coût de transport")
        self.c_rupt = c_rupt  # Stockage du coût associé à la rupture (avec c_rupt > c_trans)
        self.gap = self.c_trans / self.taux_journalier  # Montant à accumuler pour décharger

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
        self.data = pd.DataFrame(lignes)
        if 'code_agence' not in self.data:
            raise ValueError("Impossible de créer l'index: 'code'agence' n'existe pas")
        self.data.set_index("code_agence", inplace = True)
        return self.data
    

    def save_data_optim(self, overwrite: Optional[bool] = False):
        '''Sauvegarde des données créées pour l'optimisation à l'adresse self.filepath_optim'''

        if not self.new_filepath.lower().endswith('.csv'):
            raise ValueError("Le chemin de sauvegarde ne construit pas un fichier au format csv")
        if self.data is None:
            raise ValueError("Aucune donnée disponible pour une sauvegarde")
        if os.path.exists(self.filepath_optim):
            if overwrite:
                print("Ecrasement du fichier existant...")

            else:
                print("Le fichier existe déjà, il n'y a pas besoin de re-sauvegarder")
                print("Si vous voulez remplacer le fichier existant, relancez la fonction avec l'argument 'overwrite' à True")
                return
        self.data.to_csv(self.new_filepath, index = True)
        print(f"Données sauvegardées vers {self.new_filepath}")

    
    def load_data_optim(self):
        '''Permet de charger les données nécessaires aux méthodes d'optimisation'''

        # Vérifications sur le chemin d'accès
        if not os.path.exists(self.filepath_optim):
            raise FileNotFoundError("Le fichier de données n'a pas encore été créé")
        if not os.path.isfile(self.filepath_optim):
            raise ValueError("Le chemin d'accès spécifié ne pointe pas vers un fichier")
        if not self.filepath_optim.lower().endswith('csv'):
            raise ValueError("Le fichier n'est pas au format csv")
        # Chargement des données:
        try:
            self.data = pd.read_csv(self.filepath_optim, index_col = 0)
            print(f"Données chargées depuis {self.filepath_optim}")
        except Exception as e:
            raise IOError(f"Erreur lors du chargement des fichiers depuis {self.filepath_optim}")

    
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

    
    # On considère maintenant que les données sont propres et on code maintenant les fonctions pour
    # évaluer la méthode par agence. Une méthode finale permettra de lancer les calculs pour toutes
    # les agences.


    def generate_bootstrap_scenario(self, code_agence, n_iter: Optional[int] = 100):
        '''Génère un scénario de données pour l'agence spécifiée par bootstrap'''

        nb_j_ouvres = self.data["code_agence"]["nb_j_ouvres"]
        liste_flux = list(self.data["flux_net"].values())
        liste_scenario = np.zeros((n_iter, nb_j_ouvres))
        scenario = np.zeros(nb_j_ouvres)
        for i in range(n_iter):
            for j in range(nb_j_ouvres):
                scenario[j] = np.random.choice(liste_flux)
            liste_scenario[i] = scenario
        return liste_scenario
    

    def simulate_system_MC(self, code_agence, seuil_min):  # On pourrait aussi compter les ruptures, les passages...
        '''Simulation du modèle et calcul du coût pour un seuil_min donné.'''
        liste_scenario = self.generate_bootstrap_scenario(code_agence)
        liste_cost = []
        cost, total_cost = 0,0
        stock = seuil_min
        freq_pos = self.data.loc[code_agence, "freq_pos"] 
        for i in range(liste_scenario.shape[0]):
            cost = 0
            stock = seuil_min
            for flux in liste_scenario[i]:
                cost += self.t_int/100 * freq_pos * seuil_min  # On pénalise sur la valeur du seuil min (assez faiblement)
                stock = stock + flux
                if stock < 0:  # Rupture de l'agence en fin de journée
                    cost += self.c_rupt # On compte un coût de rupture (correspond ici à 10 passages)
                    stock = seuil_min  # On revient au seuil min
                elif 0 <= stock <= self.threshold_order:  # Stock trop bas: on commande
                    cost += self.c_trans  # On rajoute un coût de transport
                    stock = seuil_min
                elif stock > seuil_min + self.gap:
                    cost += self.c_trans
                    stock = seuil_min
            liste_cost.append(cost)
        total_cost = np.mean(liste_cost)  # Evaluation Monte-Carlo du coût associé au seuil seuil_min
        return total_cost
                

    def bayesian_optim_seuil(self):
        '''Fonction pour lancer l'optimisation bayésienne du seuil min'''
        pass
                
                
                







