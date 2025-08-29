import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Union
from DataCharger_Class import DataCharger
from BasicStats_Class import BasicStats
from scipy.stats import gaussian_kde
from skopt import gp_minimize
from skopt.utils import OptimizeResult
import os
import traceback
import json

# Pour le moment, on crée donc la donnée suivante, avec pour chaque agence:
# Une colonne 'flux_net' qui contient un dictionnaire type (date, flux_net) pour chaque journée
# Une colonne 'nb_j_ouvres' qui renseigne le nombre de jours ouvrés pour l'agence en question sur l'année dernière
# Une colonne 'code_agence' pour savoir de quelle agence on parle (quand même)
# Une colonne 'freq_pos' pour connaître à quel degré l'agence était créditrice / débitrice l'année passée
# Deux colonnes de 'seuil' pour avoir une base de comparaison
# Une colonne de 'quantiles' qui contient un dictionnaire des valeurs de chacun des quantiles
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


    def __init__(self, filepath : str, filepath_optim : str, optim_csv : str, c_trans : float = 150.0,
                c_rupt : float = 1500.0, t_int : float = 0.02, already_created_optim : Optional[bool] = False):
        '''Constructeur de la classe d'optimisation du seuil min'''

        self.data = None  # Données pour l'optimisation
        self.random_state = None  # Graine pour le bootstrap
        self.threshold_order = None  # Valeur seuil pour la commande dans le modèle
        self.solution = {}   # Dictionnaire vide pour contenir les solutions (par agence)
        self.bootstrap_ratio = 0.85  # On va se baser à la fois sur du bootstrap et sur de l'estimation
        self.already_created_optim = already_created_optim  # Savoir s'il y a besoin de créer la donnée ou non
        self.reorder_values = None  # Liste des valeurs de réapprovisionnement possibles
        self.possible_quantiles = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        # Par défaut, on considère différents quantiles des retraits
        # Une méthode permet à l'utilisateur de changer ces valeurs à sa guise 

        # Vérification des arguments du constructeur: 
        if not all(isinstance(x, float) for x in (c_trans, c_rupt)):
            raise TypeError("Tous les coûts rentrés doivent être des flottants")
        if not isinstance(t_int, float):
            raise TypeError("Le taux d'intérêt (annuel) doit être un entier")
        self.t_int = t_int  # Stockage du taux d'intérêt annuel (fixé par défaut à 2%)
        self.taux_journalier = (1 + self.t_int)**(1/365) - 1  # Stockage du taux d'intérêt journalier
        if not (50 <= c_trans <= 2000):
            raise ValueError("Le coût de transport doit être compris entre 50 et 2000 MAD")
        self.c_trans = c_trans   # Stockage du coût de transport
        if c_rupt <= c_trans:
            raise ValueError("Le coût de rupture doit nécessairement être supérieur au coût de transport")
        self.c_rupt = c_rupt  # Stockage du coût associé à la rupture (avec c_rupt > c_trans)
        self.gap = self.c_trans / self.taux_journalier  # Montant à accumuler pour décharger

        # Vérification des fichiers:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Le fichier '{filepath}' n'existe pas ou est introuvable")
        if not filepath.lower().endswith('.csv'):
            raise ValueError(f"Le fichier '{filepath}' n'est pas un CSV.")
        self.filepath = filepath
        if not filepath_optim.lower().endswith('.json'):
            raise ValueError(f"Le fichier '{filepath_optim}' n'est pas un JSON.")
        self.filepath_optim = filepath_optim
        if not optim_csv.lower().endswith(".json"):
            raise ValueError(f"Le fichier '{optim_csv}' n'est pas un JSON. ")
        self.optim_csv = optim_csv

    
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
        lignes  = {}
        for agence in liste_agences:
            try:
                class_data = DataCharger(filepath = self.filepath, code = agence, annee = year)
                class_data.assignation_simple()
                # print(f"Agence {agence} — class_data type: {type(class_data)}")  # Débogage
                # if hasattr(class_data, 'data'):
                #     print(f"Agence {agence} — class_data.data type: {type(class_data.data)}")
                data_agence = BasicStats(class_data)
                print(f"Agence {agence} — data_agence.data type: {type(data_agence.data)}")
                lignes[str(agence)] = data_agence.data_retrieval_optim()
            except Exception as e:
                print(f"Erreur lors du chargement des données de l'agence {agence} : {e}")
                traceback.print_exc()  # trace complète de l'erreur
                continue
        if not lignes:
            raise ValueError("Aucune donnée n'a pu être traitée")
        self.data = lignes
        return self.data
        
    

    def save_data_optim(self, overwrite: Optional[bool] = False):
        '''Sauvegarde des données créées pour l'optimisation à l'adresse self.filepath_optim'''

        if not self.filepath_optim.lower().endswith('.json'):
            raise ValueError("Le chemin de sauvegarde ne construit pas un fichier au format JSON")
        if self.data is None:
            raise ValueError("Aucune donnée disponible pour une sauvegarde")
        if os.path.exists(self.filepath_optim):
            if overwrite:
                print("Ecrasement du fichier existant...")

            else:
                print("Le fichier existe déjà, il n'y a pas besoin de re-sauvegarder")
                print("Si vous voulez remplacer le fichier existant, relancez la fonction avec l'argument 'overwrite' à True")
                return

        def convert_to_python(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            else:
                return obj
            
        data_python = convert_to_python(self.data)
        with open(self.filepath_optim,"w") as f:
            json.dump(data_python, f, indent = 4)
        print(f"Données sauvegardées vers {self.filepath_optim}")

    
    def load_data_optim(self):
        '''Permet de charger les données nécessaires aux méthodes d'optimisation'''

        # Vérifications sur le chemin d'accès
        if not os.path.exists(self.filepath_optim):
            raise FileNotFoundError("Le fichier de données n'a pas encore été créé")
        if not os.path.isfile(self.filepath_optim):
            raise ValueError("Le chemin d'accès spécifié ne pointe pas vers un fichier")
        if not self.filepath_optim.lower().endswith('.json'):
            raise ValueError("Le fichier n'est pas au format JSON")
        # Chargement des données:
        try:
            with open(self.filepath_optim, "r")as f:
                self.data = json.load(f)
            print(f"Données chargées depuis {self.filepath_optim}")
        except Exception as e:
            raise IOError(f"Erreur lors du chargement du fichier JSON depuis {self.filepath_optim}")
        self.reorder_values = pd.DataFrame ({k:v["quantiles"] for k,v in self.data.items()}).T
        self.reorder_values.columns = self.reorder_values.columns.astype(float)
        print("DEBUG : Premières lignes de self.reorder_values : ", self.reorder_values.head(5))
        # print("=== DEBUG reorder_values ===")
        # print(type(self.reorder_values))
        # print(self.reorder_values.head())     # les 5 premières lignes
        # print("Colonnes :", self.reorder_values.columns 
        # if hasattr(self.reorder_values, "columns") else "Pas de colonnes")
        # print("Index :", self.reorder_values.index[:5]) 

    
    def choice_reorder_value(self, code_agence : int, value: Optional[float] = 0.8,
                        other_choice : Optional[float] = None):
        '''Méthode pour choisir le quantile pour réapprovisionnement'''

        # Other_choice permet de tester une valeur entrée par l'utilisateur, qu'il aurait envie de tester :
        if not other_choice:
            if not isinstance(value, float):
                raise TypeError("Le quantile doit être un float")
            if value not in self.possible_quantiles:
                raise ValueError (f"Le quantile sélectionné doit être dans la liste suivante: {self.possible_quantiles}")
            self.threshold_order = self.reorder_values.loc[str(code_agence), value]  # Permet de sélectionner le quantile voulu pour la commande
        else:
            if not isinstance(other_choice, float):
                raise TypeError("L'argument 'other_choice' doit être un float")
            self.threshold_order = other_choice


    def set_random_seed(self, seed: int):
        '''Permet d'initialiser la graine du générateur pseudo aléatoire'''

        if not hasattr(self,'random_state'):
            raise AttributeError("self n'a pas d'attribut 'random_state'")
        else:
            if not isinstance(seed, int):
                raise TypeError("La graine doit obligatoirement être un entier")
            self.random_state = seed


    def set_reorder_values(self, values : Union[list, np.array]):
        '''Permet de mettre à jour les valeurs de réapprovisionnement'''

        if not hasattr(self, 'reorder_values'):
            raise AttributeError("self n'a aucun attribut 'reorder_values")
        if not (isinstance(values, list) or isinstance(values, np.array)):
            raise TypeError("Les valeurs doivent être sous la forme d'une liste ou d'un tableau numpy")
        self.reorder_values = values

    
    # On considère maintenant que les données sont propres et on code maintenant les fonctions pour
    # évaluer la méthode par agence. Une méthode finale permettra de lancer les calculs pour toutes
    # les agences.


# Méthodes pour estimer de manière non-paramétrique la loi des flux nets (par KDE):

    def flux_kde(self, code_agence : int, factor : Optional[float] = 0.7, 
                     method : Optional[str] = "scott"):
        '''Méthode pour donner une estimation KDE de l'ensemble des flux'''

        flux_dict = self.data[str(code_agence)]["flux_net"]
        flux_net = list(flux_dict.values())
        def bandwidt(s):
            base = s.scotts_factor() if method == "scott" else s.silverman_factor()
            return base * factor
        return gaussian_kde(flux_net, bw_method = bandwidt)


    def deterministic_resample(self, kde : gaussian_kde, n : int):
        '''Méthode pour reproduire l'échantillonnage par KDE'''

        if kde is None:
            return None
        rng = np.random.default_rng(self.random_state)
        return kde.resample(n, seed = rng)


    def generate_kde_scenario(self, code_agence : int, n_iter : Optional[int] = 100,
                              factor : Optional[int] = 0.7,
                              method : Optional[str] = "scott", plot = False):
        '''Permet d'échantillonner des valeurs de flux suivant une estimation KDE de la loi empirique'''

        nb_j_ouvres = self.data[str(code_agence)]["nb_j_ouvrés"]
        rng = np.random.default_rng(self.random_state)
        flux_dict = self.data[str(code_agence)]["flux_net"]
        flux_net = list(flux_dict.values())
        kde_estimate = self.flux_kde(code_agence = code_agence, factor = factor, method = method)
        scenario_kde = np.zeros((n_iter, nb_j_ouvres))
        for i in range(n_iter):
            scenario_kde[i] = self.deterministic_resample(kde_estimate, nb_j_ouvres).flatten()
        if plot:
            sns.histplot(flux_net, bins = 50, stat = "density", color = "yellow", label = "Valeurs réelles du flux")
            x_eval = np.linspace(min(flux_net), max(flux_net), 2000)
            plt.plot(x_eval, kde_estimate(x_eval), color = "m", label = "Estimation KDE")
            plt.legend()
            plt.title(f"Estimation KDE des flux de l'agence {code_agence} en 2024")
            plt.show()
        return scenario_kde

    # On passe ensuite à une estimation par signe des flux: 

    def flux_kde_by_sign(self, code_agence : int, factor_pos : Optional[float] = 0.7,
                         factor_neg : Optional[float] = 0.7,
                         method : Optional[str] = "scott"):
        '''Méthode pour estimer par KDE d'une part la loi des flux positifs, de l'autre celle des flux négatifs'''

        flux_dict = self.data[str(code_agence)]["flux_net"]
        flux_net = list(flux_dict.values())
        flux_post = [value for value in flux_net if value >=0]
        flux_neg = [value for value in flux_net if value < 0]  # Partition des flux en positifs et négatifs
        proba = self.data[str(code_agence)]["freq_pos"]
        def fit_sign(s, factor):
            if len(s) < 2 : 
                return None
            def bandwidt(s):
                base = s.scotts_factor() if method == "scott" else s.silverman_factor()
                return base * factor
            return gaussian_kde(s, bw_method = bandwidt)
        return fit_sign(flux_post, factor_pos), fit_sign(flux_neg, factor_neg), proba
    

    def generate_kde_bysign_scenario(self, code_agence : int, n_iter : Optional[int] = 100, 
                                     factor_pos : Optional[float] = 0.7,
                                     factor_neg : Optional[float] = 0.7,
                                     method : Optional[str] = "scott"):
        '''Méthode pour générer des scénarios après estimation kde par parties'''

        nb_j_ouvres = self.data[str(code_agence)]["nb_j_ouvrés"]
        rng = np.random.default_rng(self.random_state)
        kde_pos, kde_neg, proba = self.flux_kde_by_sign(code_agence = code_agence, factor_pos = factor_pos,
                                                        factor_neg = factor_neg, method = method)
        # On sépare ensuite les flux suivant le signe:
        flux_net_pos = [v for v in list(self.data[str(code_agence), "flux_net"].values()) if v>= 0]
        flux_net_neg = [v for v in list(self.data[str(code_agence), "flux_net"].values()) if v < 0]
        scenario = np.zeros((n_iter, nb_j_ouvres))
        for i in range(n_iter):
            signs = rng.random(nb_j_ouvres) < proba  # On regarde les signes des valeurs générées
            n_pos, n_neg = signs.sum(), (~signs).sum()  # Compte le nombre de valeurs positives et négatives
            if n_pos:
                if kde_pos is not None:
                    scenario[i,signs] = self.deterministic_resample(kde_pos, n_pos).flatten() # On tire des valeurs positives de kde_pos
                else:
                    scenario[i,signs] = rng.choice(flux_net_pos, size = n_pos, replace = True)  # Sinon on choisit des valeurs positives directement depuis flux_net_pos
            if n_neg:
                if kde_neg is not None:
                    scenario[i,~signs] = self.deterministic_resample(kde_neg, n_neg).flatten()
                else:
                    scenario[i,~signs] = rng.choice(flux_net_neg, size = n_neg, replace = True) # Même logique pour les valeurs négatives
        return scenario


    def generate_bootstrap_scenario(self, code_agence : int, n_iter: Optional[int] = 200): # Fonction pour les scénarios bootstrap
        '''Génère un scénario de données pour l'agence spécifiée par bootstrap'''

        nb_j_ouvres = self.data[str(code_agence)]["nb_j_ouvrés"]
        flux_dict = self.data[str(code_agence)]["flux_net"]
        liste_flux = list(flux_dict.values())
        print(f" DEBUG : Liste tronquée des flux pour l'agence {code_agence}: ", liste_flux[:10])
        rng = np.random.default_rng(self.random_state)  # Création d'un générateur pseudo-aléatoire
        liste_scenario = rng.choice(liste_flux, size = (n_iter, nb_j_ouvres), replace = True)
        return liste_scenario
    

    def generate_scenario(self, code_agence : int, n_bootstrap : Optional[int] = 1000,
                          n_kde : Optional[int] = 100, kde_mode : Optional[str] = "single",
                          kde_factor : Optional[float] = 0.7, method : Optional[str] = "scott",
                          return_labels : Optional[bool] = False):
        '''Méthode pour combiner la génération de scénarios bootstrap et par estimation KDE.
        Par défaut, on génère 1000 scénarios bootstrap et 100 scénarios par KDE (pour introduire de la variabilité)'''

        parts, labels = [], []
        if n_bootstrap > 0:
            parts.append(self.generate_bootstrap_scenario(code_agence = code_agence, n_iter = n_bootstrap))
            labels += ["bootstrap"]*n_bootstrap
        if n_kde > 0:
            if kde_mode == 'single':
                kde_scenario = self.generate_kde_scenario(code_agence = code_agence, factor = kde_factor, 
                                             method = method)
            elif kde_mode == "signmix":
                kde_scenario = self.generate_kde_bysign_scenario(code_agence = code_agence, 
                                                                 factor_pos = kde_factor, factor_neg = kde_factor,
                                                                 method = method)
            else:
                raise ValueError("L'argument 'kde_mode' doit valoir soit 'single' soit 'signmix'")
            parts.append(kde_scenario)
            labels += ["kde"]*n_kde
        if not parts:
            raise ValueError("n_bootstrap et n_kde ne peuvent pas être simultanément nuls")
        scenario = np.vstack(parts)
        if return_labels:
            return scenario, labels
        return scenario



# Méthodes pour la simulation des scénarios et l'optimisation des seuils:
    
    def simulate_one_scenario(self, code_agence : int, scenario : np.ndarray, seuil_min : float):
        '''Méthode pour simuler un scénario et calculer les différents coûts'''

        stock = seuil_min
        passages, ruptures, decharges = 0, 0, 0
        c_trans, c_rupt, c_opport = 0.0, 0.0, 0.0
        freq_pos = self.data[str(code_agence)]["freq_pos"]
        for flux in scenario:
            c_opport += 100* self.t_int * freq_pos * seuil_min  # Pénalité sur le seuil min (à investiguer...)
            stock += flux
            if stock < 0:  # En cas de rupture
                c_rupt += self.c_rupt
                stock = seuil_min
                ruptures += 1
            elif 0 <= stock <= self.threshold_order:  # En cas de réapprovisionnement
                c_trans += self.c_trans
                stock = seuil_min
                passages += 1
            elif stock > seuil_min + self.gap:  # En cas d'opportunité suffisamment élevée on décharge
                c_trans += self.c_trans
                stock = seuil_min
                decharges += 1
        return [c_trans, c_rupt, c_opport, passages, ruptures, decharges]
    

    def estimate_smin_MC(self, code_agence : int, 
                         seuil_min : float, n_bootstrap : Optional[int] = 1000,
                         n_kde : Optional[int] = 100, kde_mode : Optional[str] = "single",
                          kde_factor : Optional[float] = 0.7, method : Optional[str] = "scott",
                          return_labels : Optional[bool] = False):
        '''Evaluation du seuil_min par méthode Monte-Carlo (MC)'''

        liste_scenario = self.generate_scenario(code_agence = code_agence, n_bootstrap = n_bootstrap, n_kde = n_kde, kde_mode = kde_mode,
                                                kde_factor = kde_factor, method = method, return_labels = return_labels)
        total_c_trans, total_c_rupt, total_c_opport = 0.0, 0.0, 0.0
        total_rupt, total_decharges, total_passages = 0, 0, 0
        for scenario in liste_scenario:
            liste_elements = self.simulate_one_scenario(code_agence, scenario, seuil_min)
            total_c_trans += liste_elements[0]
            total_c_rupt += liste_elements[1]
            total_c_opport += liste_elements[2]
            total_passages += liste_elements[3]
            total_rupt += liste_elements[4]
            total_decharges += liste_elements[-1]
        total_cost = total_c_trans + total_c_rupt + total_c_opport

        n = liste_scenario.shape[0]
        dict_resultats = {'cost_trans': total_c_trans / n, 'cost_rupt': total_c_rupt / n,
                          'cost_opport': total_c_opport / n,'nb_moy_passages': total_passages / n,
                          'nb_moy_rupt': total_rupt / n, 'nb_moy_decharges': total_decharges /n,
                          'total_cost': total_cost / n}
        return dict_resultats
                

    def bayesian_optim_seuil(self, code_agence : int, n_calls = 30, 
                            max_threshold = 2_000_000, n_bootstrap : Optional[int] = 1000,
                            n_kde : Optional[int] = 100, kde_mode : Optional[str] = "single",
                            kde_factor : Optional[float] = 0.7, method : Optional[str] = "scott",
                            return_labels : Optional[bool] = False): 
        '''Fonction pour réaliser l'optimisation bayésienne du seuil min'''

        # On commence par initialiser les valeurs d'une certaine manière:
        flux_dict = self.data[str(code_agence)]["flux_net"]
        withdrawals = list(flux_dict.values())
        print("DEBUG : withdrawals tronqué : ", withdrawals[:10])
        withdrawals = [w for w in withdrawals if w < 0]
        upper_bound = float(max_threshold)
        lower_bound = float(self.threshold_order) if self.threshold_order is not None else 0.0
        print("DEBUG : valeur de self.threshold_order :", self.threshold_order)

        if len(withdrawals) > 0:
        # Calculer les percentiles
            perc_50 = np.percentile(np.abs(withdrawals), 50)
            perc_75 = np.percentile(np.abs(withdrawals), 75)
            perc_90 = np.percentile(np.abs(withdrawals), 90)
        
        # S'assurer que les points sont dans les bornes
            initial_points = [
                max(lower_bound, min(upper_bound, perc_50)),
                max(lower_bound, min(upper_bound, perc_75)),
                max(lower_bound, min(upper_bound, perc_90))
            ]
        
        # Si tous les percentiles sont hors bornes, utiliser des valeurs par défaut
            if all(p == lower_bound for p in initial_points) or all(p == upper_bound for p in initial_points):
                initial_points = [
                    lower_bound + (upper_bound - lower_bound) * 0.2,
                    lower_bound + (upper_bound - lower_bound) * 0.5,
                    lower_bound + (upper_bound - lower_bound) * 0.8
                ]
        else:
        # Points par défaut dans les bornes
            initial_points = [
                lower_bound + (upper_bound - lower_bound) * 0.2,
                lower_bound + (upper_bound - lower_bound) * 0.5,
                lower_bound + (upper_bound - lower_bound) * 0.8
            ]
    
        print(f"DEBUG : bornes = [{lower_bound}, {upper_bound}]")
        print(f"DEBUG : points initiaux = {initial_points}")
        # Fonction objectif pour l'optimisation bayésienne:
        def objective(params):
            seuil = params[0]
            result = self.estimate_smin_MC(code_agence = code_agence, seuil_min = seuil, n_bootstrap = n_bootstrap, n_kde = n_kde,
                                           kde_mode = kde_mode, kde_factor = kde_factor, method = method, 
                                           return_labels = return_labels)
            return result["total_cost"]
        print("DEBUG : valeur de self.threshold_order :", self.threshold_order)
        resultat = gp_minimize(objective, 
                               dimensions = [(lower_bound, upper_bound)],
                               n_calls = n_calls,
                               x0 = [[x] for  x in initial_points],
                               random_state = self.random_state)
        optimal = {"optim_smin": resultat.x[0],
                   "optim_cost": resultat.fun,
                   "convergence": resultat.func_vals,
                   "evaluations": resultat.x_iters}
        return optimal, resultat



# Méthodes pour évaluer la performance de l'optimisation bayésienne et la dépendance aux paramètres du modèle :
    
    def plot_convergence(self, result : OptimizeResult):
        '''Permet de tester la convergence de l'optimisation bayésienne'''

        # Il faut accéder au résultat de gp_minimize de la fonction précédente
        n_calls = len(result.func_vals)
        func_vals = result.func_vals
        best_so_far = np.minimum.accumulate(func_vals)
        plt.figure(figsize = (10,6))
        plt.plot(range(n_calls), func_vals, "o-", label = "Valeur testée")
        plt.plot(range(n_calls), best_so_far, "r--", label = "Meilleur coût cumulé")
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("Coût cumulé")
        plt.title("Convergence de l'optimisation bayésienne")
        plt.legend()
        plt.grid(True, alpha = 0.5)
        plt.show()


    def explore_smin_quantile(self, code_agence : int, 
                              n_bootstrap : Optional[int] = 1000, n_kde : Optional[int] = 100):
        '''Permet d'étudier la dépendance de s_min à la valeur retenue du quantile'''

        # On stocke les différentes valeurs en fonction des quantiles
        s_min_vals = {k : None for k in self.possible_quantiles}
        cost_vals = {k : None for k in self.possible_quantiles}
        passages_vals = {k : None for k in self.possible_quantiles}
        ruptures_vals = {k : None for k in self.possible_quantiles}
        decharges_vals = {k : None for k in self.possible_quantiles}
        for quantile in self.possible_quantiles:
            self.choice_reorder_value(code_agence = code_agence, value = quantile)
            s_min = self.bayesian_optim_seuil(code_agence = code_agence, n_bootstrap = n_bootstrap, n_kde = n_kde)[0]["optim_smin"]
            s_min_vals[quantile] = s_min
            eval = self.estimate_smin_MC(code_agence, s_min, n_bootstrap = n_bootstrap, n_kde = n_kde)
            cost_vals[quantile] = eval["total_cost"]
            passages_vals[quantile] = eval["nb_moy_passages"]
            ruptures_vals[quantile] = eval["nb_moy_rupt"]
            decharges_vals[quantile] = eval["nb_moy_decharges"]

        x_vals = list(self.possible_quantiles)
        s_min_list = [s_min_vals[q] for q in x_vals]
        cost_list = [cost_vals[q] for q in x_vals]
        passages_list = [passages_vals[q] for q in x_vals]
        ruptures_list = [ruptures_vals[q] for q in x_vals]
        decharges_list = [decharges_vals[q] for q in x_vals]
        # Création des plots : 
        fig,axs = plt.subplots(2, 2, figsize = (12,8))
        axs[0,0].plot(x_vals, s_min_list, marker = 'o', color = 'blue')
        axs[0,0].set_title("Seuil min en fonction du quantile de réapprovisionnement")
        axs[0,0].set_xlabel("Quantile")
        axs[0,0].set_ylabel("Valeur de s_min")

        axs[0,1].plot(x_vals, cost_list, marker = 'o', color = 'red')
        axs[0,1].set_title("Coût total annuel moyen en fonction du quantile de réapprovisionnement")
        axs[0,1].set_xlabel("Quantile")
        axs[0,1].set_ylabel("Valeur du coût total annuel moyen")

        axs[1,0].plot(x_vals, passages_list, marker = 'o', color = 'green')
        axs[1,0].set_title("Nombre moyen de passages par an en fonction du quantile de réapprovisionnement")
        axs[1,0].set_xlabel("Quantile")
        axs[1,0].set_ylabel("Moyenne du nombre de passages annuels")

        axs[1,1].plot(x_vals, ruptures_list, marker = 'o', color = 'purple')
        axs[1,1].set_title("Nombre moyen de ruptures par an en fonction du quantile de réapprovisionnement")
        axs[1,1].set_xlabel("Quantile")
        axs[1,1].set_ylabel("Moyenne du nombre de ruptures annuelles")
        plt.show()

    
    def dependency_on_seed(self, code_agence : int, quantile : int, n_seeds : Optional[int] = 15, 
                           n_calls : Optional[int] = 30, n_bootstrap : Optional[int] = 1000, n_kde : Optional[int] = 100,
                           seed_master : Optional[int ]= 1234, plot : Optional[bool] = True):
        '''Méthode pour étudier l'influence de la graine sur la valeur retournée par l'optimisation'''

        self.choice_reorder_value(code_agence = code_agence, value = quantile)
        rng = np.random.default_rng(seed_master)
        seeds = rng.integer(low = 0, high = 1_000, size = n_seeds).tolist()  # Génère des seeds
        results = []
        init_seed = self.random_state  # On stocke la valeur initiale pour la réinitialiser à la fin
        print("Valeur initiale de seed: ", self.random_state)
        for seed in seeds:
            self.set_random_seed(seed)
            optimal, _ = self.bayesian_optim_seuil(code_agence = code_agence, n_calls = n_calls,
                                                   n_bootstrap = n_bootstrap, n_kde = n_kde)
            results.append({"seed": seed, "s_min": optimal["optim_smin"],
                            "total_cost": optimal["optim_cost"]})
        df_results = pd.DataFrame(results)
        if plot:
            fig, axs = plt.subplots(1,2, figsize = (12,5))
            axs[0].scatter(df_results["seed"], df_results["s_min"], c = "blue", label = "Seuil optimal")
            axs[0].set_title(f"Seuil Optimal en fonction des valeurs de graine ({seeds}) testées pour le PRNG")
            axs[0].set_xlabel("Valeur de la graine")
            axs[0].set_ylabel("Valeur du seuil associé")
            axs[0].legend()

            axs[1].scatter(df_results["seed"], df_results["total_cost"], c = "red", label = "Coût associé au seuil")
            axs[1].set_title(f"Coût optimal en fonction des valeurs de graine ({seeds}) testées pour le PRNG")
            axs[1].set_xlabel("Valeur de la graine")
            axs[1].set_ylabel("Valeur du coût optimal associé")
            axs[1].legend()
            plt.show()
        self.set_random_seed(init_seed)  # Permet de remettre la valeur initiale de la graine
        print("Valeur finale de la graine: ", self.random_state)  # Permet de débugguer au cas où
        return df_results, seeds
    


# Méthodes pour charger les données, lancer l'optimisation et sauvegarder les résultats:

    def optim_one_agency(self, code_agence : int, n_calls : Optional[int] = 30, n_bootstrap : Optional[int] = 1000,
                         n_kde : Optional[int] = 100, kde_mode : Optional[str] = "single",
                         kde_factor : Optional[float] = 0.7, method : Optional[str] = "scott",
                         return_labels : Optional[bool] = False): 
        '''Permet de lancer l'optimisation pour une seule agence (donc d'obtenir une analyse plus fine)'''
        
        if not self.already_created_optim:
            self.remplissage_data_optim()
            self.save_data_optim()
        self.load_data_optim()
        self.explore_smin_quantile(code_agence = code_agence,
                                   n_bootstrap = n_bootstrap, n_kde = n_kde)
        for q in self.possible_quantiles:
            self.choice_reorder_value(code_agence = code_agence, value = q)
            _ , result = self.bayesian_optim_seuil(code_agence = code_agence, n_calls = n_calls,
                                                      n_bootstrap = n_bootstrap, n_kde = n_kde,
                                                       kde_mode = kde_mode, kde_factor = kde_factor, method = method)
            self.dependency_on_seed(code_agence = code_agence, quantile = q, n_bootstrap = n_bootstrap, n_kde = n_kde)
            self.plot_convergence(result)


    def optim_all_agencies(self, n_calls : Optional[int] = 30, n_bootstrap : Optional[int] = 1000,
                           n_kde : Optional[int] = 100, kde_mode : Optional[str] = "single",
                           kde_factor : Optional[float] = 0.7, method : Optional[str] = "scott",
                           return_labels : Optional[bool] = False):
        '''Permet de lancer l'optimisation globale pour toutes les agences et de récupérer les valeurs de seuil
        En pratique on crée un CSV compact plus facile à manipuler pour un dashboard et un second CSV, plus éclaté
        pour une vérification visuelle'''

        if not self.already_created_optim:
            self.remplissage_data_optim()
            self.save_data_optim()
        self.load_data_optim()
        list_result_optim_compact = []
        list_result_optim_flat = []
        for code_agence in self.data.keys():
            dict_result_optim_compact, dict_result_optim_flat = {}, {}
            dict_result_optim_compact["code_agence"] = code_agence
            dict_result_optim_flat["code_agence"] = code_agence
            for q in self.possible_quantiles:
                self.choice_reorder_value(code_agence = code_agence, value = q)
                optimal,_ = self.bayesian_optim_seuil(code_agence = code_agence, n_calls = n_calls,
                                                      n_bootstrap = n_bootstrap, n_kde = n_kde,
                                                       kde_mode = kde_mode, kde_factor = kde_factor, method = method)
                donnees_s_min = self.estimate_smin_MC(code_agence = code_agence, seuil_min = optimal["optim_smin"],
                                                      n_bootstrap = n_bootstrap, n_kde = n_kde, kde_mode = kde_mode,
                                                      kde_factor = kde_factor, method = method, return_labels = return_labels)
                dict_result_quantile = {"optim_smin": optimal['optim_smin'], "optim_total_cost" : optimal['optim_cost'],
                                        "cost_trans": donnees_s_min['cost_trans'], "cost_rupt": donnees_s_min['cost_rupt'],
                                        'cost_opport': donnees_s_min['cost_opport'], "nb_moy_passages": donnees_s_min['nb_moy_passages'],
                                        "nb_moy_ruptures": donnees_s_min['nb_moy_rupt'], "nb_moy_decharges": donnees_s_min['nb_moy_decharges']}
                dict_result_optim_flat[f"optim_smin_q{q}"] = optimal['optim_smin']
                dict_result_optim_flat[f"optim_total_cost_q{q}"] = optimal['optim_cost']
                dict_result_optim_flat[f"cost_trans_q{q}"] = donnees_s_min['cost_trans']
                dict_result_optim_flat[f"cost_rupt_q{q}"] = donnees_s_min['cost_rupt']
                dict_result_optim_flat[f"cost_opport_q{q}"] = donnees_s_min['cost_opport']
                dict_result_optim_flat[f"nb_moy_passages_q{q}"] = donnees_s_min['nb_moy_passages']
                dict_result_optim_flat[f"nb_moy_ruptures_q{q}"] = donnees_s_min['nb_moy_rupt']
                dict_result_optim_flat[f"nb_moy_decharges_q{q}"] = donnees_s_min['nb_moy_decharges']
                dict_result_optim_compact[f'quantile {q}'] = dict_result_quantile
            list_result_optim_compact.append(dict_result_optim_compact)
            list_result_optim_flat.append(dict_result_optim_flat)
        results_optim_compact = pd.DataFrame(list_result_optim_compact)
        results_optim_compact.set_index("code_agence")
        results_optim_flat = pd.DataFrame(list_result_optim_flat)
        results_optim_flat.set_index("code_agence")
        # Vérifications sur le chemin d'accès de la sauvegarde:
        if not isinstance(self.optim_csv, str):
            raise TypeError("Le chemin d'accès spécifié doit être spécifié sous la forme d'une chaîne de caractères")
        if not self.optim_csv.endswith('.csv'):
            raise ValueError("Le chemin d'accès passé en argument n'est pas au format csv")
        base, ext = os.path.splitext(self.optim_csv)
        results_optim_compact.to_csv(f"{base}_optim_compact{ext}", index = True)
        results_optim_flat.to_csv(f"{base}_optim_flat{ext}", index = True)
        return results_optim_compact, results_optim_flat


# - Il faut rajouter une fonction pour lancer l'optimisation (en fait plusieurs) (en cours)
# - Enfin, il faudra créer une autre classe (type dashboard) pour résumer les analyses et les valeurs de seuils               


# Il manquera le nombre de décharges dans la fonction d'évaluation en fonction des quantiles
# Il faudrait aussi une fonction qui évalue le seuil obtenu en fonction de la valeur de la pénalité 
# sur le seuil min (il faut qu'en ordre de grandeur il soit du même ordre que les autres coûts).
# Enfin, on pourrait rajouter un Q-Q plot pour comparer l'estimation kde aux flux réels.

# On pourrait rajouter une fonction qui montre comment varie la valeur de seuil en fonction de la valeur de la pénalité




