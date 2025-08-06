import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List



class DataCharger:

    def __init__(self, filepath : str = None, code : list[int] = None, annee: list[int]= None, choice : Optional[int] = None):
        '''Constructeur de la classe DataCharger'''

        self.filepath = filepath  # chemin d'accès pour les données complètes nettoyées
        self.year = annee   # année du filtrage
        self.code = code   # agences du filtrage
        self.choice = choice   # pour que l'utilisateur puisse changer de choix
        self.dataset = None   # Donnée complète
        self.data_agence = None  # Toutes les données d'un certain groupe d'agences 
        self.data_years = None   # Filtrage de self.data_agence sur un certain nombre d'années
        self.data = None   # Données finales considérées pour analyse en aval
        self.grouped = None
        print("N.B. : La liste des années et/ou des codes d'agence peut être modifiée à l'aide de la méthode 'change_agence_year_choice'.")


    def load_csv(self):
        '''Chargement des données complètes (attribuées à self.dataset) et vérification du tri'''

        # Vérification préalables sur le fichier:
        if not hasattr(self, 'filepath') or self.filepath is None:
            raise ValueError("Aucun chemin d'accès de fichier spécifié dans self.filepath")
        if not isinstance(self.filepath, str):
            raise TypeError(f"self.filepath doit forcément être une chaîne de caractère")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Pas de fichier trouvé à l'adresse {self.filepath}")
        # Ouverture du fichier et assignation à self.dataset:
        try:
            self.dataset = pd.read_csv(self.filepath, index_col = 0)
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier CSV: {e}")
        if 'date_heure_operation' not in self.dataset.columns:
            raise ValueError("La colonne 'date_heure_operation' est absente des données")
        # self.dataset = self.dataset.sort_values("date_heure_operation")
        self.dataset.index = self.dataset.index.astype(int)
        self.dataset["date_heure_operation"] = pd.to_datetime(self.dataset["date_heure_operation"]) # On s'assure du type de 'date_heure_operation'
        self.dataset.reset_index(inplace = True)
        self.dataset = self.dataset.sort_values(by = ["code_agence", "jour", "date_heure_operation"])
        self.dataset.set_index('code_agence', inplace = True)
        print("Données complètes chargées avec succès")
        print("Visualisation des données: ")
        print(self.dataset.head(20))  # Vérification visuelle


    def check_data(self, check_dataset : Optional[bool] = True,
                    check_data: Optional[bool] = False, allow_empty: Optional[bool] = False):
        '''Vérification de l'existence des données'''

        if check_dataset:
            if not hasattr(self, 'dataset') or self.dataset is None:
                raise AttributeError("L'attribut 'dataset' est vide")
            if not allow_empty and self.dataset.empty:
                raise ValueError("Les données chargées dans 'dataset' sont vides")
        if check_data:
            if not hasattr(self,'data') or self.data is None:
                raise AttributeError("L'attribut 'data' est vide")
            if not allow_empty and self.data.empty:
                raise ValueError("Les données chargées dans 'data' sont vides")


    def check_missing_columns(self, columns: list[str], source : Optional[str] = 'dataset', raise_error : Optional[bool] = True):
        '''Vérifie que les colonnes en argument sont bien présentes dans le dataset'''

        df = None
        if not isinstance(columns,list) or not all(isinstance(col,str) for col in columns):
            raise TypeError("L'argument 'columns' doit être une liste de chaînes de caractères")
        if source == 'dataset':
            if not hasattr(self,'dataset') or self.dataset is None:
                 raise AttributeError("L'attribut 'dataset' est manquant ou vide")
            df = self.dataset
        elif source == 'data':
            if not hasattr(self,'data') or self.data is None:
                raise AttributeError("L'attribut 'data' est manquant ou vide")
            df = self.data
        else:
            raise ValueError("L'attribut 'source' doit valoir 'dataset' ou 'data'")
        missing = set(columns) - set(df.columns)
        if missing:
            if raise_error:
                raise ValueError(f"Colonnes absentes dans le dataset: {missing}")
            else:
                print(f"Colonnes manquantes dans le dataset considéré: {missing}")
                return missing


    def change_assignation(self, agence : Optional[List[int]] = None, annee : Optional[List[int]] = None, choice : Optional[int] = None):
        '''Pour changer le choix de l'assignation des données'''

        self.check_data(check_data = False)
        if not any([agence, annee, choice]):
            raise ValueError("Au moins l'un des arguments ('agence', 'annee', 'choice') doit être renseigné correctement")
        if agence:
            if not all(isinstance(a, int) for a in (agence if isinstance(agence, list) else [agence])):
                raise TypeError("Tous les éléments de la liste agence doivent être des entiers")
            agence = agence if isinstance(agence, list) else [agence]
            self.code = agence
            print(f"Une (ou plusieurs) nouvelle(s) agence(s) a (ont) été sélectionnée(s) : {agence}")
        if annee:
            if not all(isinstance(y,int) for y in (annee if isinstance(annee,list) else [annee])):
                raise TypeError("Tous les éléments de la liste annee doivent être des entiers")
            annee = annee if isinstance(annee, list) else [annee]
            self.year = annee
            print(f"Une (ou plusieurs) nouvelle(s) années(s) a (ont) été sélectionnée(s) : {annee}")
        if choice is not None:
            if choice not in [1,2,3]:
                raise ValueError("choice doit être un entier compris entre 1 et 3 inclus")
            self.choice = choice
            print(f'Un nouveau choix a été effectué : {choice}')
        self.selection_agence()
        self.selection_annee()
        self.assignation_donnee()
        self.assert_assign_data()
        return self.data


    def nb_agences_annees_dataset(self):
        '''Affiche le nombre d'agences présentes dans les données'''

        self.check_data(check_data = False)
        nb_agences = self.dataset.index.nunique()
        print("Le nombre d'agences présentes dans le dataset complet est de :", nb_agences)
        annee_min = self.dataset["date_heure_operation"].dt.year.min()
        annee_max = self.dataset["date_heure_operation"].dt.year.max()
        print(f"Le dataset va de {annee_min} jusqu'à {annee_max}")


    def liste_annees_agences_data(self):  
        '''Renvoie la liste des agences et des années disponibles dans le dataset complet'''

        self.check_data(check_data = False)
        columns = ["date_heure_operation"]
        self.check_missing_columns(columns)
        try:
            liste_agences = sorted(self.dataset.index.unique().tolist())
            liste_annees = sorted(self.dataset["date_heure_operation"].dt.year.unique().tolist())
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'extraction des années ou agences : {e}")
        print("Liste des agences présentes dans le dataset complet :", liste_agences)
        print("Liste des années présentes dans le dataset complet :", liste_annees)
        return liste_agences, liste_annees


    def selection_agence(self):
        '''Sélection des données pour l'attribut self.data_agence'''

        self.check_data(check_data = False)
        liste_agences = self.liste_annees_agences_data()[0]
        if self.code is None:
            print("Aucun code d'agence(s) spécifié: self.code est vide")
            return   # On termine alors la fonction
        if not isinstance(self.code, list):
            self.code = [self.code]
        code_invalides = [c for c in self.code if c not in liste_agences]
        if code_invalides:
            print(f"Les codes suivants ne sont pas présents dans le dataset: {code_invalides}")
        codes_valides = [c for c in self.code if c in liste_agences]
        if not codes_valides:
            print("Pas de code valide spécifié")
            return 
        self.data_agence = self.dataset.loc[self.dataset.index.isin(codes_valides)].copy()
        print(f"Données chargées pour l'agence {codes_valides}")


    def selection_annee(self):
        '''Sélection des années pour l'attribut self.data_years'''

        self.check_data(check_data = False)
        self.check_missing_columns(["date_heure_operation"], source = 'dataset')
        if not self.year:
            print("Aucune année sélectionnée: self.year est vide")
        if not isinstance(self.year, list):
            self.year = [self.year]
        try:
            self.year = [int(y) for y in self.year]
        except ValueError:
            print("Erreur: self.year doit contenir des entiers représentant des années")
            return   # On termine la fonction dans ce cas
        if not self.code:
            self.data_years = self.dataset.loc[self.dataset["date_heure_operation"].dt.year.isin(self.year)].copy()
            print(f"Données complètes chargées pour l'(les) année(s) {self.year}")
        else:
            if not hasattr(self, 'data_agence') or self.data_agence is None:
                print("Erreur: self.data_agence est vide")
                return
            self.data_years = self.data_agence.loc[self.data_agence["date_heure_operation"].dt.year.isin(self.year)].copy()
            print(f"Données chargées pour l'(les) agence(s) {self.code} pour l'(les) année(s) {self.year}")


    def group_by_agence(self):
        if self.code:
            self.grouped = dict(tuple(self.dataset.groupby("code_agence")))
        print("Le dataset grouped qui groupe les données par agence a bien été créé, et est disponible dans l'argument self.grouped")


    def assignation_donnee(self): 
        '''Méthode pour assigner la donnée voulue à self.data'''

        if self.choice:
            if self.choice == 1:
                self.data = self.dataset
            if self.choice == 2:
                self.data = self.data_agence
            elif self.choice == 3:
                self.data = self.data_years
            else:
                print(f"Choix invalide {self.choice}. On charge le dataset complet dans self.data")
                self.data = self.dataset
        else:
            if self.year:
                self.data = self.data_years
            elif self.code:
                self.data = self.data_agence
            else:
                self.data = self.dataset  # Choix par défaut
        self.data = self.data.reset_index(inplace = True)
        self.data = self.data.sort_values(by = ["code_agence","jour", "date_heure_operation"])
        self.data.set_index("code_agence", inplace = True)  # On s'assure du bon tri
        print("Les données correspondantes ont bien été chargées et triées dans self.data")
        print("N.B.: Le choix des données peut toujours être modifié à l'aide de la méthode 'change_assignation' avec le paramètre other_choice")
        print("other_choice doit alors être donné sous la forme d'une liste du type [agences, annee, choix]")
        return self.data
    

    def assert_assign_data(self):
        '''Vérification de l'assignation des données'''

        if self.code is not None: 
            assert isinstance(self.code, list), "self.code n'est pas spécifié ou pas dans le bon format"
            agences_data = self.data.index.tolist()
            agences_invalides = [agence for agence in agences_data if agence not in self.code]
            if agences_invalides:
                raise ValueError(f"Agences invalides trouvées après filtrage: {agences_invalides}")
        if self.year is not None:
            assert isinstance(self.year, list), "self.year n'est pas spécifié ou pas dans le bon format"
            annees_data = self.data["date_heure_operation"].dt.year.tolist()
            annees_invalides = [annee for annee in annees_data if annee not in self.year]
            if annees_invalides:
                raise ValueError(f"Années invalides trouvées après filtrage: {annees_invalides}")
        print("Les données ont été correctement filtrées selon le choix de l'utilisateur.")


    def visu_data(self, quantité: Optional[int] = 10):
        '''Visualisation des données stockées dans self.data'''

        if not isinstance(quantité,int):
            raise TypeError("Le nombre d'observations spécifié doit être entier")
        if self.data:
            print(self.data.head(quantité))
        else:
            print("Aucune donnée affectée à self.data")


    def visu_dataset(self, quantité: Optional[int] = 10):
        '''Visualisation des données stockées dans self.dataset'''

        if not isinstance(quantité, int):
            raise TypeError("Le nombre d'observations spécifié doit être entier")
        if self.dataset:
            print(self.dataset.head(10))
        else:
            print("Aucune donnée affectée à self.dataset")


    def visu_data_agence(self, quantité: Optional[int] = 10):
        '''Visualisation des données stockées dans self.data_agence'''

        if not isinstance(quantité, int):
            raise TypeError("Le nombre d'observations spécifié doit être entier")
        if self.data_agence:
            print(self.data_agence.head(10))
        else:
            print("Aucune donnée affectée à self.data_agence")


    def visu_data_years(self, quantité: Optional[int] = 10):
        '''Visualisation des données stockées dans self.data_years'''

        if not isinstance(quantité, int):
            raise TypeError("Le nombre d'observations spécifié doit être entier")
        if self.data_years:
            print(self.data_years.head(10))
        else:
            print("Aucune donnée affectée à self.data_years")


    def preparer_donnees(self):
        '''Permet de compiler les principales méthodes de la classe'''

        self.load_csv()
        self.nb_agences_annees_dataset()
        self.selection_agence()
        self.selection_annee()
        self.assignation_donnee()
        self.assert_assign_data()
        return self.dataset, self.assignation_donnee()
        

    def assignation_simple(self):
        '''Simple chargement des données pour passer à la classe suivante'''

        self.load_csv()
        if not any([self.code, self.year]):
            print("Aucune agence ou année spécifiée. Tous les enregistrements seront chargés")
        self.change_assignation(agence = self.code, annee = self.year)
        self.assert_assign_data()
        return self.data