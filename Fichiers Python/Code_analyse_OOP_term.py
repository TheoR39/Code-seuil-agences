import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# from datetime import datetime
from typing import Optional, List


# Classe de preprocessing de la data brute: 

class PreprocessingRawData:

    def __init__(self, filepath : Optional[List[str]] = None, newfilepath : Optional[str] = None, aggregate: Optional[bool] = False):
        '''Initialisation des paramètres et vérification des types des variables passées en argument du constructeur'''

        # Vérification des types des variables:
        if filepath is not None:
            if not isinstance(filepath,list) or not all(isinstance(fp,str) for fp in filepath):
                raise ValueError("Le paramètre 'filepath' fourni n'est pas correct")
        if filepath is None:
            raise ValueError("Le paramètre 'filepath' est manquant")
        for fp in filepath:
            if not os.path.isfile(fp):
                print(f"Avertissement: le fichier {fp} n'existe pas.")
        if newfilepath is not None and not isinstance(newfilepath, str):
            raise ValueError("Le paramètre 'newfilepath' n'est pas correct")
        if not isinstance(aggregate,bool):
            raise ValueError("Le paramètre 'aggregate' n'est pas correct")
        # Initialisation des attributs:
        self.filepath = filepath         
        self.newfilepath = newfilepath
        self.data = None
        self.aggregate = aggregate  # Savoir s'il faut nettoyer ou pas (à vérifier dans le main)


    def load_raw_csv(self): # A compléter...
        '''Chargement des données brutes à l'aide d'une liste de chemin d'accès et concaténation du résultat'''

        if self.filepath is None:
            raise ValueError("Aucun chemin de fichier fourni...")
        elif isinstance(self.filepath, str):
            filepaths = [self.filepath]
        else:
            filepaths = self.filepath
        list_dfs = []
        for filepath in filepaths:
            df = pd.read_csv(filepath, index_col = "code_agence")
            df = df.sort_index()
            list_dfs.append(df)
            print(f"Données chargées depuis {filepath}")
        self.data = pd.concat(list_dfs, axis = 0)
        self.data = self.data.sort_index()
        print(f"Concaténation terminée, pour un nombre total d'observations de {self.data.shape[0]}")
        print("Visualisation des données après concaténation :")
        print(self.data.head(10))

    def check_not_None(self):
        '''Fonction pour vérifier que les données sont bien chargées'''

        if self.data is None:
            raise ValueError("self.data doit d'abord être initialisé")
        
    
    def check_missing_columns(self, columns: list[str]):
        '''Vérifie que les colonnes en argument sont bien présentes dans le dataset self.data'''

        if not isinstance(columns,list) or not all(isinstance(col,str) for col in columns):
            raise TypeError("L'argument 'columns' doit être une liste de chaînes de caractères")
        if not hasattr(self,'data') or self.data is None:
            raise AttributeError("L'attribut 'dataset' est manquant ou vide")
        missing = set(columns) - set(self.data.columns)
        if missing:
            raise ValueError(f"Les colonnes suivantes sont absentes de self.data: {missing}")


    def describe_data(self):
        '''Simple description des données avec .describe()'''

        self.check_not_None()
        print("Description générale des données :")
        print(self.data.describe())


    def info_data(self):
        '''Simple description des données avec .info()'''

        self.check_not_None()
        print("Infos importantes sur les données :")
        print(self.data.info()) 


    def visu_data(self, quant : Optional[int] = 8):
        '''Visualisation des données avec .head()'''

        self.check_not_None()
        if not isinstance(quant, int) or quant <=0:
            raise ValueError("La quantité demandée n'est pas entière ou négative")
        print("Visualisation des données :")
        print(self.data.head(quant))


    def remove_duplicates(self):
        '''Retrait des doublons présents dans le dataset contenu dans self.data avec .drop_duplicates()'''

        self.check_not_None()
        print("Nombre d'observations avec doublons :", self.data.shape[0])
        self.data.drop_duplicates(keep = 'first', inplace = True)
        print("Nombre d'observations après nettoyage des doublons :", self.data.shape[0])
        # Vérification:
        nb_restants = self.data.duplicated().sum()
        if nb_restants != 0:
            print(f"Il reste encore {nb_restants} de doublons")
        print("Tous les doublons ont été éliminés")


    def time_format(self):
        '''Création de la colonne 'date_heure_operation' en Datetime (pour filtrer sur les dates)'''

        # On retire les observations manquantes:
        print("Version modifiée de time_format exécutée")  # Pour vérification...
        self.check_not_None()
        nb_avant = self.data.shape[0]
        self.data = self.data.dropna(subset = ["date_operation"])
        nb_apres = self.data.shape[0]
        print(f"Suppression de {nb_avant - nb_apres} observations à cause de dates manquantes")
        # Vérification sur les colonnes:
        columns = ['date_operation', 'heure_operation']
        self.check_missing_columns(columns)
        # Conversion et vérification:
        self.data["date_operation"]=  pd.to_datetime(self.data["date_operation"]).dt.normalize()

        def format_heure(h):  # Pour gérer les données factices
            h_str = str(h)
            if ":" in h_str:
                return h_str
            h_str = h_str.zfill(6)
            return f"{h_str[:2]}:{h_str[2:4]}:{h_str[4:]}"
        self.data["heure_operation"] = self.data["heure_operation"].apply(format_heure)
        self.data["heure_operation"] = pd.to_timedelta(self.data["heure_operation"].astype('str'))
        self.data["date_heure_operation"] = self.data["heure_operation"] + self.data["date_operation"]
        if not pd.api.types.is_datetime64_any_dtype(self.data["date_heure_operation"]):
            raise TypeError("'date_heure_operation n'est pas un datetime")
        else:
            print("La colonne 'date_heure_operation' est bien de type datetime")
        # Trie sur la date et visualisation:
        self.data = self.data.sort_values("date_heure_operation")
        print("Visualisation des données après reformatage des dates (regarder la nouvelle colonne 'date_heure_operation'): ")
        print(self.data.head())


    def algebrisation_montants(self):
        '''Algébrise les montants des opérations: + pour un dépôt et - pour un retrait'''
        
        # Vérifications préliminaires:
        self.check_not_None()
        columns = ["montant_operation", "sens_operation"]
        self.check_missing_columns(columns)
        # Algébrisation des montants:
        self.data.loc[self.data["sens_operation"] == 'D', "montant_operation"] *= -1
        # Abandon de la colonne 'sens_operation' devenue obsolète:
        self.data = self.data.drop(columns = ["sens_operation"])
        # Visualisation après transformation:
        print("Visualisation des données après algébrisation des montants (regarder la colonne montant_operation): ")
        print(self.data.head())


    def check_currency(self):
        '''Retourne les observations (et les dates correspondantes) dont la devise n'est pas en MAD'''
        
        # Vérifications préliminaires:
        self.check_not_None()
        columns = ["devise"]
        self.check_missing_columns(columns)
        # Décompte du nombre d'opérations incorrectes:
        print(self.data["devise"].value_counts(dropna=False))
        if not (self.data["devise"] != 'MAD').all():
            print("Toutes les observations sont bien en MAD")
            return False
        else:
            # Print à l'utilisateur les opérations problématiques et leurs dates
            obs_non_MAD = self.data[self.data["devise"] != 'MAD']
            print("N.B.: S'il y a des observations dont la devise n'est pas en MAD, il faut trouver le taux de change de la journée")
            print("Il faut ensuite convertir à l'aide de la méthode 'change_currency'")
            nb_to_change = obs_non_MAD.shape[0]
            liste_devises = obs_non_MAD["devise"].tolist()
            liste_date = obs_non_MAD["date_heure_operation"].tolist()
            print("Nombre d'observations à changer :", nb_to_change)
            print("Liste des dates des observations problématiques :", liste_date)
            print("Liste des devises des observations problématiques :", liste_devises)
            return True

    def change_currency(self, taux):  # Potentiellement à changer (si trop d'observations problématiques)
        '''Changement (à la main) des observations problématiques'''

        self.check_not_None()
        obs_non_MAD = self.data[self.data["devise"] != 'MAD']
        nb_to_change = obs_non_MAD.shape[0]
        liste_devises = obs_non_MAD["devise"].tolist()
        liste_date = pd.to_datetime(obs_non_MAD["date_heure_operation"]).tolist()
        if len(taux) != nb_to_change:
            raise ValueError(f"Le nombre de taux ({len(taux)}) ne correspond pas au nombre d'observations à changer ({nb_to_change})")
        for i in range(nb_to_change):
            mask = (self.data["devise"] == liste_devises[i]) & (self.data["date_heure_operation"] == liste_date[i])
            self.data.loc[mask, "montant_operation"] *= taux[i]
            self.data.loc[mask, "devise"] = "MAD"
        print("Conversion des devises effectuée")
        print("Vérification d'autres devises restantes :", (self.data["devise"] != 'MAD').any())


    def filtre_etat_origine_operation(self):
        '''Filtre sur les opérations en agence (OA) et sur les opérations terminées (2)'''

        # Vérifications preliminaires:
        self.check_not_None()
        columns = ["etat_operation", "application_origine_operation"]
        self.check_missing_columns(columns)
        print("Nombre d'observations avant filtrage: ", self.data.shape[0])
        # Filtrage sur 'etat_origine_operation':
        self.data = self.data[self.data["etat_operation"] == 2]
        print("Nombre d'observations après filtrage sur la colonne 'etat_operation' :", self.data.shape[0])
        # Filtrage sur 'etat_origine_operation':
        self.data = self.data[self.data["application_origine_operation"] == 'OA']
        print("Nombre d'observations après filtrage sur la colonne 'application_origine_operation' :", self.data.shape[0])


    def filtre_type_operation(self):
        '''Filtrage sur le type d'opération'''

        # Vérifications préliminaires:
        self.check_not_None()
        columns = ["libelle_long_operation"]
        self.check_missing_columns(columns)
        # Filtrage sur le type d'opération:
        vals_libelle = self.data["libelle_long_operation"].dropna().unique()
        categ_retraits = [valeur for valeur in vals_libelle if str(valeur).startswith("RETRAIT")]
        categ_versements = [valeur for valeur in vals_libelle if str(valeur).startswith("VERSEMENT")]
        categ_autres_especes = [valeur for valeur in vals_libelle if str(valeur).endswith("ESPECE")]
        categ_conservees = categ_retraits + categ_versements + categ_autres_especes
        self.data = self.data[self.data["libelle_long_operation"].isin(categ_conservees)]
        print("Liste des catégories d'opérations retenues :", categ_conservees)
        print("Nombre d'observations restantes :", self.data.shape[0])

    def remove_columns(self):
        '''Abandon des colonnes inutilisées'''

        # On vérifie d'abord que toutes les colonnes spécifiées sont bien présentes dans les données:
        self.check_not_None()
        columns = ['identifiant_compte', 'reference_operation', 'code_marche', 'etat_operation', 
        'code_famille_operation', 'code_type_operation', 'application_origine_operation', 'motif_operation', 'devise', 'numero_caisse',
        'heure_operation', 'date_operation', 'code_banque', 'date_valeur'] 
        self.check_missing_columns(columns)
        # Abandon des colonnes inutiles:
        print("Nombre de colonnes avant traitement :", self.data.shape[1])
        self.data = self.data.drop(columns = columns, errors = 'ignore')
        print("Nombre de colonnes retenues :", self.data.shape[1])
        print("Informations après nettoyage des colonnes inutiles :")
        print(self.data.info())


    def save_cleaned_data(self, newfilepath : str):
        '''Sauvegarde des données nettoyées dans newfilepath'''

        self.check_not_None()
        if self.aggregate and os.path.exists(newfilepath):
            complete = pd.read_csv(filepath = newfilepath, index_col = 0)
            to_save = pd.concat([complete,self.data], ignore_index = False)
        else:
            to_save = self.data
        to_save = to_save.sort_index()
        to_save = to_save.sort_values("date_heure_operation")
        to_save.to_csv(newfilepath, index = True, encoding = 'utf-8')
        print("Données nettoyées enregistrées au format csv")


    def preprocessing(self):
        '''Agglomère les différentes fonctions pour un preprocessing global (et complet)'''

        self.load_raw_csv()
        self.describe_data()
        self.info_data()
        self.visu_data()
        self.remove_duplicates()
        self.time_format()
        self.algebrisation_montants()
        self.filtre_etat_origine_operation()
        self.filtre_type_operation()
        if self.check_currency():
            saisie = input("Insérez ici la liste des valeurs des taux de change appropriés pour les dates et les devises données, séparés par une virgule :")
            taux =  [float(elem.strip()) for elem in saisie.split(',')]
            self.change_currency(taux)
        self.remove_columns()
        self.visu_data()
        if not self.newfilepath:
            raise ValueError("Aucun chemin de sauvegarde spécifié")
        self.save_cleaned_data(self.newfilepath)









# Classe de chargement des données utiles à l'analyse:

class DataCharger:

    def __init__(self, filepath : str = None, code : list[int] = None, annee: list[int]= None, choice : Optional[int] = None):
        '''Constructeur de la classe DataCharger'''

        # Pour info, on met choice par défaut = 1 si rien n'est spécifié
        # Cela permet de laisser passer l'ensemble du dataset, sans sélection préalable.
        self.filepath = filepath
        self.year = annee
        self.code = code
        self.choice = choice
        self.dataset = None
        self.data_agence = None
        self.data_years = None
        self.data = None
        self.grouped = None
        print("N.B. : La liste des années et/ou des codes d'agence peut être modifiée à l'aide de la méthode 'change_agence_year_choice'.")


    def load_csv(self):
        '''Chargement des données complètes (attribuées à self.dataset)'''

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
        self.dataset = self.dataset.sort_index()
        self.dataset = self.dataset.sort_values("date_heure_operation")
        self.dataset.index = self.dataset.index.astype(int)
        print("Données complètes chargées avec succès")
        print("Visualisation des données: ")
        print(self.dataset.head(10))


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


    def verif_vides(self):
        '''Enlève les observations avec des valeurs vides'''

        self.check_data(check_data = False)
        for column in self.dataset.columns:
            if self.dataset[column].isna().any():
                print(f"La colonne {column} contient des valeurs manquantes")
                self.dataset.dropna(subset = [column], inplace = True)
            else:
                print(f'Aucune valeur manquante dans la colonne {column}')


    def verif_encodage(self): 
        '''Vérification de l'encodage des colonnes importantes (en particulier pour les dates)'''
        
        # On commence par vérifier que les colonnes sont bien présentes dans le dataset:
        self.check_data(check_data = False)
        columns = ["date_heure_operation", "libelle_long_operation",
                         "libelle_court_operation", "identifiant_operation"]
        self.check_missing_columns(columns)
        try:
            self.dataset["date_heure_operation"] = pd.to_datetime(self.dataset["date_heure_operation"])
            self.dataset.index = self.dataset.index.astype(int)
            self.dataset["libelle_long_operation"] = self.dataset["libelle_long_operation"].astype('string')
            self.dataset["libelle_court_operation"] = self.dataset["libelle_court_operation"].astype('string')
            self.dataset["identifiant_operation"] = self.dataset["identifiant_operation"].astype('string')
        except Exception as e:
            raise RuntimeError(f"Erreur lors du reformatage: {e}")
        print("Encodage réalisé avec succès")
        print(self.dataset.info())


    def completion_data(self):  
        '''Ajout de colonnes nécessaires au traitement des données (notamment les flux)'''

        self.check_data(check_data = False)
        columns = ["date_heure_operation", "montant_operation"]
        self.check_missing_columns(columns)
        self.dataset["jour"] = self.dataset["date_heure_operation"].dt.date
        self.dataset["crédit"] = self.dataset["montant_operation"].apply(lambda x: x if x>0 else 0)
        self.dataset["débit"] = self.dataset["montant_operation"].apply(lambda x: -x if x<0 else 0)
        self.dataset["flux_net"] = self.dataset.groupby("jour")["montant_operation"].cumsum()


    def change_assignation(self, agence : Optional[List[int]] = None, annee : Optional[List[int]] = None, choice : Optional[int] = None):
        '''Pour changer le choix de l'assignation des données'''

        self.check_data(check_data = False)
        if not any([agence, annee, choice]):
            raise ValueError("Au moins l'un des arguments ('agence', 'annee', 'agence') doit être renseigné correctement")
        if agence:
            agence = agence if isinstance(agence, list) else [agence]
            self.code = agence
            print(f"Une (ou plusieurs) nouvelle(s) agence(s) a (ont) été sélectionnée(s) : {agence}")
        if annee:
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
            liste_agences = self.dataset.index.unique().tolist()
            liste_annees = self.dataset["date_heure_operation"].dt.year.unique().tolist()
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'extraction des années ou agences : {e}")
        print("Liste des agences présentes dans le dataset complet :", liste_agences)
        print("Liste des années présentes dans le dataset complet :", liste_annees)
        return liste_agences, liste_annees


    def selection_agence(self):

        self.check_data(check_data = False)
        if self.code:
            self.code = self.code if isinstance(self.code,list) else [self.code]
            self.data_agence = self.dataset.loc[self.dataset.index.isin(self.code)].copy()
            print(f"Données chargées pour l'agence {self.code}")
        else:
            print("Aucun code spécifié ou pas dans le bon format...")

    def selection_annee(self):
        if self.year:
            self.year = self.year if isinstance(self.year,list) else [self.year]
            if not self.code:
                self.data_years = self.dataset.loc[self.dataset["date_heure_operation"].dt.year.isin(self.year)].copy()
                print(f"Données complètes chargées pour l'(les) année(s) {self.year}")
            else:
                self.data_years = self.data_agence.loc[self.data_agence["date_heure_operation"].dt.year.isin(self.year)].copy()
                print(f"Données chargées pour l'(les) agence(s) {self.code} pour l'(les) année(s) {self.year}")
        else:
            print("Aucune année sélectionnée ou pas dans le bon format...")


    def group_by_agence(self):
        if self.code:
            self.grouped = dict(tuple(self.dataset.groupby("code_agence")))
        print("Le dataset grouped qui groupe les données par agence a bien été créé, et est disponible dans l'argument self.grouped")

    def assignation_donnee(self):  # A modifier pour prendre en compte le fait qu'on peut ne renvoyer que le dataset complet
        # Il ne faut filtrer sur self.choice qu'après avoir filtré sur l'agence
        # On considère self.choice comme un choix par défaut (ou pas, on peut faire les deux)
        if self.choice:
            if self.choice == 1:
                self.data = self.dataset
            if self.choice == 2:
                self.data = self.data_agence
            elif self.choice == 3:
                self.data = self.data_years
        else:
            if self.year:
                self.data = self.data_years
            elif self.code:
                self.data = self.data_agence
            else:
                self.data = self.dataset  # En fait si, on prend bien self.dataset par défaut ici...
        self.data = self.data.sort_index()
        self.data = self.data.sort_values("date_heure_operation")
        print("Les données correspondantes ont bien été chargées et triées dans self.data")
        print("N.B.: Le choix des données peut toujours être modifié à l'aide de la méthode 'change_assignation' avec le paramètre other_choice")
        print("other_choice doit alors être donné sous la forme d'une liste du type [agences, annee, choix]")
        return self.data
    

    def assert_assign_data(self):
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

    def visu_data(self):
        if self.data:
            print(self.data.head(10))
        else:
            print("Aucune donnée affectée à self.data")

    def visu_dataset(self):
        if self.dataset:
            print(self.dataset.head(10))
        else:
            print("Aucune donnée affectée à self.dataset")

    def visu_data_agence(self):
        if self.data_agence:
            print(self.data_agence.head(10))
        else:
            print("Aucune donnée affectée à self.data_agence")

    def visu_data_years(self):
        if self.data_years:
            print(self.data_years.head(10))
        else:
            print("Aucune donnée affectée à self.data_years")


    def preparer_donnees(self):
        '''Permet de compiler les principales méthodes de la classe'''

        self.load_csv()
        self.verif_vides()
        self.verif_encodage()
        self.completion_data()
        self.nb_agences_annees_dataset()
        self.selection_agence()
        self.selection_annee()
        self.assignation_donnee()
        self.assert_assign_data()
        return self.dataset, self.assignation_donnee()
        

    def assignation_simple(self):
        '''Simple chargement des données pour passer à la classe suivante'''

        self.load_csv()
        self.verif_vides()
        self.verif_encodage()
        self.completion_data()
        self.change_assignation(agence = self.code, annee = self.year)
        self.assert_assign_data()
        return self.data







# Classe d'analyse des données, basée sur statistiques descriptives et visualisations: 

class BasicStats:  

    def __init__(self, class_data : DataCharger):
        self.object = class_data  
        self.mois_possibles = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
        self.month = None
        self.saisons = ["Printemps", "Eté", "Automne", "Hiver"]
        self.semestres = ["1er semestre", "2nd semestre"]
        # self.semaines = [f"Semaine {i}" for i in range(1,self.data["date_heure_operation"].dt.isocalendar().week.max() + 1)]
        self.trimestres = ["1er trimestre", "2ème trimestre", "3ème semestre", "4ème semestre"]
        self.type_periode = {"month": self.mois_possibles, "season": self.saisons,
                             "semester": self.semestres, "quarter": self.trimestres}
 
    @property
    def data(self):
        return self.object.data

    @property
    def agence(self):
        agences = self.data.index.unique()
        assert len(agences) == 1, f"Le jeu de données contient plusieurs agences au lieu d'en contenir une seule {agences}"
        return self.data.index[0]

    @property
    def year(self):
        annees = self.data["date_heure_operation"].dt.year.unique()
        assert len(annees) == 1, f"Le jeu de données contient plusieurs années au lieu d'en contenir une seule {annees}"
        return annees[0]
    
    def semaines_data(self):
        if self.data is None:
            raise ValueError("Il faut d'abord commencer par initialiser self.data")
        else:
            self.semaines = [f"Semaine {i}" for i in range(1,self.data["date_heure_operation"].dt.isocalendar().week.max() + 1)]
            self.type_periode["week"] = self.semaines
        # Pb: le calcul des semaines s'appuie sur self.data



# Méthodes pour le calcul des statistiques descriptives (retraits / versements / flux):

    def nb_obs_jour(self):
        nb_jours_ouvres = self.data["jour"].nunique()
        print(f"Nombre de jours ouvrés dans l'année {self.year} :", nb_jours_ouvres)
        nb_obs_jour = self.data.groupby("jour")["montant_operation"].count()
        nb_retraits_jours = self.data[self.data["débit"] != 0].groupby("jour")["débit"].count()
        nb_versements_jours = self.data[self.data["crédit"] != 0].groupby("jour")["crédit"].count()
        nombre = {"j_ouvres": nb_jours_ouvres, "obs_j" : nb_obs_jour, "retraits_j" : nb_retraits_jours, "versements_j": nb_versements_jours}
        return nombre

    def vals_seuil_nb_obs(self):
        data = self.nb_obs_jour()
        min_obs_jour = data["obs_j"].min()
        max_obs_jour = data["obs_j"].max()
        moy_obs_jour = data["obs_j"].mean()
        min_nb_retrait_jour = data["retraits_j"].min()
        max_nb_retrait_jour = data["retraits_j"].max()
        moy_nb_retrait_jour = data["retraits_j"].mean()
        median_nb_retrait_jour = data["retraits_j"].median()
        std_nb_retrait_jour = data["retraits_j"].std()
        min_nb_versement_jour = data["versements_j"].min()
        max_nb_versement_jour = data["versements_j"].max()
        moy_nb_versement_jour = data["versements_j"].mean()
        median_nb_versement_jour = data["versements_j"].median()
        std_nb_versement_jour = data["versements_j"].std()
        print("Moyenne des opérations (versements / retraits) par jour :", moy_obs_jour)
        print("Plus petit nombre d'opérations observées en un jour :", min_obs_jour)
        print("Plus grand nombre d'opérations observées en un jour :", max_obs_jour)
        print("Moyenne du nombre de retraits par jour: ", moy_nb_retrait_jour)
        print("Médiane du nombre de retraits par jour: ", median_nb_retrait_jour)
        print("Ecart-type du nombre de retraits par jour: ", std_nb_retrait_jour)
        print("Plus petit nombre de retraits observés en un jour :", min_nb_retrait_jour)
        print("Plus grand nombre de retraits observés en un jour :", max_nb_retrait_jour)
        print("Plus petit nombre de versements observés en un jour :", min_nb_versement_jour)
        print("Plus grand nombre de versements observés en un jour :", max_nb_versement_jour)
        print("Moyenne du nombre de versements par jour: ", moy_nb_versement_jour)
        print("Médiane du nombre de versements par jour: ", median_nb_versement_jour)
        print("Ecart-type du nombre de versements par jour: ", std_nb_versement_jour)
        summary_obs = {"min_obs_j": min_obs_jour, "max_obs_j": max_obs_jour, "moy_obs_j": moy_obs_jour,
                       "min_nb_retraits_j": min_nb_retrait_jour, "max_nb_retraits_j": max_nb_retrait_jour,
                       "moy_nb_retraits_j": moy_nb_retrait_jour, "median_nb_retraits_j": median_nb_retrait_jour,
                       "std_nb_retraits_j": std_nb_retrait_jour, "min_nb_versements_j": min_nb_versement_jour,
                       "max_nb_versements_j": max_nb_versement_jour, "moy_nb_versements_j": moy_nb_versement_jour,
                       "median_nb_versements_j": median_nb_versement_jour, "std_nb_versements_j": std_nb_versement_jour}
        return summary_obs 
    

    def montants_obs_jour(self):
        stat_versement = self.data[self.data["crédit"] != 0].groupby("jour")["crédit"]
        stat_retrait = self.data[self.data["débit"] != 0].groupby("jour")["débit"]
        max_retrait = self.data[self.data["débit"] != 0]["débit"].max()
        min_retrait = self.data[self.data["débit"] != 0]["débit"].min()
        max_versement = self.data[self.data["crédit"] != 0]["crédit"].max()
        min_versement = self.data[self.data["crédit"] != 0]["crédit"].min()
        moy_versement = stat_versement.mean()   # On va s'intéresser à la moyenne des versements
        moy_retrait = stat_retrait.mean()   # Même remarque pour les retraits
        median_retrait = stat_retrait.median()
        median_versement = stat_versement.median()
        montants = {"moy_retraits": moy_retrait, "moy_versements": moy_versement, 
                    "median_retraits": median_retrait, "median_versements": median_versement,
                    "max_retrait": max_retrait, "min_retrait": min_retrait,
                    "max_versement": max_versement, "min_versement": min_versement}
        return montants
    

    def vals_seuil_montants(self):  # On va s'intéresser aux caractéristiques de la moyenne
        montant = self.montants_obs_jour()
        moy_retraits_jour = montant["moy_retraits"].mean()
        median_retraits_jour = montant["moy_retraits"].median()  # Médiane de la moyenne
        std_retraits_jour = montant["moy_retraits"].std()
        max_retrait = montant["max_retrait"]
        min_retrait = montant["min_retrait"]
        moy_versements_jour = montant["moy_versements"].mean()
        median_versements_jour = montant["moy_versements"].median()
        std_versements_jour = montant["moy_versements"].std()
        max_versement = montant["max_versement"]
        min_versement = montant["min_versement"]
        print("Plus grand retrait effectué: ", max_retrait)
        print("Plus petit retrait effectué: ", min_retrait)
        print("Retrait moyen par jour: ", moy_retraits_jour)
        print("Médiane des retraits moyens par jour: ", median_retraits_jour)
        print("Ecart-type de la moyenne des retraits par jour ", std_retraits_jour)
        print("Plus grand versement effectué: ", max_versement)
        print("Plus petit versement effectué: ", min_versement)
        print("Versement moyen par jour: ", moy_versements_jour)
        print("Médiane des versements moyens par jour: ", median_versements_jour)
        print("Ecart-tye de la moyenne des versements par jour: ", std_versements_jour)
        summary_montants = {"moy_retraits_j": moy_retraits_jour, "max_retrait": max_retrait, 
                    "min_retrait": min_retrait, "med_retraits_j": median_retraits_jour,
                    "std_retraits_j": std_retraits_jour, "moy_versements_j": moy_versements_jour,
                    "max_versement": max_versement, "min_versement": min_versement,
                    "med_versements_j": median_versements_jour, "std_versements_j": std_versements_jour}
        return summary_montants
    

    def vals_seuil_flux(self):
        flux = self.data.groupby("jour")["flux_net"].last()
        moy_flux_jour = flux.mean()
        median_flux_jour = flux.median()
        std_flux_jour = flux.std()
        max_flux_jour = flux.max()
        min_flux_jour = flux.min()
        print("Plus grand flux journalier connu par l'agence: ", max_flux_jour)
        print("Plus petit flux journalier connu par l'agence: ", min_flux_jour)
        print("Moyenne des flux journaliers: ", moy_flux_jour)
        print("Médiane des flux journaliers: ", median_flux_jour)
        print("Ecart-type des flux journaliers: ", std_flux_jour)
        summary_flux = {"max_flux_j": max_flux_jour, "min_flux_j": min_flux_jour,
                        "moy_flux_j": moy_flux_jour, "median_flux_j": median_flux_jour,
                        "std_flux_j": std_flux_jour}
        return summary_flux
    
    def nb_clients_annee(self):
        nb_clients = self.data["identifiant_client"].nunique()
        print(f"Nombre de clients pour l'agence {self.agence} à l'année {self.year}: ", nb_clients)
        return nb_clients



# Méthodes pour afficher les box plots des distributions: 

    def boxplot_nb_operations(self):
        nb = self.nb_obs_jour()
        nb_retrait_versement = pd.DataFrame({"nb_retraits_moyens": nb["retraits_j"],
                                             "nb_versements_moyens": nb["versements_j"]}).fillna(0)
        sns.boxplot(data = nb_retrait_versement)
        plt.title(f"Distribution du nombre de retraits et versements par jour pour l'agence {self.agence}")
        plt.ylabel("Nombre de transactions")
        plt.show()

    def boxplot_moy_montant_operations(self):
        montant = self.montants_obs_jour()
        montant_retrait_versement_flux = pd.DataFrame({"montant_retraits_moyens": montant["moy_retraits"],
                                                  "montant_versements_moyens": montant["moy_versements"]})
        sns.boxplot(data = montant_retrait_versement_flux)
        plt.title(f"Distribution du montant moyen des retraits et versements par jour pour l'agence {self.agence}")
        plt.ylabel("Montant moyen des transactions par jour")
        plt.show()

    def boxplot_median_montant_operations(self): 
        median = self.montants_obs_jour()
        median_retrait_versement = pd.DataFrame({"median_montant_retraits": median["median_retraits"],
                                                  "median_montant_versements": median["median_versements"]})
        sns.boxplot(data = median_retrait_versement)
        plt.title(f"Distribution du montant médian des retraits et versements par jour pour l'agence {self.agence}")
        plt.ylabel("Montant médian des transactions par jour")
        plt.show()

    def boxplot_flux(self):
        flux = self.data.groupby("jour")["flux_net"].last()
        sns.boxplot(data = flux)
        plt.title(f"Distribution des flux nets journaliers pour l'agence {self.agence}")
        plt.ylabel("Flux à la fin de la journée")
        plt.show()



# Méthodes pour calculer les quantiles des différentes distributions: 

    def quantiles_versements(self):
        versements = self.data[self.data["crédit"] != 0]["crédit"]
        médiane = versements.quantile(0.5)
        quantile_90 = versements.quantile(0.90)
        quantile_95 = versements.quantile(0.95)
        quantile_98 = versements.quantile(0.98)
        quantile_99 = versements.quantile(0.99)
        quantile_999 = versements.quantile(0.999)
        print("Médiane des versements: ", médiane)
        print("Quantile 0.90 des versements: ", quantile_90)
        print("Quantile 0.95 des versements: ", quantile_95)
        print("Quantile 0.98 des versements: ", quantile_98)
        print("Quantile 0.99 des versements: ", quantile_99)
        print("Quantile 0.999 des versements: ", quantile_999)
        dict_quantile_versements = {"quant_50": médiane, "quant_90": quantile_90, "quant_95": quantile_95, "quant_98": quantile_98, "quant_99": quantile_99, "quant_999": quantile_999}
        return dict_quantile_versements
        
    def quantiles_retraits(self):
        retraits = self.data[self.data["débit"] != 0]["débit"]
        médiane = retraits.quantile(0.5)
        quantile_90 = retraits.quantile(0.90)
        quantile_95 = retraits.quantile(0.95)
        quantile_98 = retraits.quantile(0.98)
        quantile_99 = retraits.quantile(0.99)
        quantile_999 = retraits.quantile(0.999)
        print("Médiane des retraits: ", médiane)
        print("Quantile 0.90 des retraits: ", quantile_90)
        print("Quantile 0.95 des retraits: ", quantile_95)
        print("Quantile 0.98 des retraits: ", quantile_98)
        print("Quantile 0.99 des retraits: ", quantile_99)
        print("Quantile 0.999 des retraits: ", quantile_999)
        dict_quantile_retraits = {"quant_50": médiane, "quant_90": quantile_90, "quant_95": quantile_95, "quant_98": quantile_98, "quant_99": quantile_99, "quant_999": quantile_999}
        return dict_quantile_retraits
    
    def quantiles_flux(self):
        flux = self.data.groupby("jour")["flux_net"].last()
        médiane = flux.quantile(0.5)
        quantile_90 = flux.quantile(0.90)
        quantile_98 = flux.quantile(0.98)
        quantile_99 = flux.quantile(0.99)
        quantile_999 = flux.quantile(0.999)
        print("Médiane des flux: ", médiane)
        print("Quantile 0.90 des flux journaliers: ", quantile_90)
        print("Quantile 0.98 des flux journaliers: ", quantile_98)
        print("Quantile 0.99 des flux journaliers: ", quantile_99)
        print("Quantile 0.999 des flux journaliers: ", quantile_999)
        dict_quantile_flux = {"quant_50": médiane, "quant_90": quantile_90, "quant_98": quantile_98, "quant_99": quantile_99, "quant_999": quantile_999}
        return dict_quantile_flux
        
    def define_quantile(self, value : float):
        if not (0 < value < 1):
            raise ValueError("La valeur entrée doit être strictement comprise entre 0 et 1")
        else:
            new_quantile_retrait = self.data[self.data["débit"] != 0]["débit"].quantile(value)
            new_quantile_versement = self.data[self.data["crédit"] != 0]["crédit"].quantile(value)
            new_quantile_flux = self.data.groupby("jour")["flux_net"].last().quantile(value)
            print(f"Quantile {value} pour les retraits: ", new_quantile_retrait)
            print(f"Quantile {value} pour les versements: ", new_quantile_versement)
            print(f"Quantile {value} pour les flux journaliers: ", new_quantile_flux)


# Méthodes pour visualiser les distributions (retraits, versements, flux nets) sur l'année:

    def distribution_retraits(self, nb_bins: int = 50):
        plt.figure(figsize = (14,12))
        max_retrait = self.data[self.data["débit"] != 0]["débit"].max()
        sns.histplot(self.data[self.data["débit"] != 0]["débit"], bins = nb_bins, kde = False, color = 'red')
        plt.xlim(0,max_retrait)
        plt.title(f"Distribution des retraits pour l'agence {self.agence} (en {self.year})")
        plt.xlabel("Montant retiré")
        plt.ylabel("Nombre de retraits")
        plt.grid(True)
        plt.show()

    def distribution_versements(self, nb_bins: int = 50):
        plt.figure(figsize = (14,12))
        max_versement = self.data[self.data["crédit"] != 0]["crédit"].max()
        sns.histplot(self.data[self.data["crédit"] != 0]["crédit"], bins = nb_bins, kde = False, color = 'green')
        plt.title(f"Distribution des versements pour l'agence {self.agence} (en {self.year})")
        plt.xlim(0,max_versement)
        plt.xlabel("Montant versé")
        plt.ylabel("Nombre de versements")
        plt.grid(True)
        plt.show()

    def distribution_flux(self, nb_bins: int = 30):
        flux = self.data.groupby("jour")["flux_net"].last()
        min_flux = flux.min()
        max_flux = flux.max()
        plt.figure(figsize=(14, 12))
        sns.histplot(flux, bins=nb_bins, kde=False, color='#40E0D0')
        plt.title(f"Distribution du flux net journalier pour l'agence {self.agence} (en {self.year})")
        plt.xlim(min_flux - 1000, max_flux + 1000) # On se laisse une petite marge quand même 
        plt.xlabel("Flux net journalier")
        plt.ylabel("Nombre de jours")
        plt.grid(True)
        plt.show()

    def comparaison_distributions_v_r(self, nb_bins: int = 50):
        plt.figure(figsize=(14, 12))
        retraits = self.data[self.data["débit"] != 0]["débit"]
        versements = self.data[self.data["crédit"] != 0]["crédit"]
        max_val = min(retraits.max(), versements.max())   # Quitte à couper, on conserve la plus petite pour visualiser
        sns.histplot(retraits, bins=nb_bins, kde=False, color='red', label="Retraits", alpha=0.2)
        sns.histplot(versements, bins=nb_bins, kde=False, color='green', label="Versements", alpha=0.2)
        plt.xlim(0, max_val)
        plt.xlabel("Montant")
        plt.ylabel("Nombre d'opérations")
        plt.title(f"Distribution des montants retirés et versés pour l'agence {self.agence} (en {self.year})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def custom_distrib_retraits(self, value_sup : float, value_inf: Optional[float] = None, nb_bins: int = 40):
        if value_inf is None:
            plt.figure(figsize = (14,12))
            sns.histplot(self.data[(self.data["débit"] != 0) & (self.data["débit"] <= value_sup)]["débit"], bins = nb_bins, kde = False, color = 'orange')
            plt.title(f"Distribution des retraits pour l'agence {self.agence} (en {self.year})")
            plt.xlim(0, value_sup)
            plt.xlabel("Montant retiré")
            plt.ylabel(f"Nombre de retraits inférieurs à {value_sup}")
            plt.grid(True)
            plt.show()
        else:
            plt.figure(figsize = (14,12))
            sns.histplot(self.data[(self.data["débit"] != 0) & (self.data["débit"] <= value_sup) & (self.data["débit"] >= value_inf)]["débit"], bins = nb_bins, kde = False, color = 'orange')
            plt.title(f"Distribution des retraits pour l'agence {self.agence} (en {self.year})")
            plt.xlim(value_inf, value_sup)
            plt.xlabel("Montant retiré")
            plt.ylabel(f"Nombre de retraits compris entre {value_inf} et {value_sup}")
            plt.grid(True)
            plt.show()

    def custom_distrib_versements(self, value_sup : float, value_inf: Optional[float] = None, nb_bins: int = 40):
        if value_inf is None:
            plt.figure(figsize = (14,12))
            sns.histplot(self.data[(self.data["crédit"] != 0) & (self.data["crédit"] <= value_sup)]["crédit"], bins = nb_bins, kde = False, color = 'blue')
            plt.title(f"Distribution des versements pour l'agence {self.agence} (en {self.year})")
            plt.xlim(0,value_sup)
            plt.xlabel("Montant versé")
            plt.ylabel(f"Nombre de versements inférieurs à {value_sup}")
            plt.grid(True)
            plt.show()
        else:
            plt.figure(figsize = (14,12))
            sns.histplot(self.data[(self.data["crédit"] != 0) & (self.data["crédit"] <= value_sup) & (self.data["crédit"] >= value_inf)]["crédit"], bins = nb_bins, kde = False, color = 'blue')
            plt.title(f"Distribution des versements pour l'agence {self.agence} (en {self.year})")
            plt.xlim(value_inf, value_sup)
            plt.xlabel("Montant versé")
            plt.ylabel(f"Nombre de versements compris entre {value_inf} et {value_sup}")
            plt.grid(True)
            plt.show()



# Méthodes pour la visualisation du comportement de l'agence en fonction du seuil:

    def possib_type_periode(self):
        liste_periodes = ['week', 'month', 'quarter', 'semester', 'season']
        print("La liste possible des périodes temporelles est: ", liste_periodes)


    def periodes_possibles(self, type_periode : str):
        assert type_periode in self.type_periode.keys(), "Période non reconnue ou mal formatée"
        print(f"Modalités possibles pour {type_periode}: ")
        for i, nom in enumerate(self.type_periode[type_periode], start=1):
            print(f"{i}. {nom}")


    def filtre_sur_periode(self, type_periode : str, nb_periode : int):
        ''' type_periode: 'month', 'quarter', 'semester', 'season', 'week'
        nb_periode: dépend de la période (1-12 pour mois...)'''
        liste_periodes = ['week', 'month', 'quarter', 'semester', 'season']
        assert type_periode in liste_periodes, "type_periode doit être une période temporelle valide"
        assert isinstance(nb_periode, int), "nb_periode doit être un entier"
        df = self.data.copy()
        dates = df["date_heure_operation"]
        if type_periode == 'month':
            df_filtered = df[dates.dt.month == nb_periode]  # Un entier entre 1 et 12
        elif type_periode == 'quarter':
            df_filtered = df[dates.dt.quarter == nb_periode] # Un entier entre 1 et 4 ici
        elif type_periode == 'semester':
            if nb_periode == 1:
                df_filtered = df[dates.dt.month.isin(range(1,7))]  # 1er semestre
            elif nb_periode == 2:
                df_filtered = df[dates.dt.month.isin(range(7,13))]  # 2nd semestre
            else:
                raise ValueError("L'argument 'nb_periode' doit être égal à 1 ou 2")
        elif type_periode == 'season':
            seasons = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]} #Printemps, Ete, Automne, Hiver
            if nb_periode not in seasons.keys():
                raise ValueError("L'argument 'nb_periode' doit être compris entre 1 et 4")
            df_filtered = df[dates.dt.month.isin(seasons[nb_periode])]
        elif type_periode == 'week':
            df_filtered = df[dates.dt.isocalendar().week == nb_periode]
        else:
            raise ValueError("Type de période inconnu ou problème ailleurs...")
        return df_filtered


    def plot_cumsum_montants_mois(self, type_periode : str, nb_periode : int):
        data_period = self.filtre_sur_periode(type_periode, nb_periode)
        data_period = data_period.sort_values("date_heure_operation")
        data_period["somme_cumule_montants"] = data_period["montant_operation"].cumsum()
        vals_fin_jour = data_period.groupby("jour")["somme_cumule_montants"].last().reset_index()
        plt.figure(figsize = (12,10))
        plt.plot(vals_fin_jour["jour"], vals_fin_jour["somme_cumule_montants"], marker = 'o')
        plt.title(f"Evolution des flux journaliers à la fin de chaque journée pour l'agence {self.agence} - {self.type_periode[type_periode][nb_periode-1]} {self.year}")
        plt.xlabel("Jour")
        plt.ylabel("Montant à la fin de la journée (en supposant repartir du stock de la veille)")
        plt.xticks(rotation = 45)
        plt.show()


    def seuil_debut_jour(self, type_periode : str, nb_periode : int, seuil : int = 0):
        data_period = self.filtre_sur_periode(type_periode, nb_periode)
        data_period = data_period.sort_values("date_heure_operation")
        df_cumule_jour = data_period.groupby("jour")["flux_net"].last().reset_index()
        df_cumule_jour["flux_net"] = df_cumule_jour["flux_net"] + seuil
        plt.figure(figsize=(12, 10))
        plt.scatter(df_cumule_jour["jour"], df_cumule_jour["flux_net"], color='blue', label='Stock en fin de journée')
        plt.axhline(y=seuil, color='red', linestyle='--', label=f'Seuil = {seuil}')
        for jour in df_cumule_jour["jour"]:
            plt.vlines(x=jour, ymin=seuil, ymax=df_cumule_jour.loc[df_cumule_jour["jour"] == jour, "flux_net"].values[0], 
                       colors='gray', linestyles='dotted', alpha=0.7)
        plt.title(f"Flux net final par jour, en partant chaque jour d'un seuil {seuil} pour l'agence {self.agence} - {self.type_periode[type_periode][nb_periode-1]} {self.year}")
        plt.xlabel("Journée")
        plt.ylabel(f"Stock à la fin de la journée (seuil initial = {seuil})")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.show()


    def pire_debit_jour_critique(self, type_periode : str, nb_periode : int):
        data_period = self.filtre_sur_periode(type_periode, nb_periode)
        data_period = data_period.sort_values("date_heure_operation")
        pire_debit = data_period["flux_net"].min()
        date_critique = data_period.loc[data_period["flux_net"] == pire_debit, "date_heure_operation"]
        print(f"Pire débit atteint par l'agence {self.agence} en {self.type_periode[type_periode][nb_periode-1]} {self.year}: ", pire_debit)
        print(f"Jour critique pour l'agence {self.agence} en {self.type_periode[type_periode][nb_periode-1]} {self.year}: ", date_critique)
        return pire_debit, date_critique.tolist()
    

    def nb_clients_actifs_periode(self, type_periode : str, nb_periode : int):
        data_period = self.filtre_sur_periode(type_periode, nb_periode)
        nb_clients_periode = data_period["identifiant_client"].nunique()
        print(f"Nombre de clients pour l'agence {self.agence} sur la période {self.type_periode[type_periode][nb_periode-1]} {self.year}: ", nb_clients_periode)
        return nb_clients_periode
    
    
    def calcul_proba_rupture(self, seuil : float, n_iter: int = 1000, ci : float = 0.95):
        flux_seuil = self.data.groupby("jour")["flux_net"].last() + seuil
        n_jour = self.nb_obs_jour()["j_ouvres"]
        ruptures = []   # On va évaluer 1000 fois la proba de rupture par Bootstrap
        normal = []
        for _ in range(n_iter):
            sample = flux_seuil.sample(n = n_jour, replace = True)
            nb_ruptures = (sample < 0).sum()
            nb_normal = n_jour - nb_ruptures
            ruptures.append(nb_ruptures / n_jour)
            normal.append(nb_normal / n_jour)
        proba_moy = np.mean(ruptures)
        proba_healthy = np.mean(normal)
        borne_inf_rupt = np.percentile(ruptures, (1-ci)/2*100)
        borne_sup_rupt = np.percentile(ruptures, (1+ci)/2*100)
        borne_inf_norm = np.percentile(normal, (1-ci)/2*100)
        borne_sup_norm = np.percentile(normal, (1+ci)/2*100)
        print(f"La probabilité de rupture pour l'agence {self.agence} au niveau de seuil {seuil} est de: ", proba_moy)
        print(f"La probabilité que l'agence passe la journée sans rupture avec le seuil {seuil} est de: ", proba_healthy)
        print(f"Intervalle de confiance bootstrap sur la probabilité de rupture: ", [round(borne_inf_rupt,3), round(borne_sup_rupt,3)])
        print(f"Intervalle de confiance bootstrap pour la probabilité de tranquilité: ", [round(borne_inf_norm,3), round(borne_sup_norm,3)])
        dict_proba = {"proba_rupt_estimee": proba_moy, "proba_tranq_estimee": proba_healthy, 
                      "CI_rupt": [round(borne_inf_rupt,3), round(borne_sup_rupt,3)], "CI_norm": [round(borne_inf_norm,3), round(borne_sup_norm,3)] }
        return dict_proba
    
    
    def calcul_proba_rupt_quant_99(self):  # Idee de la proba de rupture au quantile_99 des retraits
        quant_99 = self.quantiles_retraits()["quant_99"]
        estimate = self.calcul_proba_rupture(seuil = quant_99)
        return estimate["proba_rupt_estimee"] 
    

    def plot_seuil_proba_rupture(self, n_iter: int = 1000, ci: float = 0.95):
        quant_99 = self.quantiles_retraits()["quant_99"]
        minorant = quant_99 - 200000
        if minorant < 0:
            minorant = quant_99 - 100000
            if minorant < 0:
                minorant = quant_99
        liste_seuils = np.arange(minorant, quant_99 + 1000000, 50000)
        results = []
        for seuil in liste_seuils:
            proba_estimee = self.calcul_proba_rupture(seuil, n_iter=n_iter, ci=ci)
            proba_estimee["seuil"] = seuil
            results.append(proba_estimee)
        df_resultats = pd.DataFrame(results)
        seuils = df_resultats["seuil"]
        y_rupt = df_resultats["proba_rupt_estimee"]
        yerr_rupt = np.array([
            y_rupt - df_resultats["CI_rupt"].apply(lambda x: x[0]),
            df_resultats["CI_rupt"].apply(lambda x: x[1]) - y_rupt
            ])
        y_tranq = df_resultats["proba_tranq_estimee"]
        seuil_tranq_90 = df_resultats[y_tranq >= 0.90]["seuil"].min()
        seuil_tranq_95 = df_resultats[y_tranq >=0.95]["seuil"].min()
        yerr_tranq = np.array([
            y_tranq - df_resultats["CI_norm"].apply(lambda x: x[0]),
            df_resultats["CI_norm"].apply(lambda x: x[1]) - y_tranq
            ])
        fig, axs = plt.subplots(1, 2, figsize=(12, 10), sharex=True)
        if not pd.isna(seuil_tranq_90):
            axs[0].axvline(seuil_tranq_90, color='blue', linestyle='--', linewidth=2,
                   label=f"Tranquillité ≥ 90% ({int(seuil_tranq_90)})")
            axs[1].axvline(seuil_tranq_90, color='blue', linestyle='--', linewidth=2)
        if not pd.isna(seuil_tranq_95):
            axs[0].axvline(seuil_tranq_95, color='red', linestyle='--', linewidth=2,
                   label=f"Tranquillité ≥ 95% ({int(seuil_tranq_95)})")
            axs[1].axvline(seuil_tranq_95, color='red', linestyle='--', linewidth=2)
        axs[0].errorbar(seuils, y_rupt, yerr=yerr_rupt, fmt='o', color='darkred',
                    ecolor='lightcoral', capsize=3, label="Proba de rupture")
        if not pd.isna(seuil_tranq_90):
            axs[1].annotate(f"{int(seuil_tranq_90)}",
                    xy=(seuil_tranq_90, 0.92),
                    xytext=(seuil_tranq_90 + 10000, 0.92),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=10, color='blue')
        if not pd.isna(seuil_tranq_95):
            axs[1].annotate(f"{int(seuil_tranq_95)}",
                    xy=(seuil_tranq_95, 0.97),
                    xytext=(seuil_tranq_95 + 10000, 0.97),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        axs[0].set_xlabel("Seuil de départ")
        axs[0].set_ylabel("Probabilité de rupture")
        axs[0].set_title(f"Probabilité de rupture – Agence {self.agence} ({self.year})")
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].errorbar(seuils, y_tranq, yerr=yerr_tranq, fmt='o', color='darkgreen',
                    ecolor='lightgreen', capsize=3, label="Proba de tranquillité")
        axs[1].set_xlabel("Seuil de départ")
        axs[1].set_ylabel("Probabilité de non-rupture")
        axs[1].set_title(f"Probabilité de tranquillité – Agence {self.agence} ({self.year})")
        axs[1].grid(True)
        axs[1].legend()

        seuils = seuils.to_list()
        axs[0].set_xticks(seuils)
        axs[1].set_xticks(seuils)
        axs[0].tick_params(axis='x', rotation=45)
        axs[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()



# Méthodes pour les retraits importants, susceptibles de faire tomber l'agence en rupture: 

    def set_seuil_imp(self, seuil):
        self.seuil = seuil

    def retraits_imps(self):   # A modifier pour prendre en compte le cas où l'agence n'aurait vu aucun retrait de ce type
        retraits_imp = self.data[self.data["débit"] >= self.seuil].copy()
        dict_retraits_imp = {jour : [len(groupe), list(groupe["débit"])] 
                             for jour, groupe in retraits_imp.groupby("jour")}
        if retraits_imp.empty:
            print(f"Aucun retrait important détecté (au sens de la valeur seuil fournie {self.seuil})")
            return {}
        else:
            nb_retraits_imp = len(dict_retraits_imp)
            print(f"Nombre de retraits importants (supérieurs à {self.seuil}) pour l'agence {self.agence} en {self.year}: ", nb_retraits_imp)
            return dict_retraits_imp

    def distribution_retraits_imp(self):   # Renforcer la robustesse de la fonction
        dict_requis = self.retraits_imps()
        if not dict_requis:
            print(f"Aucun retrait détecté supérieur au seuil fourni {self.seuil}")
            return
        plot_retraits_imps = pd.DataFrame([
            {'date': pd.to_datetime(date), 'somme_retraits_imps_jour': sum(montants), "nombre_retraits_imp_jour": nb}
            for date, (nb,montants) in dict_requis.items()
        ])
        plot_retraits_imps = plot_retraits_imps.sort_values('date')
        fig, ax1 = plt.subplots(figsize = (15,13))
        color1 = 'tab:blue'
        ax1.set_xlabel('Date des retraits')
        ax1.set_ylabel("Somme des montants des retraits (en MDH)", color = color1)
        ax1.plot(plot_retraits_imps["date"], plot_retraits_imps["somme_retraits_imps_jour"], color = color1, marker = 'o', label = 'Montants retraits')
        ax1.tick_params(axis = 'y', labelcolor = color1)
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel("Nombre de retraits dans la journée", color = color2)
        ax2.plot(plot_retraits_imps["date"], plot_retraits_imps["nombre_retraits_imp_jour"], color = color2, marker = 's', linestyle = '--', label = 'Nombre retraits')
        ax2.tick_params(axis = 'y', labelcolor = color2)
        plt.title(f"Evolution de la somme et du nombre des retraits journaliers importants (supérieurs à {self.seuil} MAD) pour l'agence {self.agence} sur l'année {self.year}")
        fig.autofmt_xdate()
        plt.show()

    def freq_retraits_imps(self):
        freq_imp = (self.data[self.data["débit"] != 0]["débit"] > self.seuil).mean()*100
        print(f"La fréquence des retraits supérieurs à {self.seuil} pour l'agence {self.agence} en {self.year} est de: ", freq_imp)
        return freq_imp 
    
    def count_freq_above(self, threshold):
        jours_ouvres = self.nb_obs_jour()["j_ouvres"]
        above = self.data[self.data["débit"]>=threshold]
        count = above.shape[0]
        freq = count / jours_ouvres
        print("Nombre de retraits qui dépassent la valeur fixée: ", count)
        print("Fréquence de retraits qui dépassent la valeur fixée: ", freq)
        return count, freq

    def meshgrid_threshold(self):
        mesh = [threshold for threshold in np.arange(100000,1500000,100000)]
        for threshold in mesh:
            self.count_freq_above(threshold)
        print("Fin de l'exploration")

    def custom_meshgrid_threshold(self, limit_1 : int, limit_2 : int, jump : int):
        mesh = [threshold for threshold in np.arange(limit_1,limit_2,jump)]
        for threshold in mesh:
            self.count_freq_above(threshold)
        print("Fin de l'exploration")


# Méthodes pour lancer une première analyse globale et pour construire un DataFrame nécessaire au clustering:

    def analyse_preliminaire_data(self):
        self.object.nb_agences_annees_dataset()
        self.nb_clients_annee()
        self.vals_seuil_nb_obs()
        self.vals_seuil_montants()
        self.quantiles_retraits()
        self.quantiles_versements()
        self.quantiles_flux()
        seuil = int(input("Entrez un seuil de retrait important en fonction des quantiles précédents: "))
        self.set_seuil_imp(seuil)
        self.distribution_retraits_imp()
        self.freq_retraits_imps()
        self.boxplot_nb_operations()
        self.boxplot_moy_montant_operations()
        self.boxplot_median_montant_operations()
        self.distribution_versements()
        self.distribution_retraits()
        self.distribution_flux()
        self.comparaison_distributions_v_r()
        self.meshgrid_threshold()    
        self.plot_seuil_proba_rupture()


    def analyse_exploratoire_périodique(self): # On pourrait éventuellement la rendre plus robuste
        self.possib_type_periode()  # Permet de montrer à l'utilisateur les périodes disponibles
        type_periode = input("Entrez la période d'analyse souhaitée, en suivant les possibilités présentées: ")
        self.periodes_possibles(type_periode)  # Permet ensuite d'afficher les modalités de la période sélectionnée 
        nb_periode = int(input("Entrez le numéro de la période voulue: "))
        self.nb_clients_actifs_periode(type_periode, nb_periode)
        self.pire_debit_jour_critique(type_periode, nb_periode)
        self.plot_cumsum_montants_mois(type_periode, nb_periode)
        self.seuil_debut_jour(type_periode, nb_periode)


    def data_retrieval_clustering(self):
        dict_agence = {}
        dict_agence["code_agence"] = self.agence
        result_nb = self.vals_seuil_nb_obs()   
        result_quant = self.vals_seuil_montants()  # Vérifier cette fonction (qu'elle affiche bien tout)
        quantiles_retraits = self.quantiles_retraits()
        result_quant_95 = self.count_freq_above(quantiles_retraits["quant_95"])
        result_quant_99 = self.count_freq_above(quantiles_retraits["quant_99"])
        dict_agence["Moy_nb_versements_j"] = result_nb["moy_nb_versements_j"]
        dict_agence["Moy_nb_retraits_j"] = result_nb["moy_nb_retraits_j"]
        dict_agence["Moy_versements_j"] = result_quant["moy_versements_j"]
        dict_agence["Moy_retraits_j"] = result_quant["moy_retraits_j"]
        dict_agence["Median_nb_versements_j"] = result_nb["median_nb_versements_j"]
        dict_agence["Median_nb_retraits_j"] = result_nb["median_nb_retraits_j"]
        dict_agence["Median_versements_j"] = result_quant["med_versements_j"]
        dict_agence["Median_retraits_j"] = result_quant["med_retraits_j"]
        dict_agence["Std_nb_versements_j"] = result_nb["std_nb_versements_j"]
        dict_agence["Std_nb_retraits_j"] = result_nb["std_nb_retraits_j"]
        dict_agence["Std_versements_j"] = result_quant["std_versements_j"]
        dict_agence["Std_retraits_j"] = result_quant["std_retraits_j"]
        dict_agence[f"Nb_clients_{self.year}"] = self.nb_clients_annee()
        dict_agence["Nb_moy_transactions_j"] = result_nb["moy_obs_j"].mean()
        dict_agence["Nb_retraits_sup_quant_95"] = result_quant_95[0]
        dict_agence["Nb_retraits_sup_quant_99"] = result_quant_99[0]
        dict_agence["Freq_retraits_sup_quant_95"] = result_quant_95[0]
        dict_agence["Freq_retraits_sup_quant_99"] = result_quant_99[1]
        dict_agence["Retrait_max"] = result_quant["max_retrait"]
        dict_agence["Versement_max"] = result_quant["max_versement"]
        dict_agence["proba_rupt_quant_99"] = self.calcul_proba_rupt_quant_99() # 21 features
        return dict_agence   
    
    # Pb: On conserve beaucoup de features. Il faudra surement faire un tri avant de clusteriser
    # ou appliquer une méthode de réduction de dimension (type PCA...).
    # Rajouter des docstrings
    # Utiliser logging au lieu de print (il reste encore du boulot...)
    # Vérifier l'encodage des semaines (une fois que self.data est défini)
    # Implémenter une méthode pour le calcul empirique du seuil optimal