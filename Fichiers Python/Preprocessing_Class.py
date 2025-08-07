import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List



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
            df = pd.read_csv(filepath, low_memory = False, on_bad_lines = 'skip')
            df = df.sort_values("code_agence")  # On pourrait rajouter une vérif
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
            h_str = str(h).strip()
            if not h_str or h_str.lower() == 'nan':
                return None
            if ":" in h_str:
                parts = h_str.split(":")
                if len(parts) == 2:
                    return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:00"
                elif len(parts) == 3:
                    return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
                else:
                    return None
            h_str = h_str.zfill(6)
            return f"{h_str[:2]}:{h_str[2:4]}:{h_str[4:6]}"
        self.data["heure_operation"] = self.data["heure_operation"].apply(format_heure)
        self.data["heure_operation"] = pd.to_timedelta(self.data["heure_operation"], errors = 'coerce')
        nb_invalid = self.data["heure_operation"].isna().sum()
        if nb_invalid > 0:
            print(f"Attention: {nb_invalid} lignes ont une heure non convertible (valeurs mises à NaT)")
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

    
    def rectify_currency(self):
        '''Prend un taux moyen de conversion de 10 pour rectifier les observations non MAD'''

        self.check_not_None()
        columns = ["devise"]
        self.check_missing_columns(columns)
        mask = self.data["devise"] != 'MAD'
        if self.data.loc[mask].empty:
            print("Aucune observation à changer")
        else:
            self.data.loc[mask, "montant_operation"] *= 10
            print(f"{mask.sum()} observations corrigées.")


    def check_currency(self):
        '''Retourne les observations (et les dates correspondantes) dont la devise n'est pas en MAD'''
        
        # Vérifications préliminaires:
        self.check_not_None()
        columns = ["devise"]
        self.check_missing_columns(columns)
        # Décompte du nombre d'opérations incorrectes:
        print(self.data["devise"].value_counts(dropna=False))
        if (self.data["devise"] != 'MAD').any():
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


    def verif_vides(self):
        '''Enlève les observations avec des valeurs vides'''

        self.check_not_None()
        for column in self.data.columns:
            if self.data[column].isna().any():
                print(f"La colonne {column} contient des valeurs manquantes")
                self.data.dropna(subset = [column], inplace = True)
            else:
                print(f'Aucune valeur manquante dans la colonne {column}')

    
    def verif_encodage(self): 
        '''Vérification de l'encodage des colonnes importantes (en particulier pour les dates)'''
        
        # On commence par vérifier que les colonnes sont bien présentes dans le dataset:
        self.check_not_None()
        columns = ["date_heure_operation", "libelle_long_operation",
                         "libelle_court_operation", "identifiant_operation"]
        self.check_missing_columns(columns)
        try:
            self.data["date_heure_operation"] = pd.to_datetime(self.data["date_heure_operation"])
            self.data.index = self.data.index.astype(int)
            self.data["libelle_long_operation"] = self.data["libelle_long_operation"].astype('string')
            self.data["libelle_court_operation"] = self.data["libelle_court_operation"].astype('string')
            self.data["identifiant_operation"] = self.data["identifiant_operation"].astype('string')
        except Exception as e:
            raise RuntimeError(f"Erreur lors du reformatage: {e}")
        print("Encodage réalisé avec succès")
        print(self.data.info())


    def completion_data(self):  
        '''Ajout de colonnes nécessaires au traitement des données (notamment les flux)'''

        self.check_not_None()
        columns = ["date_heure_operation", "montant_operation"]
        self.check_missing_columns(columns)
        self.data["jour"] = self.data["date_heure_operation"].dt.date
        self.data = self.data.sort_values(by = ["code_agence","jour", "date_heure_operation"])
        self.data["crédit"] = self.data["montant_operation"].apply(lambda x: x if x>0 else 0)
        self.data["débit"] = self.data["montant_operation"].apply(lambda x: -x if x<0 else 0)
        self.data["flux_net"] = self.data.groupby(["code_agence","jour"])["montant_operation"].cumsum()
        self.data = self.data.set_index("code_agence")
        # On vérifie ensuite visuellement que tout s'est bien passé: 
        self.visu_data(30)

    
    def save_cleaned_data(self, newfilepath : str):
        '''Sauvegarde des données nettoyées dans newfilepath'''

        self.check_not_None()
        if self.aggregate and os.path.exists(newfilepath):
            complete = pd.read_csv(filepath = newfilepath, index_col = 0)
            to_save = pd.concat([complete,self.data], ignore_index = False)
        else:
            to_save = self.data
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
        self.rectify_currency()
        self.remove_columns()
        self.verif_vides()
        self.verif_encodage()
        self.completion_data()
        if not self.newfilepath:
            raise ValueError("Aucun chemin de sauvegarde spécifié")
        self.save_cleaned_data(self.newfilepath)


    def preprocessing_precise(self):
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
        if self.check_currency():   # A modifier pour prendre une fonction plus simple
            saisie = input("Insérez ici la liste des valeurs des taux de change appropriés pour les dates et les devises données, séparés par une virgule :")
            taux =  [float(elem.strip()) for elem in saisie.split(',')]
            self.change_currency(taux)
        self.remove_columns()
        self.verif_vides()
        self.verif_encodage()
        self.completion_data()
        if not self.newfilepath:
            raise ValueError("Aucun chemin de sauvegarde spécifié")
        self.save_cleaned_data(self.newfilepath)