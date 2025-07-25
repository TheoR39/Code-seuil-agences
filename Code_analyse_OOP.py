## Classes de preprocessing et de visualisation générale des données :

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from typing import Optional, List, Tuple, Dict

# 1ère classe : Preprocessing des données issues de Dremio :

class PreprocessingRawData:

    def __init__(self, filepath : Optional[List[str]] = None):
        self.filepath = filepath
        self.data = None

    def load_raw_csv(self):
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
        self.data = pd.concat(list_dfs, ignore_index = False)
        self.data = self.data.sort_index()
        print(f"Concaténation terminée, pour un nombre total d'observations de {self.data.shape[0]}")
        print("Visualisation des données après concaténation :")
        print(self.data.head(10))

    def describe_data(self):
        print("Description générale des données :")
        print(self.data.describe())

    def info_data(self):
        print("Infos importantes sur les données :")
        print(self.data.info())

    def visu_data(self):
        print("Visualisation des données :")
        print(self.data.head())

    def remove_duplicates(self):
        print("Nombre d'observations avec doublons :", self.data.shape[0])
        self.data.drop_duplicates(keep = 'first', inplace = True)
        print("Nombre d'observations après nettoyage des doublons :", self.data.shape[0])

    def time_format(self):
        self.data = self.data.dropna(subset = ["date_operation"])
        self.data["date_operation"]=  pd.to_datetime(self.data["date_heure_operation"]).dt.normalize()
        self.data["heure_operation"] = pd.to_timedelta(self.data["heure_operation"].astype('str') + ':00')
        self.data["date_heure_operation"] = self.data["heure_operation"] + self.data["date_operation"]
        self.data = self.data.sort_values("date_heure_operation")
        print("Visualisation des données après reformatage des dates (regarder la nouvelle colonne date_heure_operation): ")
        print(self.data.head())

    def algebrisation_montants(self):
        self.data.loc[self.data["sens_operation"] == 'D', "montant_operation"] *= -1
        self.data = self.data.drop(columns = ["sens_operation"])
        print("Visualisation des données après algébrisation des montants (regarder la colonne montant_operation): ")
        print(self.data.head())

    def check_currency(self):
        print(self.data["devise"].value_counts(dropna=False))
        if not (self.data["devise"] != 'MAD').any():
            print("Toutes les observations sont bien en MAD")
            return False
        else:
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

    def change_currency(self, taux):
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
        print("Vérification d'autres devises restantes :", (self.data["devise"] != 'MAD'.any()))

    def filtre_etat_operation(self):
        self.data = self.data[self.data["etat_operation"] == 2]
        print("Nombre d'observations après filtrage sur la colonne 'etat_operation' :", self.data.shape[0])

    def filtre_type_operation(self):
        vals_libelle = self.data["libelle_long_operation"].dropna().unique()
        categ_retraits = [valeur for valeur in vals_libelle if str(valeur).startswith("RETRAIT")]
        categ_versements = [valeur for valeur in vals_libelle if str(valeur).startswith("VERSEMENT")]
        categ_autres_especes = [valeur for valeur in vals_libelle if str(valeur).endswith("ESPECE")]
        categ_conservees = categ_retraits + categ_versements + categ_autres_especes
        self.data = self.data[self.data["libelle_long_operation"].isin(categ_conservees)]
        print("Liste des catégories d'opérations retenues :", categ_conservees)
        print("Nombre d'observations restantes :", self.data.shape[0])

    def remove_columns(self):
        print("Nombre de colonnes avant traitement :", self.data.shape[1])
        self.data = self.data.drop(columns = ['identifiant_compte', 'reference_operation', 'code_marche', 'etat_operation', 
        'code_famille_operation', 'code_type_operation', 'application_origine_operation', 'motif_operation', 'devise', 'numero_caisse',
        'heure_operation', 'date_operation', 'date_valeur', 'code_banque'], errors = 'ignore')
        print("Nombre de colonnes retenues :", self.data.shape[1])
        print("Informations après nettoyage des colonnes inutiles :")
        print(self.data.info())

    def save_cleaned_data(self, newfilepath : str):
        self.data.to_csv(newfilepath, index = True, encoding = 'utf-8')
        print("Données nettoyées enregistrées au format csv")

    def preprocessing(self):
        self.load_raw_data()
        self.describe_data()
        self.info_data()
        self.visu_data()
        self.remove_duplicates()
        self.time_format()
        self.algebrisation_montants()
        self.filtre_etat_operation()
        self.filtre_type_operation()
        if self.check_currency():
            saisie = input("Insérez ici la liste des valeurs des taux de change appropriés pour les dates et les devises données, séparés par une virgule :")
            taux =  [float(elem.strip()) for elem in saisie.split(',')]
            self.change_currency()
        self.remove_columns()
        self.visu_data()
        newfilepath = input("Insérez le nouveau chemin d'accès de la donnée nettoyée (en csv) :")
        self.save_cleaned_data(newfilepath)



# 2ème classe : Sélection de la donnée avant visualisation:

class DataCharger:

    def __init__(self, filepath = None, code = None, annee = None, choice = None):
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
        self.dataset = pd.read_csv(self, self.filepath, index_col = 0)
        self.dataset = self.dataset.sort_index()
        self.dataset = self.dataset.sort_values("date_heure_operation")
        self.dataset.index = self.dataset.index.astype(int)
        print("Données complètes chargées")
        print("Visualisation des données :")
        print(self.dataset.head(10))

    def verif_vides(self):
        for column in self.dataset.columns:
            if self.dataset[column].isna().any():
                print(f"La colonne {column} contient des valeurs manquantes")
                self.dataset.dropna(subset = [column], inplace = True)
            else:
                print(f'Aucune valeur manquante dans la colonne {column}')

    def verif_encodage(self):
        self.dataset["date_heure_operation"] = pd.to_datetime(self.dataset["date_heure_operation"])
        self.dataset.index = self.dataset.index.astype(int)
        self.dataset["libelle_long_operation"] = self.dataset["libelle_long_operation"].astype('string')
        self.dataset["libelle_court_operation"] = self.dataset["libelle_court_operation"].astype('string')
        self.dataset["identifiant_operation"] = self.dataset["identifiant_operation"].astype('string')
        print(self.dataset.info())

    def completion_data(self):
        self.dataset["jour"] = self.dataset["date_heure_operation"].dt.date
        self.dataset["crédit"] = self.dataset["montant_operation"].apply(lambda x: x if x>0 else 0)
        self.dataset["débit"] = self.dataset["montant_operation"].apply(lambda x: -x if x<0 else 0)

    def change_agence_year_choice(self, agence = None, annee = None, choice = None):
        if agence:
            self.code = agence
            print(f"Une (ou plusieurs) nouvelle(s) agence(s) a (ont) été sélectionnée(s) : {agence}")
        if annee:
            self.year = year
            print(f"Une (ou plusieurs) nouvelle(s) années(s) a (ont) été sélectionnée(s) : {annee}")
        if choice:
            self.choice = choice
            print(f'Un nouveau choix a été effectué : {choice}')

    def nb_agences_annees_dataset(self):
        nb_agences = self.dataset.index.nunique()
        print("Le nombre d'agences présentes dans le dataset complet est de :", nb_agences)
        annee_min = self.dataset["date_heure_operation"].dt.year.min()
        annee_max = self.dataset["date_heure_operation"].dt.year.max()
        print(f"Le dataset va de {annee_min} jusqu'à {annee_max}")

    def liste_annees_agences_data(self):
        if not self.data.empty:
            liste_agences = self.data.index.unique().tolist()
            liste_annees = self.data["date_heure_operation"].dt.year.unique().tolist()
        else:
            liste_agences = self.dataset.index.unique().tolist()
            liste_annees = self.dataset["date_heure_operation"].dt.year.unique().tolist()
        print("Liste des agences présentes dans le dataset complet :", liste_agences)
        print("Liste des années présentes dans le dataset complet :", liste_annees)
        return liste_agences, liste_annees

    def selection_agence(self):
        if self.code:
            self.data_agence = self.dataset[self.dataset.index.isin(self.code)].copy()
            print(f"Données chargées pour l'agence {self.code}")
        else:
            print("Aucun code spécifié ou pas dans le bon format...")

    def selection_annee(self):
        if self.year:
            if not self.code:
                self.data_years = self.dataset[self.dataset["date_heure_operation"].dt.year.isin(self.year)].copy()
                print(f"Données complètes chargées pour l'(les) année(s) {self.year}")
            else:
                self.data_years = self.data_agence[self.data_agence["date_heure_operation"].dt.year.isin(self.year)].copy()
                print(f"Données chargées pour l'(les) agence(s) {self.code} pour l'(les) année(s) {self.year}")
        else:
            print("Aucune année sélectionnée ou pas dans le bon format...")

    def group_by_agence(self):
        if self.code:
            self.grouped = dict(tuple(self.dataset.groupby("code_agence")))
        print("Le dataset grouped qui groupe les données par agence a bien été créé, et est disponible dans l'argument self.grouped")

    def assignation_donnee(self):
        if self.choice:
            if self.choice:
                if self.choice == 1:
                    self.data = self.dataset
                elif self.choice == 2:
                    self.data = self.data_agence
                else:
                    self.data = self.data_years
        else:
            if self.year:
                self.data = self.data_years
            elif self.code:
                self.data = self.data_agence
            else:
                self.data = self.dataset
        self.data = self.data.sort_values("date_heure_operation")
        print("Les données correspondantes ont bien été chargées et triées dans self.data")
        print("N.B.: Le choix des données peut toujours être modifié à l'aide de la méthode 'change_assignation' avec le paramètre other_choice")
        print("other_choice doit alors être donné sous la forme d'une liste du type [agences, annee, choix]")
        return self.data

    def change_assignation(self, other_choice):
        self.change_agence_year_choice(other_choice[0], other_choice[1], other_choice[2])
        return self.assignation_donnee()

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

    def preparer_donneees(self):
        self.load_csv()
        self.verif_vides()
        self.verif_encodage()
        self.completion_data()
        self.nb_agences_annees_dataset()
        self.selection_agence()
        self.selection_annee()
        self.assignation_donnee()
        return self.assignation_donnee()   # Potentiellement optionnel (à voir) (apparemment nécessaire)
        


# 3ème classe : Visualisation des données et statistiques élémentaires :

class BasicStats:

    def __init__(self, class_data, data, agence = None, year = None):
        self.object = class_data
        self.data = data
        self.dataset = data
        self.agence = agence
        self.year = year
        self.month = None
        self.mois_possibles = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
        print("Visualisation préliminaire des données :")
        print(self.data.head(10))

    def agences_annees_possibles(self):
        liste_agences, liste_annees = self.object.liste_annees_agences_data()
        print("Liste des agences possibles :", liste_agences)
        print("Liste des années possibles :", liste_annees)

    def choix_agence_annee_mois(self, agence, year, month = None):
        if agence:
            if agence in self.data.index.unique():
                self.agence = agence
                self.data = self.data[self.data.index == self.agence].copy()
            else:
                print("La valeur spécifiée n'est pas bonne. Choisissez une agence présente dans le dataset")
        if year:
            liste_annees = self.data["date_heure_operation"].dt.year.unique()
            if year in liste_annees:
                self.year = year
                self.data = self.data[self.data["date_heure_operation"].dt.year == self.year].copy()
            else:
                print("L'année fournie n'est pas présente dans le dataset")
        if month:
            if month in range(1,13):
                self.month = month
            else:
                print("La valeur donnée du mois n'est pas correcte")

    def etendue_date(self):
        date_min = self.data["date_heure_operation"].min()
        date_max = self.data["date_heure_operation"].max()
        print("Plus petite date du jeu de données :", date_min)
        print("Plus grande date du jeu de données :", date_max)

    def visu_data(self, nb_lignes = 5):
        print(self.data.head(nb_lignes))

    def info_data(self):
        print(self.data.info())

    def nb_obs_jour(self):   # Fonction potentiellement à retravailler pour inclure un filtre sur le mois ou la semaine
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
        min_nb_versement_jour = data["versements_j"].min()
        max_nb_versement_jour = data["versements_j"].max()
        print("Plus petit nombre d'opérations observées en un jour :", min_obs_jour)
        print("Plus grand nombre d'opérations observées en un jour :", max_obs_jour)
        print("Moyenne des opérations (versements / retraits) par jour :", moy_obs_jour)
        print("Plus petit nombre de retraits observés en un jour :", min_nb_retrait_jour)
        print("Plus grand nombre de retraits observés en un jour :", max_nb_retrait_jour)
        print("Plus petit nombre de versements observés en un jour :", min_nb_versement_jour)
        print("Plus grand nombre de versements observés en un jour :", max_nb_versement_jour)

    def boxplot_nb_operations(self):
        nb = self.nb_obs_jour()
        nb_retrait_versement = pd.DataFrame({"nb_retraits_moyens": nb["retraits_j"],
                                             "nb_versements_moyens": nb["versements_j"]}).fillna(0)
        sns.boxplot(data = nb_retrait_versement)
        plt.title(f"Distribution du nombre de retraits et versements par jour pour l'agence {self.agence}")
        plt.ylabel("Nombre de transactions")
        plt.show()

    def montants_obs_jour(self):
        moy_retraits_jour = self.data[self.data["débit"] != 0].groupby("jour")["débit"].mean()
        median_retraits_jour = self.data[self.data["débit"] != 0].groupby("jour")["débit"].median()
        moy_versements_jour = self.data[self.data["crédit"] != 0].groupby("jour")["crédit"].mean()
        median_versements_jour = self.data[self.data["crédit"] != 0].groupby("jour")["crédit"].median()
        montants = {"moy_retraits_j": moy_retraits_jour, "moy_versements_j": moy_versements_jour, "med_retraits_j":median_retraits_jour,
                    "med_versements_j": median_versements_jour}
        return montants

    def vals_seuil_montants(self):
        montant = self.montants_obs_jour()
        retraits_moy_j = montant["moy_retraits_j"].mean()
        versement_moy_j = montant["moy_versements_j"].mean()
        print(f"Retrait moyen par jour pour l'agence {self.agence}: ", retraits_moy_j)
        print(f"Versement moyen par jour pour l'agence {self.agence}: ", versement_moy_j)

    def boxplot_moy_montant_operations(self):
        montant = self.montants_obs_jour()
        montant_retrait_versement = pd.DataFrame({"montant_retraits_moyens": montant["moy_retraits_j"],
                                                  "montant_versements_moyens": montant["moy_versements_j"]})
        sns.boxplot(data = montant_retrait_versement)
        plt.title(f"Distribution du montant moyen des retraits et versements par jour pour l'agence {self.agence}")
        plt.ylabel("Montant moyen des transactions")
        plt.show()

    def boxplot_median_montant_operations(self):
        median = self.montants_obs_jour()
        median_retrait_versement = pd.DataFrame({"median_montant_retraits": median["med_retraits_j"],
                                                  "median_montant_versements": median["med_versements_j"]})
        sns.boxplot(data = median_retrait_versement)
        plt.title(f"Distribution du montant médian des retraits et versements par jour pour l'agence {self.agence}")
        plt.ylabel("Montant médian des transactions")
        plt.show()

    def quantiles_retraits(self):
        médiane = self.data[self.data["débit"] != 0]["débit"].quantile(0.5)
        quantile_90 = self.data[self.data["débit"] != 0]["débit"].quantile(0.90)
        quantile_98 = self.data[self.data["débit"] != 0]["débit"].quantile(0.98)
        quantile_99 = self.data[self.data["débit"] != 0]["débit"].quantile(0.99)
        quantile_999 = self.data[self.data["débit"] != 0]["débit"].quantile(0.999)
        print("Médiane des retraits: ", médiane)
        print("Quantile 0.90 des retraits: ", quantile_90)
        print("Quantile 0.98 des retraits: ", quantile_98)
        print("Quantile 0.99 des retraits: ", quantile_99)
        print("Quantile 0.999 des retraits: ", quantile_999)

    def quantiles_versements(self):
        médiane = self.data[self.data["crédit"] != 0]["crédit"].quantile(0.5)
        quantile_90 = self.data[self.data["crédit"] != 0]["crédit"].quantile(0.90)
        quantile_98 = self.data[self.data["crédit"] != 0]["crédit"].quantile(0.98)
        quantile_99 = self.data[self.data["crédit"] != 0]["crédit"].quantile(0.99)
        quantile_999 = self.data[self.data["crédit"] != 0]["crédit"].quantile(0.999)
        print("Médiane des retraits: ", médiane)
        print("Quantile 0.90 des retraits: ", quantile_90)
        print("Quantile 0.98 des retraits: ", quantile_98)
        print("Quantile 0.99 des retraits: ", quantile_99)
        print("Quantile 0.999 des retraits: ", quantile_999)

    def define_quantile(self, value : float):
        if not (0 < value < 1):
            raise ValueError("La valeur entrée doit être strictement comprise entre 0 et 1")
        else:
            new_quantile_retrait = self.data[self.data["débit"] != 0]["débit"].quantile(value)
            new_quantile_versement = self.data[self.data["crédit"] != 0]["crédit"].quantile(value)
            print(f"Quantile {value} pour les retraits: ", new_quantile_retrait)
            print(f"Quantile {value} pour les versements: ", new_quantile_versement)

    def distribution_retraits(self, nb_bins = 50):
        plt.figure(figsize = (14,12))
        sns.histplot(self.data[self.data["débit"] != 0]["débit"], bins = nb_bins, kde = False, color = 'red')
        plt.title(f"Distribution des retraits pour l'agence {self.agence} (en {self.year})")
        plt.xlabel("Montant retiré")
        plt.ylabel("Nombre de retraits")
        plt.grid(True)
        plt.show()

    def distribution_versements(self, nb_bins = 50):
        plt.figure(figsize = (14,12))
        sns.histplot(self.data[self.data["crédit"] != 0]["crédit"], bins = nb_bins, kde = False, color = 'green')
        plt.title(f"Distribution des versements pour l'agence {self.agence} (en {self.year})")
        plt.xlabel("Montant versé")
        plt.ylabel("Nombre de versements")
        plt.grid(True)
        plt.show()

    def custom_distrib_retraits(self, value_sup : float, value_inf = None, nb_bins = 40):
        if not value_inf:
            plt.figure(figsize = (14,12))
            sns.histplot(self.data[(self.data["débit"] != 0) & (self.data["débit"] <= value_sup)]["débit"], bins = nb_bins, kde = False, color = 'orange')
            plt.title(f"Distribution des retraits pour l'agence {self.agence} (en {self.year})")
            plt.xlabel("Montant retiré")
            plt.ylabel(f"Nombre de retraits inférieurs à {value_sup}")
            plt.grid(True)
            plt.show()
        else:
            plt.figure(figsize = (14,12))
            sns.histplot(self.data[(self.data["débit"] != 0) & (self.data["débit"] <= value_sup) & (self.data["débit"] >= value_inf)]["débit"], bins = nb_bins, kde = False, color = 'orange')
            plt.title(f"Distribution des retraits pour l'agence {self.agence} (en {self.year})")
            plt.xlabel("Montant retiré")
            plt.ylabel(f"Nombre de retraits compris entre {value_inf} et {value_sup}")
            plt.grid(True)
            plt.show()

    def custom_distrib_versements(self, value_sup, value_inf = None, nb_bins = 40):
        if not value_inf:
            plt.figure(figsize = (14,12))
            sns.histplot(self.data[(self.data["crédit"] != 0) & (self.data["crédit"] <= value_sup)]["crédit"], bins = nb_bins, kde = False, color = 'blue')
            plt.title(f"Distribution des versements pour l'agence {self.agence} (en {self.year})")
            plt.xlabel("Montant versé")
            plt.ylabel(f"Nombre de versements inférieurs à {value_sup}")
            plt.grid(True)
            plt.show()
        else:
            plt.figure(figsize = (14,12))
            sns.histplot(self.data[(self.data["crédit"] != 0) & (self.data["crédit"] <= value_sup) & (self.data["crédit"] >= value_inf)]["crédit"], bins = nb_bins, kde = False, color = 'blue')
            plt.title(f"Distribution des versements pour l'agence {self.agence} (en {self.year})")
            plt.xlabel("Montant versé")
            plt.ylabel(f"Nombre de versements compris entre {value_inf} et {value_sup}")
            plt.grid(True)
            plt.show()

    def plot_cumsum_montants_mois(self, mois):
        self.month = self.data[self.data["date_heure_operation"].dt.month == mois]
        self.month = self.month.sort_values("date_heure_operation")
        self.month = self.month.copy()
        self.month["somme_cumule_montants"] = self.month["montant_operation"].cumsum()
        vals_fin_jour = self.month.groupby("jour")["somme_cumule_montants"].last().reset_index()
        plt.figure(figsize = (12,10))
        plt.plot(vals_fin_jour["jour"], vals_fin_jour["somme_cumule_montants"], marker = 'o')
        plt.title(f"Evolution des montants (versements - retraits) à la fin de chaque journée pour l'agence {self.agence} - {self.mois_possibles[mois-1]} {self.year}")
        plt.xlabel("Jour")
        plt.ylabel("Montant à la fin de la journée")
        plt.xticks(rotation = 45)
        plt.show()

    def seuil_debut_jour(self, mois, seuil = 0):
        self.month = self.data[self.data["date_heure_operation"].dt.month == mois]
        self.month = self.month.sort_values("date_heure_operation")
        self.month = self.month.copy()
        self.month["cumsum_montants"] = self.month.groupby("jour")["montant_operation"].cumsum()
        df_cumule_jour = self.month.groupby("jour")["cumsum_montants"].last().reset_index()
        df_cumule_jour["cumsum_montants"] = df_cumule_jour["cumsum_montants"] + seuil
        plt.figure(figsize = (12,10))
        plt.plot(df_cumule_jour["jour"], df_cumule_jour["cumsum_montants"], marker = 'o')
        plt.title(f"Montants cumulés (versements - retraits) par jour, en supposant un seuil {seuil} pour l'agence {self.agence} - {self.mois_possibles[mois-1]} {self.year}")
        plt.xlabel("Journée")
        plt.ylabel(f"Montant cumulé sur la journée en partant d'un seuil {seuil}")
        plt.xticks(rotation = 45)
        plt.grid(True)
        plt.show()

    def pire_debit(self, mois):
        self.month = self.data[self.data["date_heure_operation"].dt.month == mois]
        self.month = self.month.sort_values("date_heure_operation")
        self.month = self.month.copy()
        self.month["somme_cumule_montants"] = self.month.groupby("jour")["montant_operation"].cumsum()
        pire_debit = self.month["somme_cumule_montants"].min()
        print(f"Pire débit atteint par l'agence {self.agence} en {self.mois_possibles[mois-1]} {self.year}: ", pire_debit)
        return pire_debit

    def jour_critique(self, mois):
        pire_debit = self.pire_debit()
        self.month = self.data[self.data["date_heure_operation"].dt.month == mois]
        self.month = self.month.sort_values("date_heure_operation")
        self.month = self.month.copy()
        self.month["somme_cumule_montants"] = self.month.groupby("jour")["montant_operation"].cumsum()
        jour_critique = self.month.loc[self.month["somme_cumule_montants"] == pire_debit]
        print(f"Jour critique pour l'agence {self.agence} en {self.mois_possibles[mois-1]} {self.year}: ", jour_critique)

    def nb_clients(self, mois = None):
        if not mois:
            nb_clients = self.data["identifiant_client"].nunique()
            print(f"Nombre de clients pour l'agence {self.agence} à l'année {self.year}: ", nb_clients)
        else:
            nb_clients = self.data[self.data["date_heure_operation"].dt.month == mois]["identifiant_client"].nunique()
            print(f"Nombre de clients pour l'agence {self.agence} en {self.mois_possibles[mois-1]} {self.year}: ", nb_clients)

    def retraits_imps(self, seuil = 1000000):   # A modifier pour prendre en compte le cas où l'agence n'aurait vu aucun retrait de ce type
        retraits_imp = self.data[self.data["débit"] > seuil].copy()
        dict_retraits_imp = {jour : [len(groupe), list(groupe["débit"])] 
                             for jour, groupe in retraits_imp.groupby("jour")}
        nb_retraits_imp = len(dict_retraits_imp)
        print(f"Nombre de retraits importants (supérieurs à {seuil}) pour l'agence {self.agence} en {self.year}: ", nb_retraits_imp)
        return dict_retraits_imp

    def visu_retraits_imp(self, quantite = 10, seuil = 1000000):
        dict_requis = self.retraits_imps(seuil)
        plot_retraits_imps = pd.DataFrame([
            {'date': pd.to_datetime(date), 'somme_retraits_imps_jour': sum(montants), "nombre_retraits_imp_jour": nb}
            for date, (nb,montants) in dict_requis.items()
        ])
        plot_retraits_imps.sort_values('date')
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
        plt.title(f"Evolution de la somme et du nombre des retraits journaliers importants (supérieurs à {seuil} MAD) pour l'agence {self.agence} sur l'année {self.year}")
        fig.autofmt_xdate()
        plt.show()

    def freq_retraits_imps(self, seuil = 1000000):
        freq_imp = (self.data[self.data["débit"] != 0]["débit"] > seuil).mean()*100
        print(f"La fréquence des retraits supérieurs à {seuil} pour l'agence {self.agence} en {self.year} est de: ", freq_imp)

    def analyse_preliminaire_data(self):
        self.agence_annee_possible()
        agence = int(input("Entrez un code d'agence parmi les codes disponibles précédents: "))
        annee = int(input("Entrez une année parmi la liste disponible précédente: "))
        self.choix_agence_annee_mois(agence, annee)
        self.etendue_date()
        self.visu_data(10)
        self.info_data()
        self.nb_clients()
        self.vals_seuil_nb_obs()
        self.vals_seuil_montants()
        self.quantiles_retraits()
        self.quantiles_versements()
        self.visu_retraits_imp()
        self.freq_retraits_imps()
        self.boxplot_nb_operations()
        self.boxplot_moy_montant_operations()
        self.boxplot_median_montant_operations()
        self.distribution_retraits()
        self.distrib_retraits_imp()
        month = int(input("Entrez une valeur entre 1 et 12 qui représente le mois correspondant: "))
        self.nb_clients(month)
        self.pire_debit(month)
        self.jour_critique(month)
        self.plot_cumsum_montants_mois(month)
        self.seuil_debut_jour(month)
        