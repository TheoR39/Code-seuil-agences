import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# from datetime import datetime
from typing import Optional
from DataCharger_Class import DataCharger


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
            return {"quant_retrait": new_quantile_retrait, "quant_versement": new_quantile_versement,
                    "quant_flux": new_quantile_flux}


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
        '''Fonction qui permet d'estimer la probabilité de rupture de l'agence à un seuil donné'''

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
        '''Affiche les seuils qui garantissent (avec intervalle de confiance à 95%) une probabilité
        de rupture de moins de 10% et moins de 5% pour une agence donnée'''

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


    def calcul_seuils_tranquilite(self, n_iter : Optional[int] = 1000, ci: Optional[float] = 0.95):
        '''Fonction équivalente à la précédente, mais qui permet de retourner les valeurs d'intérêt'''

        quant_99 = self.quantiles_retraits()["quant_99"]
        minorant = quant_99 - 200000
        if minorant < 0:
            minorant = quant_99 - 100000
            if minorant < 0: 
                minorant  = quant_99

        liste_seuils = np.arange(minorant, quant_99 + 1000000, 50000)
        results = []
        for seuil in liste_seuils:
            proba_estimee = self.calcul_proba_rupture(seuil, n_iter = n_iter, ci = ci)
            proba_estimee["seuil"] = seuil
            results.append(proba_estimee)
        y_tranq = [r["proba_tranq_estimee"] for r in results]
        seuil_tranq_90 = min((r["seuil"] for r in results if r["proba_tranq_estimee"] >= 0.90),
                             default = None)
        seuil_tranq_95 = min((r["seuil"] for r in results if r["proba_tranq_estimee"] >= 0.95),
                             default = None)

        return {"df_results": results, "seuil_90": seuil_tranq_90, "seuil_95": seuil_tranq_95}



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


    def evaluate_freq_flux(self):
        '''Il s'agit d'évaluer le comportement de l'agence: plutôt créditrice ou débitrice'''

        flux_jour = self.data.groupby("jour")["flux_net"].last()
        n_days = len(flux_jour)
        freq_pos = (flux_jour >= 0).sum() / n_days
        freq_neg = (flux_jour < 0).sum() / n_days
        return freq_pos, freq_neg
        



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
    

    def data_retrieval_optim(self):
        '''Fonction pour récupérer les données contenant les données nécessaires à l'optimisation'''

        dict_agence_optim = {}
        dict_agence_optim["code_agence"] = self.agence
        dict_agence_optim["nb_j_ouvrés"] = self.nb_obs_jour()["j_ouvres"]
        dict_agence_optim["freq_pos"] = self.evaluate_freq_flux()[0]
        dict_agence_optim["flux_net"] = self.data.groupby("jour")["flux_net"].last().to_dict()
        results = self.calcul_seuils_tranquilite()
        dict_agence_optim["seuil_90"] = results["seuil_90"]
        dict_agence_optim["seuil_95"] = results["seuil_95"]
        dict_agence_optim["info_comp_seuils"] = results["df_results"]
        liste_possible = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        dict_quantiles = {k : self.define_quantile(k)["quant_retrait"] for k in liste_possible}  # Permet de conserver l'indicatif du quantile
        dict_agence_optim["quantiles"] = dict_quantiles
        return dict_agence_optim
            
        # Manque les quantiles et les probas de tranquilité 90 - 95

    
    # Pb: On conserve beaucoup de features. Il faudra surement faire un tri avant de clusteriser
    # ou appliquer une méthode de réduction de dimension (type PCA...).
    # Rajouter des docstrings
    # Utiliser logging au lieu de print (il reste encore du boulot...)
    # Vérifier l'encodage des semaines (une fois que self.data est défini)
    # Implémenter une méthode pour le calcul empirique du seuil optimal