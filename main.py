import os
import argparse
from Code_analyse_OOP_term import PreprocessingRawData, DataCharger, BasicStats
from Clustering_agences import Clustering_agences


def main():

    parser = argparse.ArgumentParser(description = "Pipeline d'analyse des données de transactions des agences.")

    # Paramètres de l'analyse:
    parser.add_argument('--preprocess', action = 'store_true', help = 'Lancement du preprocessing des données brutes.')
    parser.add_argument('--clustering', action = 'store_true', help = 'Lancement du clsutering.')
    parser.add_argument('--prelim_analysis', action = 'store_true', help = "Lancement de l'analyse statistique.")

    parser.add_argument('--year', type = int, default = 2024, help = "Année du traitement.")
    parser.add_argument('--years', nargs = '*', type = int, help = "Années de sélection pour l'analyse statistique")
    parser.add_argument('--agences', nargs = '*', type = int, help = "Liste sélectionnée des agences")
    parser.add_argument('--aggregate', action = 'store_true', help = "Utilisé en cas de présence de données déjà nettoyées")
    args = parser.parse_args()

    # Preprocessing des fichiers:
    raw_dir = f"C:/data/Agences {args.year}"  # A adapter
    filepath_raw_csv = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.csv')]
    clean_filepath = "C:/data/Agences_nettoyees_completes.csv" # A modifier suivant le chemin d'accès réel


    if args.preprocess:
        print(f"Lancement du preprocessing  pour l'année {args.year}")
        print(f"Mode agrégation: {'ON' if args.aggregate else 'OFF'}")
        cleaning = PreprocessingRawData(filepath = filepath_raw_csv, newfilepath = clean_filepath, aggregate = args.aggregate)
        cleaning.preprocessing()
        print(f"Fichier nettoyé sauvegardé pour l'année {args.year} et stocké à l'adresse {clean_filepath}")
        print("Preprocessing terminé!")

    if args.prelim_analysis: # A compléter pour un peu plus tard
        print(f"Lancement d'une analyse statistique exploratoire pour l'(les) agence(s) {args.agences} sur l'(les) années {args.years}")
        data = DataCharger(filepath = clean_filepath, code = args.agences, annee = args.years)
        data.analyse_preliminaire_data()
        print("Analyse descriptive terminée!")

    if args.clustering:
        pass

    