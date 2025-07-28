import os
import argparse
from Code_analyse_OOP_term import PreprocessingRawData, DataCharger, BasicStats
from Clustering_agences import Clustering_agences


def main():

    parser = argparse.ArgumentParser(description = "Pipeline d'analyse des données de transactions des agences.")

    # Paramètres de l'analyse:
    parser.add_argument('--preprocess', action = 'store_true', help = 'Lancement du preprocessing des données brutes.')
    parser.add_argument('--year', type = int, default = 2024, help = "Année du traitement.")
    parser.add_argument('--clustering', action = 'store_true', help = 'Lancement du clsutering.')
    parser.add_argument('--prelim_analysis', action = 'store_true', help = "Lancement de l'analyse statistique.")
    args = parser.parse_args()

    # Preprocessing des fichiers:
    raw_dir = f"C:/data/Agences {args.year}"  # A adapter
    filepath_raw_csv = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.csv')]
    clean_filename = f"Agences_nettoyees_completes_{args.year}.csv"  # Permet de variabiliser suivant l'année
    new_filepath = os.path.join("C:/data", clean_filename)

    if args.preprocess:
        preprocessing = PreprocessingRawData(filepath = filepath_raw_csv, newfilepath = new_filepath)
        preprocessing.preprocessing()  # Permet d'effectuer le preprocessing
        print(f"Fichier nettoyé sauvegardé pour l'année {args.year} et stocké à l'adresse {new_filepath}")
        print("Preprocessing terminé !")

    if args.prelim_analysis: # A compléter pour un peu plus tard
        pass  

    if args.clustering:
        pass

    