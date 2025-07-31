import os
import argparse
from Code_analyse_OOP_term import PreprocessingRawData, DataCharger, BasicStats
from Clustering_agences import Clustering_agences

# Attention, il faut adapter les chemins d'accès suivant l'endroit de stockage des données

def main():
    '''Lancement de toute la pipeline, de la récupération des données au clustering (pour l'instant)'''

    parser = argparse.ArgumentParser(description = "Pipeline d'analyse des données de transactions des agences.")

    # Paramètres de l'analyse:
    parser.add_argument('--preprocess', action = 'store_true', help = 'Lancement du preprocessing des données brutes.')
    parser.add_argument('--clustering', action = 'store_true', help = 'Lancement du clsutering.')
    parser.add_argument('--prelim_analysis', action = 'store_true', help = "Lancement de l'analyse statistique.")

    parser.add_argument('--year', type = int, default = 2024, help = "Année du traitement.")
    parser.add_argument('--years', nargs = '*', type = int, help = "Années de sélection pour l'analyse statistique.")
    parser.add_argument('--agences', nargs = '*', type = int, help = "Liste sélectionnée des agences.")
    parser.add_argument('--aggregate', action = 'store_true', help = "Utilisé en cas de présence de données déjà nettoyées.")
    parser.add_argument('--already_created', action = 'store_true', help = "Les données pour le clustering existent déja.")
    args = parser.parse_args()

    # Preprocessing des fichiers:
    raw_dir = f"C:/data/Agences {args.year}"  # A adapter (dossier contenant les fichiers pour l'année 2024)
    filepath_raw_csv = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.csv')]
    clean_filepath = "C:/data/Agences_nettoyees_completes.csv"  # Pour stocker les données nettoyées
    clustering_filepath = "C:/data/Données_clustering.csv"  # Pour stocker les données du clustering
    if args.years is None:
        args.years = [args.year]

    if args.prelim_analysis and not args.agences:
        raise ValueError("Aucune agence spécifique n'a été sélectionnée.")
    
    try:
        if args.preprocess:
            print(f"Lancement du preprocessing  pour l'année {args.year}")
            print(f"Mode agrégation: {'ON' if args.aggregate else 'OFF'}")
            if not filepath_raw_csv:
                raise FileNotFoundError(f"Aucun fichier csv associé à la variable {raw_dir}")
            cleaning = PreprocessingRawData(filepath = filepath_raw_csv, newfilepath = clean_filepath, aggregate = args.aggregate)
            cleaning.preprocessing()
            print(f"Fichier nettoyé sauvegardé pour l'année {args.year} et stocké à l'adresse {clean_filepath}")
            print("Preprocessing terminé!")

        if args.prelim_analysis: # A compléter pour un peu plus tard
            if not os.path.exists(clean_filepath):
                raise FileNotFoundError(f"Pas de fichier nettoyé trouvé dans {clean_filepath}")
            try:
                if args.agence and len(args.agence) != 1:
                    print("L'analyse nécessite de ne sélectionner qu'une seule agence")
                if args.agence and len(args.agence) == 1:
                    print(f"Lancement d'une analyse statistique exploratoire pour l'(les) agence(s) {args.agences} sur l'(les) années {args.years}")
                    choice = DataCharger(filepath = clean_filepath, code = args.agences, annee = args.years)
                    choice.preparer_donnees()
                    data = BasicStats(choice)
                    data.analyse_preliminaire_data()
                    print("Fin de l'analyse exploratoire préliminaire")
                    print("Il est possible d'affiner sur une période donnée en lançant la méthode 'analyse_exploratoire_periodique'") 
            except Exception as e:
                print(f"Erreur lors de l'analyse des données de l'agence {args.agence} {e}")
        
        if args.clustering:
            if not os.path.exists(clean_filepath):
                raise FileNotFoundError(f"Fichier de données nettoyées introuvable ou inexistant à l'adresse : {clean_filepath}")
            print("Lancement du clustering des agences")
            print("Clustering suivant k-means / hierarchical clustering / DBSCAN")
            clustering = Clustering_agences(new_filepath = clustering_filepath, filepath = clean_filepath,
                                            already_created = args.already_created) # A compléter...
            clustering.agreg_clustering()
            print("Clustering terminé!")

    except Exception as e:
        print(f"Erreur lors de l'exécution {e}")
        return 1

if __name__ == "__main__":
    exit(main())

# On pourrait aussi adapter la méthode pour faire une analyse des agences une par une
# si la liste passée en argument contient plus d'un élément.
# Cependant, l'analyse en elle-même est facultative (il s'agit surtout de faire le clustering)