import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import traceback
from BasicStats_Class import BasicStats
from Preprocessing_Class import PreprocessingRawData
from Clustering_Class import Clustering_agences
from DataCharger_Class import DataCharger
from Optim_min_Class import Optim_min_threshold

# Attention, il faut adapter les chemins d'accès suivant l'endroit de stockage des données
# Il faudrait changer les args par des fonctions avec argument pour automatiser avec Streamlit (par exemple)

print("Début de l'exécution du script Python")
print(hasattr(DataCharger, 'preparer_donnees'))

def main():
    '''Lancement de toute la pipeline, de la récupération des données au clustering (pour l'instant)'''

    parser = argparse.ArgumentParser(description = "Pipeline d'analyse des données de transactions des agences.")

    # Paramètres de l'analyse:
    parser.add_argument('--preprocess', action = 'store_true', help = 'Lancement du preprocessing des données brutes.')
    parser.add_argument('--clustering', action = 'store_true', help = 'Lancement du clustering.')
    parser.add_argument('--prelim_analysis', action = 'store_true', help = "Lancement de l'analyse statistique.")
    parser.add_argument('--launch_optim_one', action = 'store_true', help = "Lancement de l'optimisation pour une seule agence")
    parser.add_argument('--launch_optim', action = 'store_true', help = "Lancement du programme d'optimisation")
    parser.add_argument('--year', type = int, default = 2024, help = "Année du traitement.")
    parser.add_argument('--years', nargs = '*', type = int, help = "Années de sélection pour l'analyse statistique.")
    parser.add_argument('--agences', nargs = '*', type = int, help = "Liste sélectionnée des agences.")
    parser.add_argument('--aggregate', action = 'store_true', help = "Utilisé en cas de présence de données déjà nettoyées.")
    parser.add_argument('--already_created', action = 'store_true', help = "Les données pour le clustering existent déja.")
    parser.add_argument('--pca', action = 'store_true', help = 'Utilisation ou non de PCA pour le clustering')
    parser.add_argument('--robust', action = 'store_true', help = 'Application ou non de RobustScaler')
    parser.add_argument('--particular_code_agence', type = int, help = "Code agence pour optim d'une seule agence")
    parser.add_argument('--already_created_optim', action = 'store_true', help = "Utilisé en cas de données d'optim déjà recueillies")
    parser.add_argument('--overwrite_optim', action = 'store_true', help = "Relance de la récupération des données d'optimisation")
    args = parser.parse_args()

    # Preprocessing des fichiers:
    raw_dir = f"C:/Users/theob/OneDrive/Documents/INSA/Stage CIH/Code CIH/Data/donnees_agences_{args.year}"  # A adapter (dossier contenant les fichiers pour l'année 2024)
    filepath_raw_csv = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.csv')]
    clean_filepath = "C:/Users/theob/OneDrive/Documents/INSA/Stage CIH/Code CIH/Data/Agences_nettoyees_completes.csv"  # Pour stocker les données nettoyées
    clustering_filepath = "C:/Users/theob/OneDrive/Documents/INSA/Stage CIH/Code CIH/Data/Données_clustering.csv"  # Pour stocker les données du clustering
    data_gathered_optim = "C:/Users/theob/OneDrive/Documents/INSA/Stage CIH/Code CIH/Data/Données_préliminaires_optim.json" # Pour stocker les données préliminaires à l'optimisation
    data_result_optim_one = "C:/Users/theob/OneDrive/Documents/INSA/Stage CIH/Code CIH/Data/Données_finales_optim_agence_par_agence.json"
    data_result_optim = "C:/Users/theob/OneDrive/Documents/INSA/Stage CIH/Code CIH/Data/Données_finales_optim.json" # Pour stocker les résultats de l'optimisation
    if args.years is None:
        args.years = [args.year]


    if args.prelim_analysis and not args.agences:
        raise ValueError("Aucune agence spécifique n'a été sélectionnée.")
    
    try:
        if args.preprocess:
            print(hasattr(DataCharger, 'preparer_donnees'))
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
                if args.agences and len(args.agences) != 1:   # erreur ici !
                    print("L'analyse nécessite de ne sélectionner qu'une seule agence")
                if args.agences and len(args.agences) == 1:
                    print(f"Lancement d'une analyse statistique exploratoire pour l'(les) agence(s) {args.agences} sur l'(les) années {args.years}")
                    choice = DataCharger(filepath = clean_filepath, code = args.agences, annee = args.years)
                    choice.preparer_donnees()
                    data = BasicStats(choice)
                    data.analyse_preliminaire_data()
                    print("Fin de l'analyse exploratoire préliminaire")
                    print("Il est possible d'affiner sur une période donnée en lançant la méthode 'analyse_exploratoire_periodique'") 
            except Exception as e:
                print(f"Erreur lors de l'analyse des données de l'agence {args.agences} : {e}")
        
        if args.clustering:
            if not os.path.exists(clean_filepath):
                raise FileNotFoundError(f"Fichier de données nettoyées introuvable ou inexistant à l'adresse : {clean_filepath}")
            print("Lancement du clustering des agences")
            print("Clustering suivant k-means / hierarchical clustering / DBSCAN")
            clustering = Clustering_agences(new_filepath = clustering_filepath, filepath = clean_filepath,
                                            already_created = args.already_created)
            if not args.pca:
                print("Lancement du clustering sans PCA par défaut: ")
                clustering.agreg_clustering()
            else:
                clustering.agreg_clustering(pca = args.pca)
            print("Clustering terminé!")

        if args.launch_optim_one:
            if not args.particular_code_agence :   
                raise ValueError("Il faut impérativement spécifier l'argument 'agence_optim' ")
            print("Lancement de l'optimisation pour une seule agence")
            optim = Optim_min_threshold(filepath = clean_filepath, filepath_optim = data_gathered_optim,
                                        filepath_optim_one_agency = data_result_optim_one, optim_json = data_result_optim, already_created_optim = args.already_created_optim,
                                        overwrite = args.overwrite_optim)
            optim.optim_one_agency(code_agence = args.particular_code_agence)
            print("Fin de l'optimisation de l'agence")

        if args.launch_optim:
            print("Lancement de l'optimisation des seuils.")
            optim = Optim_min_threshold(filepath = clean_filepath, filepath_optim = data_gathered_optim,
                                    optim_json = data_result_optim, already_created_optim = args.already_created_optim)
            optim.optim_all_agencies()
            print(f"Données d'optimisation recueillies et chargées dans {data_result_optim}")

    except Exception as e:
        print(f"Erreur lors de l'exécution {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

# On pourrait aussi adapter la méthode pour faire une analyse des agences une par une
# si la liste passée en argument contient plus d'un élément.
# Cependant, l'analyse en elle-même est facultative (il s'agit surtout de faire le clustering)
# Il s'agit maintenant surtout de lancer l'optimisation des agences.