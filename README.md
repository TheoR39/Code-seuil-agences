# Code-seuil-agences
Code for gathering, cleaning, visualizing, clustering and modeling data about agencies

- Classe_stats.ipynb:
Fichier temporaire, contenant la troisième classe (assez longue) du fichier Code_analyse_OOP, en cours de construction. 

- Fichier Code_analyse_OOP.ipynb:
Contient 3 classes de manipulation des données, dont 2 classes de preprocessing. La première classe de preprocessing, PreprocessingRawData, a pour but de nettoyer les données (issues de Dremio), et de filtrer sur les données utiles à l'analyse (opérations en agence qui influent sur son niveau de cash).
La seconde classe de cette nature, DataCharger, a pour but de préparer les données avant de les passer à la dernière classe (ajout de colonnes significatives pour la modélisation, filtrage par année / agence).
Enfin, la dernière classe, BasicStats, a quant à elle pour objectif de calculer une myriade de statistiques descriptives (moyenne, médiane, écart-type...) sur les retraits, versements et flux nets journaliers, d'afficher leurs distributions respectives, de calculer les quantiles, et de donner une première estimation du seuil en se basant sur le nombre de rupture.
Une fonction permet d'estimer la probabilité de rupture de l'agence par bootstrap, avec un intervalle de confiance à 95%, pour un seuil donné. Elle permet aussi de récupérer des features pour chaque agence afin d'appliquer des algorithmes de clustering.

- Fichier Code_analyse_OOP.py:
Comme on ne peut pas importer plusieurs classes d'un fichier Jupyter, mais seulement d'un fichier Python, il contient en substance le même code que son homologue jupytérien, mais permettant aux autres fichiers d'accéder aux classes qu'il contient.

- Fichier Clustering_agences.ipynb:
En se basant sur les features calculées pour chaque agence, le fichier doit permettre d'essayer trois algorithmes différents de clustering: K-Means, Hierarchical Clustering et DBSCAN (afin de comparer les clusters). Le but est de grouper entre elles les agences qui ont le même comportement vis-à-vis des opérations clientèles (en termes de volume d'agence, de nombres d'opérations, de montants retirés, de similarité des flux résultants...).

- Fichier Code_simulation.ipynb:
Certainement mal nommé, il s'agit ici de tenter de formaliser la recherche du seuil de chaque agence (ou de chaque cluster) à l'aide d'un problème d'optimisation (stochastique). Après avoir identifié les coûts principaux associés à une valeur de seuil (coût de refinancement ou d'opportunité, coût (fixe) de réapprovisionnement, coût de rupture), et pris en compte comme pénalités (modulables à l'aide de coefficient), on tentera si possible d'appliquer une résolution par méthode SAA couplée au choix avec un algorithme classique d'optimisation (après lissage de la fonction coût) ou solveur MILP, ce qui permettra de comparer aux valeurs statistiques trouvées précédemment.
