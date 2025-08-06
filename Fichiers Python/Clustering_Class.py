import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import os
import itertools
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans
from Code_analyse_OOP_term import DataCharger, BasicStats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import Optional
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE 
from sklearn.neighbors import NearestNeighbors
# from collections import defaultdict


class Clustering_agences:
    # On suppose que complete_data contient les données de toutes les agences sur 2024
    # On suppose également que 'code_agence' est mis en index
    # On suppose enfin que complete_data a été nettoyé au préalable avec PreprocessingRawData
    # Dans un premier temps, on a besoin de créer un nouveau DataFrame avec les features retenue
    def __init__(self, new_filepath : str, filepath : str, already_created : Optional[bool] = False):
        '''Constructeur de la classe Clustering_agences'''

        # Stockage des données:
        self.filepath = filepath  # Filepath de la donnée complète
        self.new_filepath = new_filepath  # Pour stocker les données nécessaires au clustering
        self.data = None  # On commence par créer un DataFrame vide
        self.data_scaled = None  # Pour normaliser les données
        self.already_created = already_created  # Si les données ont déjà été créées
        self.random_state = 42   # graine pour l'initialisation aléatoire

        # Pour stocker la PCA:
        self.pca_object = None   # Stockage de l'objet PCA
        self.applied_pca = None  # Stockage de l'application de PCA aux données

        # Attributs pour k-means:
        self.best_nb_kmeans = None  # Pour stocker le meilleur nombre de clusters pour k-means
        self.kmeans_best_score = None  # Stockage du silhouette score pour le meilleur k-means

        # Attributs pour Hierarchical clustering:
        self.hierarchical_best_method = None   # Stockage de la meilleure méthode pour hierarchical clustering
        self.hierarchical_best_max_clusters = None  # Stockage du meilleur max_cluster
        self.hierarchical_best_score = None  # Stockage du silhouette score pour le meilleur cluster
        
        # Attributs pour DBSCAN:
        self.dbscan_best_eps = None  # Stockage du meilleur epsilon pour DBSCAN
        self.dbscan_best_min_samples = None   # Stockage du meilleur min_samples pour DBSCAN
        self.dbscan_best_score = None  # Stockage du silhouette score pour le meilleur DBSCAN


    def remplissage_data(self): 
        '''Critique: création des données nécessaires aux algos de clustering'''

        if not hasattr(self,'filepath') or not isinstance(self.filepath, str):
            raise AttributeError("L'attribut 'filepath' n'existe pas ou est incorrect")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Aucun fichier trouvé à l'adresse {self.filepath}")
        dataset = DataCharger(self.filepath) # Chargement des données complètes nettoyées
        dataset = dataset.preparer_donnees()[0]  # Peut-être pas besoin...
        if dataset is None or dataset.empty:
            raise ValueError("Le dataset chargé est vide")
        if dataset.index is None or dataset.index.empty:
            raise ValueError("Le dataset ne contient pas d'index")
        liste_agences = sorted(dataset.index.unique().tolist())
        lignes  = []
        for agence in liste_agences:
            try:
                class_data = DataCharger(filepath = self.filepath, code = agence, annee = 2024)
                class_data.assignation_simple()
                print(f"Agence {agence} — class_data type: {type(class_data)}")  # Débogage
                if hasattr(class_data, 'data'):
                    print(f"Agence {agence} — class_data.data type: {type(class_data.data)}")
                data_agence = BasicStats(class_data)
                print(f"Agence {agence} — data_agence.data type: {type(data_agence.data)}")
                lignes.append(data_agence.data_retrieval_clustering())
            except Exception as e:
                print(f"Erreur lors du chargement des données de l'agence {agence} : {e}")
                continue
        if not lignes:
            raise ValueError("Aucune donnée n'a pu être traitée")
        self.data = pd.DataFrame(lignes)
        if 'code_agence' not in self.data.columns:
            raise ValueError("Impossible de créer l'index: 'code'agence' n'existe pas")
        self.data.set_index("code_agence", inplace = True)
        return self.data
    

    def save_data(self, overwrite : Optional[bool] = False): # Pour enregistrer une bonne fois pour toutes
        '''Sauvegarde des données créées pour le clustering à l'adresse self.new_filepath'''

        if not self.new_filepath.lower().endswith('.csv'):
            raise ValueError("Le chemin de sauvegarde ne construit pas un fichier au format csv")
        if self.data is None:
            raise ValueError("Aucune donnée disponible pour une sauvegarde")
        if os.path.exists(self.new_filepath):
            if overwrite:
                print("Ecrasement du fichier existant...")

            else:
                print("Le fichier existe déjà, il n'y a pas besoin de re-sauvegarder")
                print("Si vous voulez remplacer le fichier existant, relancez la fonction avec l'argument 'overwrite' à True")
                return
        self.data.to_csv(self.new_filepath, index = True)
        print(f"Données sauvegardées vers {self.new_filepath}")


    def load_data(self):
        '''Chargement des données nécessaires aux algos de clustering'''
        
        # Vérifications sur le chemin d'accès
        if not os.path.exists(self.new_filepath):
            raise FileNotFoundError("Le fichier de données n'a pas encore été créé")
        if not os.path.isfile(self.new_filepath):
            raise ValueError("Le chemin d'accès spécifié ne pointe pas vers un fichier")
        if not self.new_filepath.lower().endswith('csv'):
            raise ValueError("Le fichier n'est pas au format csv")
        # Chargement des données:
        try:
            self.data = pd.read_csv(self.new_filepath, index_col = 0)
            print(f"Données chargées depuis {self.new_filepath}")
        except Exception as e:
            raise IOError(f"Erreur lors du chargement des fichiers depuis {self.new_filepath}")
        

    def check_data(self, scaled : Optional[bool] = False, pca : Optional[bool] = False):
        if not hasattr(self,'data') or self.data is None:
            raise ValueError("self.data doit impérativement être initialisé")
        if scaled:
            if not hasattr(self,'data_scaled'):
                raise AttributeError("'self.data_scaled' n'existe pas")
            if self.data_scaled is None:
                raise ValueError("Il faut impérativement normaliser les données")
        if pca:
            if not hasattr(self,'applied_pca'):
                raise AttributeError("'self.applied_pca' n'existe pas")
            if self.applied_pca is None:
                raise ValueError("Il faut appliquer la PCA à self.data_scaled")
        if self.data.empty:
            raise ValueError("self.data est vide")
        if scaled:
            if self.data_scaled.empty:
                raise ValueError("self.data_scaled est vide")
        if pca:
            if self.applied_pca.empty:
                raise ValueError("self.applied_pca est vide")
            

    def set_random_seed(self, seed: int):
        if not hasattr(self,'random_state'):
            raise AttributeError("self n'a pas d'attribut 'random_state'")
        else:
            self.random_state = seed
        

    # A partir d'ici, on suppose disposer de la donnée adéquate pour lancer un algorithme de clustering
    # Cad : on suppose disposer d'un dataframe avec le code_agence en index et des features en colonne
    # On risque cependant d'avoir à faire de la réduction de dimension pour éviter trop de problèmes...

    def heatmap_features(self):
        '''Visualisation des features corrélées avec une heatmap'''

        self.check_data()
        plt.figure(figsize = (15,15))
        sns.heatmap(self.data.corr(), cmap = 'coolwarm', center = 0, annot = True, fmt = ".2f")
        plt.show()   # On visualise les corrélations potentielles pour éviter trop de features


    def scaling_data(self, robust: Optional[bool] = False):
        '''Normalisation des données avec StandardScaler pour le clustering'''

        self.check_data()
        if robust:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(self.data)
        else:
            scaler = RobustScaler()
            data_scaled = scaler.fit_transform(self.data)
        self.data_scaled = pd.DataFrame(data_scaled, columns = self.data.columns, index = self.data.index)


    def remove_feature(self, column):
        '''Permet à l'utilisateur d'enlever une feature'''

        self.check_data()
        self.data = self.data.drop(columns = column)  # Si une feature est inutile


    def remove_or_not(self):  
        '''Demande à l'utilisateur s'il souhaite retirer des features avant PCA et/ou clustering'''

        self.check_data()
        while True:
            user_input = input("Voulez-vous retirer des variables ? (True/False): " )
            if user_input in ['true', 'True', 't', 'oui', 'yes','y','Y']:
                remove = True
                break
            elif user_input in ['false', 'False', 'f', 'non', 'no','n','N']:
                remove = False
                break
            else:
                print("Réponse invalide. Répondez par 'True' ou 'False'")
        if remove:
            while True:
                user_input_2 = input("Entrez la liste des variables à supprimer, separées par des virgules:")
                retiring = [elem.strip() for elem in user_input_2.split(',') if elem.strip()]
                if not retiring:
                    print("Liste vide, entrez au moins une variable")
                    continue 
                missing_cols = [col for col in retiring if col not in self.data.columns]
                if missing_cols:
                    print(f"Ces colonnes ne sont pas présentes dans self.data: {missing_cols}")
                else:
                    self.data.drop(columns = retiring, inplace = True)
                    print(f"Colonnes supprimées: {retiring}")
                    break
        else:
            print("Aucune variable supprimée")
            

    def remove_highly_correlated(self, threshold : Optional[float] = 0.90):
        '''On enlève automatiquement les features très corrélées (ici avec un coef > 0.90)'''

        self.check_data()
        corr_matrix = self.data.corr().abs()
        upper  = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        if to_drop:
            print(f"Suppression des colonnes fortement corrélées (> {threshold}): {to_drop}")
            self.data.drop(columns = to_drop, inplace = True)
        else:
            print("Pas de colonnes fortement corrélées trouvées")
        return self.data


    def apply_PCA(self, n_components : Optional[int] = None, var_kept : Optional[float] = None):
        '''Application d'une PCA sur les données normalisées, stockées dans self.data_scaled'''

        self.check_data(True)
        if var_kept is not None:
            pca = PCA(n_components = var_kept)
        elif n_components is not None:
            pca = PCA(n_components = n_components)
        else:
            raise ValueError("Au moins l'un des deux arguments 'n_components' ou 'var_kept' doit être non vide")
        reduced = pca.fit_transform(self.data_scaled)
        column_names = [f"PCA{i+1}" for i in range(reduced.shape[1])]
        self.applied_pca = pd.DataFrame(reduced, index = self.data.index, columns = column_names)
        self.pca_object = pca
        return self.applied_pca, self.pca_object


# Méthodes pour clusteriser au sens de k-means:

    def elbow_method_k_means(self):  # On cherche à déterminer le nombre optimal de clusters
        '''Détermination du nombre de clusters k_means avec elbow_method'''

        self.check_data(True)
        inertias = []
        K = range(1, 11)  # nombre de clusters à tester
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data_scaled)
            inertias.append(kmeans.inertia_)  # inertia = somme des distances au centre du cluster
        plt.figure(figsize=(8, 5))
        plt.plot(K, inertias, 'bo-')
        plt.xlabel('Nombre de clusters')
        plt.ylabel('Inertie (Within-Cluster Sum of Squares)')
        plt.title("Méthode du coude pour déterminer le nombre optimal de clusters")
        plt.grid(True)
        plt.show()


    def silhouette_score_k_means(self):
        '''Evaluation des clusters k_means par silhouette score'''

        self.check_data(True)
        silhouette_scores = []
        for k in range(2, 11):  # 2 clusters minimum
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data_scaled)
            score = silhouette_score(self.data_scaled, labels)
            silhouette_scores.append(score)
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, 11), silhouette_scores, 'go-')
        plt.xlabel('Nombre de clusters')
        plt.ylabel('Silhouette Score')
        plt.title("Score de silhouette en fonction du nombre de clusters")
        plt.grid(True)
        plt.show()


    def find_best_k(self, max_k=15):  
        '''Détermination du meilleur nombre de clusters par silhouette score'''

        # Vérification préliminaire
        self.check_data(True)
        # Calcul des silhouette score
        silhouette_scores = []
        inertias = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data_scaled)
            silhouette_scores.append(silhouette_score(self.data_scaled, labels))
            inertias.append(kmeans.inertia_)
        best_k = silhouette_scores.index(max(silhouette_scores)) + 2
        self.best_nb_kmeans = best_k
        print(f"Meilleur nombre de clusters estimé (silhouette) : {best_k}")
        return best_k


    def clustering_k_means(self, pca: Optional[bool] = False):
        '''Création des clusters sur la base de la méthode k-means (avec ou sans PCA)'''

        self.check_data(True, pca)
        k = self.best_nb_kmeans
        assert k is not None, "Définissez d'abord le nombre optimal de clusters à créer"
        data_used = self.applied_pca if pca else self.data_scaled
        kmeans_final = KMeans(n_clusters = k, random_state = self.random_state)
        kmeans_final.fit_transform(data_used)
        cluster_labels = kmeans_final.labels_
        final_score = silhouette_score(data_used, cluster_labels)
        if not pca:
            self.data["cluster_kmeans"] = cluster_labels
            group_cluster_kmeans = self.data.groupby("cluster_kmeans")
        else:
            data_used["cluster_kmeans_pca"] = cluster_labels
            group_cluster_kmeans = data_used.groupby("cluster_kmeans_pca")
        print(f"Silhouette score atteint (pour k = {k}): {final_score}")
        for cluster_id, group in group_cluster_kmeans:  # renvoie une liste des clusters avec les agences
            print(f"\nCluster {cluster_id}: ")
            print(group.index.tolist())  
        self.kmeans_best_score = final_score
        return group_cluster_kmeans, final_score
    

    def treemap_clusters_kmeans(self, pca : Optional[bool] = False):
        '''Visualisation des clusters k-means par treemap'''

        # Choix de la colonne et vérification de la présence dans les données
        self.check_data(True, pca)
        data_used = self.applied_pca if pca else self.data
        col_name = "cluster_kmeans_pca" if data_used is self.applied_pca else "cluster_kmeans"
        if col_name not in data_used.columns:
            raise ValueError(f"Les données ne contiennent pas la colonne {col_name}")
        clusters = data_used.groupby(col_name)
        sizes = [len(group) for _,group in clusters]
        labels = [
        f"Cluster {cid}\n" + "\n".join(group.index.astype(str).tolist())
        for cid, group in clusters
        ]
        # Affichage des clusters
        plt.figure(figsize=(14, 12))
        squarify.plot(sizes=sizes, label=labels, alpha=0.8)
        plt.axis('off')
        plt.title(f"Treemap des clusters kmeans {'(avec PCA)' if pca else '(sans PCA)'} et agences associées")
        plt.tight_layout()
        plt.show()




# Méthodes pour la clusterisation hiérarchique:

    # La première méthode détermine la méthode la plus susceptible de donner des clusters adaptés
    # Elle se base sur le calcul du coefficient de corrélation cophénétique
    # Plus il est proche de 1, meilleure est la méthode
    def find_best_method(self, pca : Optional[bool] = False):
        '''On utilise le coefficient cophénétique pour trouver la meilleure méthode'''

        self.check_data(True, pca)
        data_used = self.applied_pca if pca else self.data_scaled
        methods = ['ward', 'complete', 'average', 'single']  # Méthodes possibles pour hierarchical clustering
        best_method = None
        best_score = -1  # Pire score de corrélation possible
        for method in methods:
            Z = linkage(data_used, method = method)
            coph_corr,_ = cophenet(Z, pdist(data_used))
            if coph_corr > best_score:
                best_score = coph_corr
                best_method = method
        print("Meilleure méthode estimée par corrélation cophénétique: ", best_method)
        print("Coefficient de corrélation cophénétique atteint par la méthode: ", round(best_score,4))
        self.hierarchical_best_method = best_method
        return best_method, best_score
        

    # La deuxième méthode calcule (automatiquement) le meilleur nombre de clusters en se basant sur le plus gros saut sur le dendrogramme
    # En gros, à cet endroit, le dendrogramme fusionne des groupes très différents, donc on coupe avant.
    def find_best_max_cluster(self, method : Optional[str] = "ward", pca : Optional[bool] = False):
        '''Méthode pour trouver le bon nombre de clusters pour hierarchical clustering'''

        self.check_data(True, pca)
        data_used = self.applied_pca if pca else self.data_scaled
        Z = linkage(data_used, method = method)
        distances = Z[:,2]
        deltas = np.diff(distances)
        index_max_jump  = np.argmax(deltas[-10:]) + len(deltas) - 10
        threshold_distance = distances[index_max_jump + 1]
        labels = fcluster(Z, t = threshold_distance, criterion = 'distance')
        nb_clusters = len(set(labels))
        print(f"Nombre optimal estimé de clusters hiérarchiques: {nb_clusters}")
        if nb_clusters >= 2:
            score = silhouette_score(data_used, labels)
        else: 
            score = None
        print(f"Silhouette score pour hierarchical clustering, (méthode = {method}, nb_clusters = {nb_clusters}): {score}")
        self.hierarchical_best_max_clusters = nb_clusters
        return nb_clusters, labels, threshold_distance, score
    

    def clustering_hierarchical(self, pca : Optional[bool] = False, method : str = 'ward', dendrogram : Optional[bool] = False,
                                max_cluster : Optional[int] = None, best : Optional[bool] = True):  # A modifier pour avoir même logique que kmeans
        '''Mise en oeuvre de hierarchical clustering sur les données'''

        self.check_data(True, pca)
        data_used = self.applied_pca if pca else self.data_scaled
        if best:
            if self.hierarchical_best_method is None:
                raise ValueError("L'attribut 'self.hierarchical_best_method' est vide")
            elif self.hierarchical_best_max_clusters is None:
                raise ValueError("L'attribut 'self.hierarchical_best_max_clusters' est vide")
            try:
                method = self.hierarchical_best_method
                max_cluster = self.hierarchical_best_max_clusters
            except Exception as e:
                print(f"Erreur lors de l'accès aux meilleurs paramètres : {e}")
        Z = linkage(data_used, method = method)
        if dendrogram:
            plt.figure(figsize=(14, 7))
            dendrogram(Z, labels=self.data.index.tolist(), leaf_rotation=90)
            plt.title(f"Dendrogramme du clustering hiérarchique ({'PCA' if pca else 'Standard'}) - Méthode: {method}")
            plt.xlabel("Code agence")
            plt.ylabel("Distance")
            plt.tight_layout()
            plt.show()
        labels = fcluster(Z, max_cluster, criterion='maxclust')
        if pca:
            data_used["cluster_hierarchical_pca"] = labels
            group_cluster_hierarchical = data_used.groupby("cluster_hierarchical_pca")
        else:
            self.data["cluster_hierarchical"] = labels
            group_cluster_hierarchical = self.data.groupby("cluster_hierarchical")
        if len(set(labels)) >= 2:
            final_score = silhouette_score(data_used, labels)
        else:
            final_score = None
        print(f"Silhouette score atteint (pour méthode = {method}, nb_cluster = {max_cluster}): {final_score}")
        for cluster_id, group in group_cluster_hierarchical:  # renvoie une liste des clusters avec les agences
            print(f"\nCluster {cluster_id}: ")
            print(group.index.tolist())  
        self.hierarchical_best_score = final_score
        return group_cluster_hierarchical, final_score

    

    def treemap_clusters_hierarchical(self, pca: Optional[bool] = False):
        '''Visualisation des hierarchical clusters par treemap'''

        self.check_data(True, pca)
        data_used = self.applied_pca if pca else self.data
        col_name = "cluster_hierarchical_pca" if pca else "cluster_hierarchical"
        if col_name not in data_used.columns:
            raise ValueError(f"Les données ne contiennent pas la colonne {col_name}")
        clusters = data_used.groupby(col_name)
        sizes = [len(group) for _, group in clusters]
        labels = [
        f"Cluster {cid}\n" + "\n".join(group.index.astype(str).tolist())
        for cid, group in clusters
        ]
        plt.figure(figsize = (14,12))
        squarify.plot(sizes = sizes, label = labels, alpha = 0.8)
        plt.axis('off')
        plt.title("Treemap des clusters hiérarchiques et de leurs agences")
        plt.tight_layout()
        plt.show()



    
# Méthodes pour la clusterisation DBSCAN:

    def plot_k_distance(self, min_samples : int = 5, pca : Optional[bool] = False):
        '''Méthode d'estimation du paramètre epsilon de DBSCAN (à min_samples donné)'''

        data_used = self.applied_pca if pca else self.data_scaled
        if data_used is None:
            raise ValueError("Il faut d'abord normaliser les données et/ou appliquer la PCA")
        neigh = NearestNeighbors(n_neighbors = min_samples)
        nbrs = neigh.fit(data_used)
        distances, _ = nbrs.kneighbors(data_used)
        k_distances = np.sort(distances[:,-1])
        # Plot du graphe:
        plt.figure(figsize = (8,5))
        plt.plot(k_distances)
        plt.title(f"k-distance graph pour k = {min_samples}")
        plt.xlabel("Points triés")
        plt.ylabel(f"Distance au {min_samples} plus proche voisin")
        plt.grid(True)
        plt.show()

    
    def calib_params_dbscan(self, eps_range = np.arange(0.3,3,0.1), min_samples_range = range(3,10),
                            verbose : Optional[bool] = False, pca : Optional[bool] = False):
        '''Grid_search sur le couple (eps,min_samples) pour DBSCAN en se basant sur le silhouette score'''

        self.check_data(True, pca)
        data_used = self.applied_pca if pca else self.data_scaled
        best_params = None
        best_score = -1 
        results = []
        for eps, min_samples in itertools.product(eps_range, min_samples_range):
            db = DBSCAN(eps = eps, min_samples = min_samples)
            labels = db.fit_predict(data_used)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue
            try:
                score = silhouette_score(data_used, labels)
                results.append((eps,min_samples,score))
                if score > best_score:
                    best_score = score
                    best_params = (eps,min_samples)
                if verbose:
                    print(f"eps = {eps:.2f}, min_samples = {min_samples}, nb_clusters = {n_clusters}")
            except:
                continue
        if best_params is None:
            print("Aucun clustering DBSCAN satisfaisant trouvé")
        else:
            print(f"Meilleurs paramètres par grid-search: eps = {best_params[0]:.2f}, min_samples = {best_params[1]}")
            self.dbscan_best_eps = best_params[0]
            self.dbscan_best_min_samples = best_params[1]
            return best_params, best_score, results


    def clustering_dbscan(self, eps : float = 0.5, min_samples : int = 5, pca : Optional[bool] = False,
                          best: Optional[bool] = True):
        '''Application de DBSCAN après application ou non de la PCA'''

        # A modifier pour créer le cas échéant la colonne cluster_dbscan_pca

        self.check_data(True, pca)
        data_used = self.applied_pca if pca else self.data_scaled
        if best:
            if self.dbscan_best_eps is None:
                raise ValueError("L'attribut 'self.dbscan_best_eps' est vide")
            elif self.dbscan_best_min_samples is None:
                raise ValueError("L'attribut 'self.dbscan_best_eps' est vide")
        try:
            eps = self.dbscan_best_eps
            min_samples = self.dbscan_best_min_samples
        except Exception as e:
            print(f"Erreur lors de l'accès aux meilleurs paramètres: {e}")
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        labels = dbscan.fit_predict(data_used)
        if pca:
            data_used["cluster_dbscan_pca"] = labels
        else:
            self.data["cluster_dbscan"] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  #nb_clusters
        n_noise = list(labels).count(-1)  # nb_agences en solo
        print(f"Nombre de clusters créés par DBSCAN: {n_clusters}")
        print(f"Nombre d'agences solitaires (classées comme bruit): {n_noise}")
        # Visualisation des regroupements par les clusters créés
        for cluster_id in sorted(set(labels)):
            if pca:
                membres = data_used[data_used["cluster_dbscan_pca"] == cluster_id].index.tolist()  # Liste des membres pour un cluster donné
            else:
                membres = self.data[self.data["cluster_dbscan"] == cluster_id].index.tolist()
            nom = "Bruit" if cluster_id == -1 else f"Cluster {cluster_id}"
            print(f"{nom} : {membres}")
        # Calcul du silhouette_score:
        if n_clusters > 1:
            score = silhouette_score(data_used, labels)
            self.dbscan_best_score = score
            print(f"Silhouette score pour DBSCAN: {score : .3f}")
        else:
            score = None
            print("Silhouette score non défini")
        return labels, score
    

    def treemap_clusters_dbscan(self, pca: Optional[bool] = False):
        '''Treemap pour l'affichage des clusters DBSCAN'''

        self.check_data(True, pca)
        data_used = self.applied_pca if pca else self.data
        col_name = 'cluster_dbscan_pca' if pca else 'cluster_dbscan'
        if col_name not in data_used.columns:
            raise ValueError(f"Les données ne contiennent pas la colonne {col_name}")
        clusters = data_used.groupby(col_name)
        sizes = [len(group) for _,group in clusters]
        labels = [f"Cluster {cid}" if cid != -1 else "Bruit"
                  + '\n' + '\n'.join(group.index.astype(str).tolist())
                  for cid, group in clusters]
        plt.figure(figsize = (14,12))
        squarify.plot(sizes = sizes, label = labels, alpha = 0.8)
        plt.axis('off')
        plt.title("Treemap des clusters DBSCAN et agences associées")
        plt.tight_layout()
        plt.show()




# Méthodes pour la visualisation (avec t-SNE):   # Revoir la méthode pour prendre en compte la nouvelle logique

    def tsne_visual_clusters(self, choice_method, pca : Optional[bool] = False,
                             perplexity  : int = 15):
        '''Visualisation par emploi de t-SNE'''    # A modifier en profondeur...
        
        methods = ["kmeans", "hierarchical", "dbscan"]
        if choice_method not in methods:
            raise ValueError(f"Le choix de la méthode doit être inclus dans la liste {methods}")
        cluster_col_map = {"kmeans": ["cluster_kmeans","cluster_kmeans_pca"],
                       "hierarchical": ["cluster_hierarchical", "cluster_hierarchical_pca"],
                       "dbscan": ["cluster_dbscan", "cluster_dbscan_pca"]}
        self.check_data(True, pca)
        data_used = self.applied_pca if pca else self.data
        cluster_col = cluster_col_map[choice_method][1 if pca else 0]
        if cluster_col not in data_used.columns:
            raise ValueError(f"La colonne {cluster_col} n'est pas présente dans les données")
        tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = self.random_state)
        tsne_result = tsne.fit_transform(data_used.drop(columns = [cluster_col], errors = 'ignore'))
        df_tsne = pd.DataFrame(tsne_result, columns = ['TSNE1', 'TSNE2'], index = data_used.index)
        df_tsne[cluster_col] = data_used[cluster_col]
        plt.figure(figsize = (10,6))
        sns.scatterplot(data = df_tsne, x='TSNE1', y='TSNE2', hue = cluster_col, palette = 'tab10', s=80)
        plt.title(f"Projection t-SNE des clusters ({cluster_col})")
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()



# Agrégation des méthodes de clustering:

    def check_before_apply(self):
        '''Pour s'assurer que les données soient présentes avant d'effectuer les méthodes de cluster'''

        if self.data is None:
            self.load_data()
        if self.data_scaled is None:
            self.scaling_data()   


    def compare_clusters_silhouette_score(self, pca: Optional[bool] = False):
        '''Méthode de comparaison des clusters suivant le silhouette score'''

        required_scores = {"kmeans": getattr(self, "kmeans_best_score", None),
                           "hierarchical": getattr(self, "hierarchical_best_score", None),
                           "dbscan": getattr(self, "dbscan_best_score", None)}
        missing = [method for method, score in required_scores.items() if score is None]
        if missing:
            raise ValueError(f"Les méthodes suivantes n'ont pas de score associé: {missing}")
        print("Silhouette scores: ")
        for method, score in required_scores.items():
            print(f" - {method.capitalize()} : {score:.4f}")
        best_method = max(required_scores, key = required_scores.get)
        best_score = required_scores[best_method]
        print(f"\n Meilleure méthode d'après le silhouette score: {best_method}: {best_score}")
        return best_method, best_score


    def agreg_clustering(self, pca : Optional[bool] = False,
                         best : Optional[bool] = True, robust: Optional[bool] = True):  # A compléter pour obtenir une méthode d'analyse
        '''Méthode qui applique les 3 clusters aux données construites'''

        if not self.already_created:
            self.remplissage_data()
            self.save_data(self.new_filepath)
        else:
            self.load_data()
        self.remove_highly_correlated()
        if pca:
            self.scaling_data(robust = robust)
            self.apply_PCA(var_kept = 0.95)
        self.apply_kmeans(pca)
        self.apply_hierarchical_clustering(pca = pca, best = best)
        self.apply_dbscan(pca = pca, best = best)
        self.compare_clusters_silhouette_score()


    def apply_kmeans(self, pca : Optional[bool] = False):
        '''Appelle les différentes méthodes pour effectuer une clusterisation k-means'''

        self.check_before_apply()
        self.elbow_method_k_means()
        self.silhouette_score_k_means()
        self.find_best_k()
        self.clustering_k_means(pca)
        self.treemap_clusters_kmeans(pca)
        self.tsne_visual_clusters(choice_method = "kmeans", pca = pca)  # Attention à la logique avec t-SNE
        # Effectivement, il va falloir modifier la dernière fonction pour kmeans....


    def apply_hierarchical_clustering(self, pca: Optional[bool] = False, best : Optional[bool] = True):
        '''Appelle les différentes méthodes pour effectuer une clusterisation hiérarchique'''

        self.check_before_apply()
        self.find_best_method(pca)
        self.find_best_max_cluster(pca = pca)
        self.clustering_hierarchical(pca = pca, best = best)
        self.treemap_clusters_hierarchical(pca)
        self.tsne_visual_clusters(choice_method = "hierarchical", pca = pca)


    def apply_dbscan(self, pca : Optional[bool] = False, 
                     eps_range = np.arange(0.3,3,0.1), min_samples_range = range(3,10),
                            verbose : Optional[bool] = False, best : Optional[bool] = True):
        '''Application des différentes méthodes pour le clustering DBSCAN'''

        self.check_before_apply()
        self.plot_k_distance(min_samples = 5, pca = pca)
        self.calib_params_dbscan(eps_range, min_samples_range, verbose, pca)
        self.clustering_dbscan(pca = pca, best = best)
        self.treemap_clusters_dbscan(pca)
        self.tsne_visual_clusters(choice_method = "dbscan", pca = pca)


    # On pourrait aussi utiliser UMAP apparemment (à la place ou en plus de t-SNE)
    # Il faudrait utiliser des loggings à la place des prints.






    




