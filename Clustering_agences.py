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
from Code_analyse_OOP import DataCharger, BasicStats
from sklearn.preprocessing import StandardScaler
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
        self.filepath = filepath  # Filepath de la donnée complète
        self.new_filepath = new_filepath
        self.data = None  # On commence par créer un DataFrame vide
        self.data_scaled = None
        self.best_nb_k_means = None
        self.pca_object = None
        self.applied_pca = None
        self.already_created = already_created
        self.best_eps = None
        self.best_params_dbscan = None

    def remplissage_data(self): 
        '''Critique: création des données nécessaires aux algos de clustering'''

        dataset = DataCharger(self.filepath) # Chargement des données complètes nettoyées
        liste_agences = dataset.index.tolist()
        lignes  = []
        for agence in liste_agences:
            class_data = DataCharger(filepath = self.filepath, code = agence, annee = 2024)
            data_agence = BasicStats(class_data)
            lignes.append(data_agence.data_retrieval_clustering())
        self.data = pd.DataFrame(lignes)
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

    # A partir d'ici, on suppose disposer de la donnée adéquate pour lancer un algorithme de clustering
    # Cad : on suppose disposer d'un dataframe avec le code_agence en index et des features en colonne
    # On risque cependant d'avoir à faire de la réduction de dimension pour éviter trop de problèmes...

    def heatmap_features(self):
        '''Visualisation des features corrélées avec une heatmap'''

        plt.figure(figsize = (15,15))
        sns.heatmap(self.data.corr(), cmap = 'coolwarm', center = 0, annot = True, fmt = ".2f")
        plt.show()   # On visualise les corrélations potentielles pour éviter trop de features


    def scaling_data(self):
          '''Normalisation des données avec StandardScaler pour le clustering'''

          scaler = StandardScaler()
          data_scaled = scaler.fit_transform(self.data)
          self.data_scaled = pd.DataFrame(data_scaled, columns = self.data.columns, index = self.data.index)


    def remove_feature(self, column):
        self.data = self.data.drop(columns = column)  # Si une feature est inutile


    def remove_or_not(self):  
        '''Demande à l'utilisateur s'il souhaite retirer des features avant PCA et/ou clustering'''

        if self.data is None:
            raise ValueError("self.data doit impérativement être initialisé")
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
        

    def apply_PCA(self, n_components : Optional[int] = None, var_kept : Optional[float] = None):
        '''Application d'une PCA sur les données normalisées, stockées dans self.scaled_data'''

        assert self.data_scaled is not None, "Les données doivent impérativement être standardisées"
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

        if self.data_scaled is None:
            raise ValueError("Il faut commencer par normaliser les données")
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

        if self.data_scaled is None:
            raise ValueError("Il faut tout d'abord normaliser les données")
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
        if self.data_scaled is None:
            raise ValueError("Il faut d'abord normaliser les données : self.data_scaled est vide")
        # Calcul des silhouette score
        silhouette_scores = []
        inertias = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data_scaled)
            silhouette_scores.append(silhouette_score(self.data_scaled, labels))
            inertias.append(kmeans.inertia_)
        best_k = silhouette_scores.index(max(silhouette_scores)) + 2
        self.best_nb_k_means = best_k
        print(f"Meilleur nombre de clusters estimé (silhouette) : {best_k}")
        return best_k


    def clustering_k_means(self, pca: Optional[bool] = False, random_state: Optional[int] = 42):
        '''Création des clusters sur la base de la méthode k-means (avec ou sans PCA)'''

        k = self.best_nb_k_means
        assert k is not None, "Définissez le nombre optimal de clusters à créer"
        kmeans_final = KMeans(n_clusters = k, random_state = random_state)
        if not pca:
            assert self.data_scaled is not None, "Il faut d'abord standardiser les données avec StandardScaler"
            kmeans_final.fit(self.data_scaled)
        else:
            assert self.applied_pca is not None, "Il faut d'abord appliquer la PCA aux données"
            kmeans_final.fit(self.applied_pca)
        cluster_labels = kmeans_final.labels_
        if not pca:
            self.data["cluster_kmeans"] = cluster_labels
            group_cluster_kmeans = self.data.groupby("cluster_kmeans")
        else:
            self.applied_pca["cluster_kmeans_pca"] = cluster_labels
            group_cluster_kmeans = self.data.groupby("cluster_kmeans_pca")
        for cluster_id, group in group_cluster_kmeans:  # renvoie une liste des clusters avec les agences
            print(f"\nCluster {cluster_id}: ")
            print(group.index.tolist())  
        return group_cluster_kmeans
    

    def treemap_clusters_kmeans(self, use_pca : Optional[bool] = False):
        '''Visualisation des clusters k-means par treemap'''

        # Choix de la colonne et vérification de la présence dans les données
        data_used = self.applied_pca if use_pca else self.scaled_data
        if data_used is None:
            raise ValueError("Il faut normaliser et/ou appliquer une PCA aux données au préalable")
        col_name = "cluster_kmeans_pca" if data_used == self.applied_pca else "cluster_kmeans"
        if col_name not in data_used.columns:
            raise ValueError(f"Les données ne contiennent pas la colonne {col_name}")
        clusters = self.data_used.groupby(col_name)
        sizes = [len(group) for _,group in clusters]
        labels = [
        f"Cluster {cid}\n" + "\n".join(group.index.astype(str).tolist())
        for cid, group in clusters
        ]
        # Affichage des clusters
        plt.figure(figsize=(12, 6))
        squarify.plot(sizes=sizes, label=labels, alpha=0.8)
        plt.axis('off')
        plt.title(f"Treemap des clusters kmeans {'(avec PCA)' if use_pca else '(sans PCA)'} et agences associées")
        plt.tight_layout()
        plt.show()




# Méthodes pour la clusterisation hiérarchique:

    # La première méthode détermine la méthode la plus susceptible de donner des clusters adaptés
    # Elle se base sur le calcul du coefficient de corrélation cophénétique
    # Plus il est proche de 1, meilleure est la méthode
    def find_best_method(self, pca : Optional[bool] = False):
        data_used = self.applied_pca if pca else self.data_scaled
        assert data_used is not None, "Aucune donnée disponible pour clusteriser"
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
        return best_method, best_score
        

    # La deuxième méthode calcule (automatiquement) le meilleur nombre de clusters en se basant sur le plus gros saut sur le dendrogramme
    # En gros, à cet endroit, le dendrogramme fusionne des groupes très différents, donc on coupe avant.
    def find_best_max_cluster(self, method : Optional[str] = "ward", pca : Optional[bool] = False, return_score : Optional[bool] = False):
        data_used = self.applied_pca if pca else self.data_scaled
        assert data_used is not None, "Aucune donnée disponible pour clusteriser"
        Z = linkage(data_used, method = method)
        distances = Z[:,2]
        deltas = np.diff(distances)
        index_max_jump  = np.argmax(deltas[-10:]) + len(deltas) - 10
        threshold_distance = distances[index_max_jump + 1]
        labels = fcluster(Z, t = threshold_distance, criterion = 'distance')
        nb_clusters = len(set(labels))
        print(f"Nombre optimal estimé de clusters hiérarchiques: {nb_clusters}")
        score = silhouette_score(data_used, labels)
        score = silhouette_score(data_used, labels) if return_score else None
        if return_score:
            print("Silhouette score: ", score)
        return nb_clusters, labels, threshold_distance, score
    

    def select_method_cluster(self, pca : Optional[bool] = False):
        best_method = self.find_best_method(pca)[0]
        best_cluster = self.find_best_max_cluster(pca, method = best_method)[0]
        return best_method, best_cluster   # S'assurer de la logique avec la méthode suivante...
    

    def clustering_hierarchical(self, pca : Optional[bool] = False, method : str = 'ward', dendrogram : Optional[bool] = False,
                                max_cluster : Optional[int] = None, best : Optional[bool] = False):
        data_to_use = self.data_scaled
        if pca:
            data_to_use = self.applied_pca
        assert data_to_use is not None
        if best:
            method, max_cluster = self.select_method_cluster()
        Z = linkage(data_to_use, method = method)
        if dendrogram:
            plt.figure(figsize=(14, 7))
            dendrogram(Z, labels=self.data.index.tolist(), leaf_rotation=90)
            plt.title(f"Dendrogramme du clustering hiérarchique ({'PCA' if pca else 'Standard'}) - Méthode: {method}")
            plt.xlabel("Code agence")
            plt.ylabel("Distance")
            plt.tight_layout()
            plt.show()
        labels = None
        if max_cluster:
            labels = fcluster(Z, max_cluster, criterion='maxclust')
            self.data["cluster_hierarchical"] = labels
            print(f"\nAgences regroupées en {max_cluster} clusters :")
            for cluster_id in range(1, max_cluster + 1):
                membres = self.data[self.data["cluster_hierarchical"] == cluster_id].index.tolist()
                print(f"Cluster {cluster_id} : {membres}")
        return labels
    

    def treemap_clusters_hierarchical(self):
        assert "cluster_hierarchical" in self.data.columns, "Il faut commencer par appliquer une méthode de hierarchical clustering"
        clusters = self.data.groupby("cluster_hierarchical")
        sizes = [len(group) for _, group in clusters]
        labels = [
        f"Cluster {cid}\n" + "\n".join(group.index.astype(str).tolist())
        for cid, group in clusters
        ]
        plt.figure(figsize = (12,6))
        squarify.plot(sizes = sizes, label = labels, alpha = 0.8)
        plt.axis('off')
        plt.title("Treemap des clusters hiérarchiques et de leurs agences")
        plt.tight_layout()
        plt.show()




# Méthodes pour la visualisation (avec PCA, t-SNE):   # Revoir la méthode pour prendre en compte la nouvelle logique

    def tsne_visual_clusters(self, cluster_col : str = "cluster_kmeans", pca : Optional[bool] = False,
                             perplexity  : int = 15, random_state: Optional[int] = 42):
        assert cluster_col in self.data.columns, f"Il faut d'abord appliquer le clustering qui donne la colonne {cluster_col}"
        data_used = self.applied_pca if pca else self.data_scaled
        assert data_used is not None, "Il faut impérativement normaliser les données"
        tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = random_state)
        tsne_result = tsne.fit_transform(data_used)
        df_tsne = pd.DataFrame(tsne_result, columns = ['TSNE1', 'TSNE2'], index = self.data.index)
        df_tsne[cluster_col] = self.data[cluster_col]
        plt.figure(figsize = (10,6))
        sns.scatterplot(data = df_tsne, x='TSNE1', y='TSNE2', hue = cluster_col, palette = 'tab10', s=80)
        plt.title(f"Projection t-SNE des clusters ({cluster_col})")
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
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
        plt.title(f"k-distance graph pour k = {min_samples}")
        plt.xlabel("Points triés")
        plt.ylabel(f"Distance au {min_samples} plus proche voisin")
        plt.grid(True)
        plt.show()

    
    # def retrieve_best_eps(self, best_eps):
    #     '''Retourne le meilleur epsilon en se basant sur le graphe des k-distances'''

    #     assert self.best_eps is None, "best epsilon a déjà été estimé"
    #     self.best_eps = best_eps

    
    def calib_params_dbscan(self, eps_range = np.arange(0.3,3,0.1), min_samples_range = range(3,10),
                            verbose : Optional[bool] = False, pca : Optional[bool] = False):
        '''Grid_search sur le couple (eps,min_samples) pour DBSCAN en se basant sur le silhouette score'''

        data_used = self.applied_pca if pca else self.scaled_data
        if data_used is None:
            raise ValueError("Il faut appliquer la normalisation et/ou une PCA préliminaire(s)")
        best_params = None
        best_score = -1 
        results = []
        for eps, min_samples in itertools.product(eps_range, min_samples_range):
            db = DBSCAN(eps = eps, min_samples = min_samples)
            labels = db.fit_transform(data_used)
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
            self.best_params_dbscan = best_params
            return best_params, best_score, results


    def clustering_dbscan(self, eps : float = 0.5, min_samples : int = 5, pca : Optional[bool] = False,
                          best: Optional[bool] = False):
        '''Application de DBSCAN après application ou non de la PCA'''

        data_used = self.applied_pca if pca else self.data_scaled
        if data_used is None:
            raise ValueError("Il faut impérativement commencer par normaliser les donnés et/ou appliquer une PCA")
        if best:
            eps = self.best_params_dbscan[0]
            min_samples = self.best_params_dbscan[1]
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        labels = dbscan.fit_predict(data_used)
        self.data["cluster_dbscan"] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  #nb_clusters
        n_noise = list(labels).count(-1)  # nb_agences en solo
        print(f"Nombre de clusters créés par DBSCAN: {n_clusters}")
        print(f"Nombre d'agences solitaires (classées comme bruit): {n_noise}")
        # Visualisation des regroupements par les clusters créés
        for cluster_id in sorted(set(labels)):
            membres = self.data[self.data["cluster_dbscan"] == cluster_id].index.tolist()  # Liste des membres pour un cluster donné
            nom = "Bruit" if cluster_id == -1 else f"Cluster {cluster_id}"
            print(f"{nom} : {membres}")
        # Calcul du silhouette_score:
        if n_clusters > 1:
            score = silhouette_score(data_used, labels)
            print(f"Silhouette score pour DBSCAN: {score : .3f}")
        else:
            print("Silhouette score non défini")
        return labels, score
    

    def treemap_clusters_dbscan(self):
        '''Treemap pour l'affichage des clusters'''

        if "cluster_dbscan" not in self.data.columns:
            raise ValueError("Il faut d'abord appliquer DBSCAN")
        clusters = self.data.groupby("cluster_dbscan")
        sizes = [len(group) for _,group in clusters]
        labels = [f"Cluster {cid}" if cid != -1 else "Bruit"
                  + '\n' + '\n'.join(group.index.astype(str).tolist())
                  for cid, group in clusters]
        plt.figure(figsize = (12,6))
        squarify.plot(sizes = sizes, label = labels, alpha = 0.8)
        plt.axis('off')
        plt.title("Treemap des clusters DBSCAN et agences associées")
        plt.tight_layout()
        plt.show()

    

# Agrégation des méthodes de clustering:

    def check_before_apply(self):
        '''Pour s'assurer que les données soient présentes avant d'effectuer les méthodes de cluster'''

        if self.data is None:
            self.load_data()
        if self.data_scaled is None:
            self.scaling_data()    


    def agreg_clustering(self, pca : Optional[bool] = False, use_pca : Optional[bool] = False,
                         return_score : Optional[bool] = False, best : Optional[bool] = True):  # A compléter pour obtenir une méthode d'analyse
        '''Méthode qui applique les 3 clusters aux données construites'''

        if not self.already_created:
            self.remplissage_data()
            self.save_data(self.new_filepath)
        self.apply_kmeans(pca, use_pca)
        self.apply_hierarchical_clustering(pca, return_score, best)


    def apply_kmeans(self, pca : Optional[bool] = False, use_pca : Optional[bool] = False):
        '''Appelle les différentes méthodes pour effectuer une clusterisation k-means'''

        self.check_before_apply()
        self.elbow_method_k_means()
        self.silhouette_score_k_means()
        self.find_best_k()
        self.clustering_k_means(pca)
        self.treemap_clusters_kmeans(use_pca)
        self.tsne_visual_clusters(cluster_col = "cluster_kmeans", pca = pca)  # Attention à la logique avec t-SNE


    def apply_hierarchical_clustering(self, pca: Optional[bool] = False, return_score : Optional[bool] = False,
                                      best : Optional[bool] = True):
        '''Appelle les différentes méthodes pour effectuer une clusterisation hiérarchique'''

        self.check_before_apply()
        self.find_best_method(pca)
        self.find_best_max_cluster(pca = pca, return_score = return_score)
        self.clustering_hierarchical(pca = pca, best = best)
        self.treemap_clusters_hierarchical()
        

    # On pourrait aussi utiliser UMAP apparemment (à la place ou en plus de t-SNE)
    # Il faudrait utiliser des loggings à la place des prints.






    




