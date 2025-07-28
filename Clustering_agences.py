import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans
from Code_analyse_OOP import DataCharger, BasicStats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional, int, bool
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE 
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

    def remplissage_data(self):  # Devrait permettre de créer la donnée nécessaire à passer aux algos
        dataset = DataCharger(self.filepath)
        liste_agences = dataset.index.tolist()
        lignes  = []
        for agence in liste_agences:
            class_data = DataCharger(self.filepath, agence, 2024)
            data_agence = BasicStats(class_data)
            lignes.append(data_agence.data_retrieval_clustering)
        self.data = pd.DataFrame(lignes)
        self.data.set_index("code_agence", inplace = True)
        return self.data
    
    def save_data(self, new_filepath : int): # Pour enregistrer une bonne fois pour toutes
        self.data.to_csv(new_filepath, index = True)

    def load_data(self):
        self.data = pd.read_csv(self.new_filepath, index_col = 0)

    # A partir d'ici, on suppose disposer de la donnée adéquate pour lancer un algorithme de clustering
    # Cad : on suppose disposer d'un dataframe avec le code_agence en index et des features en colonne
    # On risque cependant d'avoir à faire de la réduction de dimension pour éviter trop de problèmes...

    def heatmap_features(self):
        plt.figure(figsize = (15,15))
        sns.heatmap(self.data.corr(), cmap = 'coolwarm', center = 0, annot = True, fmt = ".2f")
        plt.show()   # On visualise les corrélations potentielles pour éviter trop de features

    def scaling_data(self):
          scaler = StandardScaler()
          data_scaled = scaler.fit_transform(self.data)
          self.data_scaled = pd.DataFrame(data_scaled, columns = self.data.columns, index = self.data.index)


    def remove_feature(self, column):
        self.data = self.data.drop(columns = column)  # Si une feature est inutile

    def remove_or_not(self):  # A modifier impérativement...
        # Pour décider s'il faut enlever des features ou pas (suivant la heatmap)
        assert self.data is not None
        remove = bool(input("Entrez 'True' si vous voulez retirer des variable, 'False' sinon"))
        if not isinstance(remove, bool):
            raise ValueError("La valeur rentrée doit être booléenne")
        if remove: 
            input_user = input("Entrez la liste des variables à supprimer avant clustering: ")
            retiring = [str(elem.strip()) for elem in input_user.split(',')]
            if not all(retired in self.data.columns for retired in retiring):
                raise ValueError("Certaines colonnes ne sont pas présentes dans self.data")
            else:
                self.data.drop(columns = retiring, inplace = True)
        

    def apply_PCA(self, n_components : Optional[int] = None, var_kept : Optional[float] = None):
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
        silhouette_scores = []
        assert self.data_scaled is not None, "Il faut d'abord standardiser les données avec StandardScaler"
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

    def find_best_k(self, max_k=10):  # Fonction à vérifier...
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
        self.data["cluster_kmeans"] = cluster_labels
        group_cluster_kmeans = self.data.groupby("cluster_kmeans")
        for cluster_id, group in group_cluster_kmeans:  # renvoie une liste des clusters avec les agences
            print(f"\nCluster {cluster_id}: ")
            print(group.index.tolist())  
        return group_cluster_kmeans
    
    def treemap_clusters_kmeans(self):
        assert "cluster_kmeans" in self.data.columns, "Il faut d'abord appliquer k-means"
        clusters = self.data.groupby("cluster_kmeans")
        sizes = [len(group) for _,group in clusters]
        labels = [
        f"Cluster {cid}\n" + "\n".join(group.index.astype(str).tolist())
        for cid, group in clusters
        ]
        plt.figure(figsize=(12, 6))
        squarify.plot(sizes=sizes, label=labels, alpha=0.8)
        plt.axis('off')
        plt.title("Treemap des clusters et agences associées")
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
        return best_method, best_cluster
    

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




# Méthodes pour la visualisation (avec PCA, t-SNE):

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

    def DBSCAN():
        pass

    

# Agrégation des méthodes de clustering:

    def agreg_clustering(self):  # A compléter pour obtenir une méthode d'analyse
        if not self.already_created:
            self.remplissage_data()
            self.save_data(self.new_filepath)
            self.load_data()
        else:
            self.load_data()
        self.heatmap_features()
        self.remove_or_not()
        self.scaling_data()

    




