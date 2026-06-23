import numpy as np
import pandas as pd
import scanpy as sc
from libpysal.weights import KNN
from esda import Moran_BV
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SpatialAutoCorr:
    def __init__(self, adata_rna, adata_atac, n_neighbors=6, verbose=True):
        """
        adata_rna: AnnData with RNA data
        adata_atac: AnnData with ATAC data
        n_neighbors: neighbors for spatial graph
        verbose: print progress
        """
        self.adata_rna = adata_rna
        self.adata_atac = adata_atac
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        if not np.array_equal(adata_rna.obs_names, adata_atac.obs_names):
            if self.verbose:
                print("Aligning spots between RNA and ATAC...")
            self.adata_atac = adata_atac[self.adata_rna.obs_names].copy()
        coords = self.adata_rna.obsm['spatial']
        self.knn_w = KNN.from_array(coords, k=self.n_neighbors)
        self.knn_w.transform = 'R'
        if self.verbose:
            print(f"Spatial weights built: {self.n_neighbors}-NN.")

    def _extract_score(self, adata, features):
        X = adata[:, features].X
        arr = X.toarray() if hasattr(X, 'toarray') else X
        return np.asarray(arr, float).mean(axis=1)

    def compute_global_bivariate(self, topic_dict, permutations=999):
        """
        Returns DataFrame of global Moran_BV (I, z_score, p_value) per topic.
        """
        results = {}
        for topic, sets in topic_dict.items():
            if self.verbose:
                print(f"Global BV Moran for {topic}...")
            y = self._extract_score(self.adata_rna, sets['genes'])
            z = self._extract_score(self.adata_atac, sets['peaks'])
            mbv = Moran_BV(y, z, self.knn_w, permutations=permutations)
            results[topic] = {'I': mbv.I, 'z_score': mbv.z_sim, 'p_value': mbv.p_z_sim}
        return pd.DataFrame(results).T

    def compute_topic_pairwise_global(self, topic, topic_dict, top_n=20, permutations=999, output_path=None):
        """
        Compute global Moran_BV for each pair among top genes and peaks of a topic.

        Parameters:
        - topic: name of the topic
        - topic_dict: dict mapping topic to {'genes': list, 'peaks': list}
        - top_n: number of top features to include
        - permutations: number of permutations for Moran_BV
        - output_path: optional path (without extension) to save results as a single CSV

        Returns:
        - I_mat, z_mat, p_mat: DataFrames of global Moran's I, z-score, and p-value
        """
        genes = topic_dict[topic]['genes'][:top_n]
        peaks = topic_dict[topic]['peaks'][:top_n]
        combined = genes + peaks
        # Initialize matrices
        I_mat = pd.DataFrame(index=combined, columns=combined, dtype=float)
        z_mat = pd.DataFrame(index=combined, columns=combined, dtype=float)
        p_mat = pd.DataFrame(index=combined, columns=combined, dtype=float)
        # Compute pairwise statistics
        for feat1 in combined:
            arr1 = self._extract_score(self.adata_rna if feat1 in genes else self.adata_atac, [feat1])
            for feat2 in combined:
                arr2 = self._extract_score(self.adata_rna if feat2 in genes else self.adata_atac, [feat2])
                mbv = Moran_BV(arr1, arr2, self.knn_w, permutations=permutations)
                I_mat.loc[feat1, feat2] = mbv.I
                z_mat.loc[feat1, feat2] = mbv.z_sim
                p_mat.loc[feat1, feat2] = mbv.p_z_sim
        # Save to single CSV if requested
        if output_path:
            records = []
            for f1 in combined:
                for f2 in combined:
                    records.append({
                        'feature1': f1,
                        'feature2': f2,
                        'I': I_mat.loc[f1, f2],
                        'z_score': z_mat.loc[f1, f2],
                        'p_value': p_mat.loc[f1, f2]
                    })
            df_out = pd.DataFrame.from_records(records)
            # ensure directory exists
            dirpath = os.path.dirname(output_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            df_out.to_csv(f"{output_path}/spatial_corr.csv", index=False)
            if self.verbose:
                print(f"Saved pairwise global BV Moran results to {output_path}/spatial_corr.csv")
        return I_mat, z_mat, p_mat

    def compute_topic_list_pairwise_global(self, topics, topic_dict, top_n=20, permutations=999, output_path=None):
        """
        Compute global Moran_BV for each pair among top genes and peaks from a list of topics.

        Parameters:
        - topics: list of topic names
        - topic_dict: dict mapping topic to {'genes': list, 'peaks': list}
        - top_n: number of top features to include per topic
        - permutations: number of permutations for Moran_BV
        - output_path: optional directory to save a single CSV file

        Returns:
        - I_mat, z_mat, p_mat: DataFrames of global Moran's I, z-score, and p-value
        """
        if self.verbose:
            print(f"Combining features from topics: {topics}")

        all_genes, all_peaks = [], []

        for topic in topics:
            if topic not in topic_dict:
                continue
            all_genes.extend(topic_dict[topic]['genes'][:top_n])
            all_peaks.extend(topic_dict[topic]['peaks'][:top_n])

        # 去重后合并
        # genes = list(set(all_genes))
        # peaks = list(set(all_peaks))
        # combined = genes + peaks
        genes = all_genes
        peaks = all_peaks

        combined = []

        for topic in topics:
            if topic not in topic_dict:
                continue
            top_genes = topic_dict[topic]['genes'][:top_n]
            top_peaks = topic_dict[topic]['peaks'][:top_n]
            combined.extend(top_genes)
            combined.extend(top_peaks)

        if self.verbose:
            print(f"Total features: {len(combined)} (genes: {len(genes)}, peaks: {len(peaks)})")

        # 初始化结果矩阵
        I_mat = pd.DataFrame(index=combined, columns=combined, dtype=float)
        z_mat = pd.DataFrame(index=combined, columns=combined, dtype=float)
        p_mat = pd.DataFrame(index=combined, columns=combined, dtype=float)

        for feat1 in combined:
            arr1 = self._extract_score(self.adata_rna if feat1 in genes else self.adata_atac, [feat1])
            for feat2 in combined:
                arr2 = self._extract_score(self.adata_rna if feat2 in genes else self.adata_atac, [feat2])
                mbv = Moran_BV(arr1, arr2, self.knn_w, permutations=permutations)
                I_mat.loc[feat1, feat2] = mbv.I
                z_mat.loc[feat1, feat2] = mbv.z_sim
                p_mat.loc[feat1, feat2] = mbv.p_z_sim

        # 保存为 CSV
        if output_path:
            records = []
            for f1 in combined:
                for f2 in combined:
                    records.append({
                        'feature1': f1,
                        'feature2': f2,
                        'I': I_mat.loc[f1, f2],
                        'z_score': z_mat.loc[f1, f2],
                        'p_value': p_mat.loc[f1, f2]
                    })
            df_out = pd.DataFrame.from_records(records)
            os.makedirs(output_path, exist_ok=True)
            df_out.to_csv(os.path.join(output_path, "spatial_corr.csv"), index=False)
            if self.verbose:
                print(f"Saved results to {output_path}/spatial_corr.csv")

        return I_mat, z_mat, p_mat

    def plot_pairwise_global_heatmap(self, I_mat, p_mat=None, cmap='RdBu_r', figsize=(6, 5.5), output_path=None, show=True):
        """
        Plot heatmap of pairwise global Moran_BV I values.
        If z_mat and p_mat provided, annotates cells as "I\n(p)".
        """
        annot = None
        if p_mat is not None:
            # annot = I_mat.round(2).astype(str) + "\n(" + p_mat.round(2).astype(str) + ")"
            annot = I_mat.round(2).astype(str)
        plt.figure(figsize=figsize)
        sns.set(font='Arial')
        ax = sns.heatmap(I_mat.astype(float), cmap=cmap, center=0,
                         annot=annot, fmt='', cbar_kws={'label': "Bivariate Moran’s I"},)
        # ax.set_title("Pairwise Global BV Moran's I", fontsize=12)
        x_labels = [lbl.capitalize() for lbl in I_mat.columns]
        y_labels = [lbl.capitalize() for lbl in I_mat.index]
        # x_labels = list(I_mat.columns)
        # y_labels = list(I_mat.index)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=12)
        ax.set_yticklabels(y_labels, rotation=0, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(f"{output_path}/spatial_corr.pdf", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
