"""backend/clustering.py

Core clustering implementations used by the backend API.

Contains:
- normalize_data: simple min-max scaler
- KMeans: lightweight K-means implementation returning labels, centroids and inertia
- DBSCAN: density-based clustering using scipy.spatial.cKDTree for neighbor queries
- evaluation utilities: silhouette, Davies-Bouldin, Calinski-Harabasz

Notes:
- Implementations are intentionally simple and readable for educational use.
- For production or better stability use scikit-learn's implementations.
"""

# IMPORT LIBRARIES
# ============================================================================
import numpy as np  # NumPy untuk operasi array dan perhitungan matematika
from typing import List, Dict, Any  # Type hints untuk dokumentasi tipe data parameter dan return value
from sklearn.metrics import silhouette_score, davies_bouldin_score  # Metrik evaluasi clustering dari scikit-learn
from scipy.spatial import cKDTree  # KD-Tree dari scipy untuk pencarian neighbor yang efisien O(log n)
import time  # Module time untuk mengukur durasi eksekusi proses







# FUNGSI HELPER: EUCLIDEAN DISTANCE
# ============================================================================
def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:  # Fungsi menghitung jarak Euclidean antara dua titik
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((point1 - point2) ** 2))  # Rumus: sqrt(sum((p1-p2)^2)) untuk setiap dimensi






# FUNGSI HELPER: NORMALISASI DATA
# ============================================================================
def normalize_data(data: np.ndarray) -> np.ndarray:  # Fungsi normalisasi data menggunakan Min-Max Scaling
    """Normalize data using min-max scaling"""
    min_vals = np.min(data, axis=0)  # Cari nilai minimum untuk setiap kolom (axis=0 berarti per kolom)
    max_vals = np.max(data, axis=0)  # Cari nilai maksimum untuk setiap kolom
    range_vals = max_vals - min_vals  # Hitung range (max - min) untuk setiap kolom
    range_vals[range_vals == 0] = 1  # Jika range = 0 (semua nilai sama), ganti dengan 1 untuk menghindari pembagian dengan 0
    
    return (data - min_vals) / range_vals  # Rumus Min-Max: (x - min) / (max - min) untuk normalisasi ke range [0, 1]







# KELAS K-MEANS CLUSTERING
# ============================================================================
class KMeans:  # Kelas untuk implementasi algoritma K-means clustering
    """K-means clustering algorithm"""
    
    def __init__(self, k: int = 3, max_iterations: int = 100, tol: float = 1e-4):  # Added tolerance parameter for early stopping
        self.k = k  # Simpan jumlah cluster yang diinginkan
        self.max_iterations = max_iterations  # Simpan jumlah iterasi maksimal untuk konvergensi
        self.tol = tol  # Tolerance untuk convergence detection
    
    def fit(self, data: np.ndarray) -> Dict[str, Any]:  # Fungsi untuk melatih model K-means pada data
        """Fit K-means model to data"""
        n, d = data.shape  # n = jumlah data points, d = jumlah dimensi/fitur
        
        indices = np.random.choice(n, self.k, replace=False)  # Pilih k indeks random tanpa penggantian
        centroids = data[indices].copy()  # Ambil k data points sebagai centroid awal
        
        labels = np.zeros(n, dtype=int)  # Array untuk menyimpan label cluster setiap data point
        inertia = 0  # Variabel untuk menyimpan total jarak intra-cluster
        iterations = 0  # Counter untuk menghitung jumlah iterasi yang dilakukan
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):  # Loop hingga max_iterations tercapai
            iterations += 1  # Increment counter iterasi
            
            # Calculate distances using broadcasting: (n, d) - (k, d) -> (n, k)
            distances = np.sqrt(np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))
            
            new_labels = np.argmin(distances, axis=1)  # Assign setiap point ke centroid terdekat
            inertia = np.sum(np.min(distances ** 2, axis=1))  # Hitung inertia
            
            if np.array_equal(labels, new_labels):  # Jika label tidak berubah
                labels = new_labels
                break
            
            labels = new_labels  # Update labels untuk iterasi berikutnya
            
            new_centroids = np.zeros((self.k, d))
            for i in range(self.k):  # Loop untuk setiap cluster
                cluster_mask = labels == i  # Boolean mask untuk cluster i
                if np.sum(cluster_mask) > 0:  # Jika cluster tidak kosong
                    new_centroids[i] = np.mean(data[cluster_mask], axis=0)  # Hitung mean
                else:  # Jika cluster kosong
                    new_centroids[i] = data[np.random.randint(n)]  # Pilih random point
            
            centroids = new_centroids  # Update centroids untuk iterasi berikutnya
            
            if (iteration + 1) % max(1, self.max_iterations // 10) == 0:
                elapsed = time.time() - start_time
                print(f"[v0] K-means iteration {iteration + 1}/{self.max_iterations} - Inertia: {inertia:.2f} - Time: {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        print(f"[v0] K-means completed in {total_time:.2f}s with {iterations} iterations")
        
        return {  # Return dictionary berisi hasil clustering
            'labels': labels.tolist(),  # Label cluster untuk setiap data point
            'centroids': centroids.tolist(),  # Koordinat centroid akhir
            'inertia': float(inertia),  # Total intra-cluster distance
            'iterations': iterations,  # Jumlah iterasi yang dilakukan sampai konvergen
            'processing_time': float(total_time)  # Added processing time
        }










# KELAS DBSCAN CLUSTERING - SIMPLIFIED & OPTIMIZED
# ============================================================================
class DBSCAN:  # Kelas untuk implementasi algoritma DBSCAN dengan optimasi KD-Tree
    """DBSCAN clustering algorithm - Simplified and optimized version"""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):  
        # Constructor dengan parameter eps (radius) dan min_samples (minimum points)
        self.eps = eps  
        # Simpan epsilon (radius untuk mencari neighbors)
        self.min_samples = min_samples  
        # Simpan minimum jumlah points untuk membentuk core point
    
    def fit(self, data: np.ndarray) -> Dict[str, Any]:  
        # Fungsi untuk melatih model DBSCAN pada data
        """
        Fit DBSCAN model to data
        
        Args:
            data: Input data array (n_samples, n_features)
        
        Returns:
            Dictionary containing clustering results
        """
        n = data.shape[0]
        labels = np.full(n, -1, dtype=int)  # -1 = noise/unvisited
        cluster_id = 0
        
        start_time = time.time()
        
        try:
            print(f"[v0] Building KD-Tree for {n} points...")
            tree = cKDTree(data)
            
            print(f"[v0] Querying neighbors with eps={self.eps}...")
            neighbors_list = tree.query_ball_point(data, self.eps)
            
            print(f"[v0] KD-Tree built successfully")
            
        except Exception as e:
            print(f"[v0] ERROR building KD-Tree: {str(e)}")
            raise ValueError(f"Failed to build KD-Tree: {str(e)}")
        
        print(f"[v0] Starting DBSCAN clustering...")
        cluster_start = time.time()
        
        try:
            for i in range(n):
                # Skip jika sudah dikunjungi
                if labels[i] != -1:
                    continue
                
                neighbors = neighbors_list[i]
                
                # Jika bukan core point, mark as noise
                if len(neighbors) < self.min_samples:
                    labels[i] = -1
                else:
                    # Expand cluster dari core point ini
                    self._expand_cluster(i, neighbors, labels, neighbors_list, cluster_id)
                    cluster_id += 1
                
                # Progress indicator
                if (i + 1) % max(1, n // 10) == 0:
                    print(f"[v0] Processed {i + 1}/{n} points...")
        
        except Exception as e:
            print(f"[v0] ERROR during clustering: {str(e)}")
            raise ValueError(f"Error during clustering: {str(e)}")
        
        cluster_time = time.time() - cluster_start
        total_time = time.time() - start_time
        
        num_clusters = len(set(labels[labels != -1]))
        num_noise = np.sum(labels == -1)
        clustering_ratio = ((n - num_noise) / n * 100) if n > 0 else 0
        
        print(f"[v0] DBSCAN completed in {total_time:.2f}s")
        print(f"[v0] Clusters: {num_clusters}, Noise: {num_noise}, Ratio: {clustering_ratio:.1f}%")
        
        return {
            'labels': labels.tolist(),
            'num_clusters': int(num_clusters),
            'num_noise': int(num_noise),
            'total_points': int(n),
            'clustering_ratio': float(clustering_ratio),
            'processing_time': float(total_time)
        }
    
    def _expand_cluster(self, point_idx: int, neighbors: List[int], labels: np.ndarray,
                       neighbors_list: List[List[int]], cluster_id: int) -> None:
        """
        Expand cluster from a core point
        
        Improved cluster expansion with better handling of core points
        and more efficient queue management using deque for better performance
        """
        from collections import deque
        
        # Assign point ke cluster
        labels[point_idx] = cluster_id
        
        # Use deque untuk efficient queue operations (O(1) append dan popleft)
        queue = deque()
        
        # Add neighbors ke queue, skip yang sudah punya label
        for neighbor in neighbors:
            if neighbor != point_idx and labels[neighbor] == -1:
                queue.append(neighbor)
        
        # Process queue
        while queue:
            current_idx = queue.popleft()
            
            # Jika sudah punya label (bukan noise), skip
            if labels[current_idx] != -1:
                continue
            
            # Assign ke cluster
            labels[current_idx] = cluster_id
            
            # Jika core point, tambah neighbors ke queue
            current_neighbors = neighbors_list[current_idx]
            if len(current_neighbors) >= self.min_samples:
                for neighbor in current_neighbors:
                    if labels[neighbor] == -1:  # Only add unvisited points
                        queue.append(neighbor)


# FUNGSI EVALUASI METRIK CLUSTERING
# ============================================================================
def calculate_evaluation_metrics(data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:  # Fungsi untuk menghitung metrik evaluasi clustering
    """Calculate clustering evaluation metrics"""
    valid_mask = labels != -1  # Buat mask untuk points yang bukan noise (label != -1)
    
    if np.sum(valid_mask) < 2:  # Jika kurang dari 2 valid points
        return {  # Return error message
            'silhouette_score': None,  # Tidak bisa hitung Silhouette Score
            'davies_bouldin_index': None,  # Tidak bisa hitung Davies-Bouldin Index
            'calinski_harabasz_index': None,  # Tidak bisa hitung Calinski-Harabasz Index
            'error': 'Tidak cukup cluster yang valid untuk evaluasi. Coba ubah parameter eps atau min_samples.',  # Pesan error
            'valid_points': int(np.sum(valid_mask)),  # Jumlah valid points
            'total_points': len(labels),  # Total points
            'noise_points': int(np.sum(labels == -1))  # Jumlah noise points
        }
    
    valid_data = data[valid_mask]  # Ambil hanya valid data points (exclude noise)
    valid_labels = labels[valid_mask]  # Ambil hanya valid labels (exclude noise)
    
    unique_clusters = len(set(valid_labels))  # Hitung jumlah unique clusters
    if unique_clusters < 2:  # Jika hanya 1 cluster (Silhouette Score memerlukan minimal 2 clusters)
        return {  # Return error message
            'silhouette_score': None,  # Silhouette Score memerlukan minimal 2 clusters
            'davies_bouldin_index': None,
            'calinski_harabasz_index': None,
            'error': f'Hanya ditemukan {unique_clusters} cluster. Silhouette Score memerlukan minimal 2 cluster.',  # Pesan error
            'valid_points': int(np.sum(valid_mask)),
            'total_points': len(labels),
            'noise_points': int(np.sum(labels == -1))
        }
    
    try:  # Try-except untuk error handling
        silhouette = silhouette_score(valid_data, valid_labels)  # Hitung Silhouette Score (range: -1 to 1, lebih tinggi lebih baik)
    except:  # Jika error
        silhouette = None  # Set ke None
    
    try:
        davies_bouldin = davies_bouldin_score(valid_data, valid_labels)  # Hitung Davies-Bouldin Index (lebih rendah lebih baik)
    except:
        davies_bouldin = None
    
    try:
        calinski_harabasz = calinski_harabasz_score(valid_data, valid_labels)  # Hitung Calinski-Harabasz Index (lebih tinggi lebih baik)
    except:
        calinski_harabasz = None
    
    return {  # Return hasil metrik evaluasi
        'silhouette_score': float(silhouette) if silhouette is not None else None,  # Silhouette Score (convert ke float jika tidak None)
        'davies_bouldin_index': float(davies_bouldin) if davies_bouldin is not None else None,  # Davies-Bouldin Index
        'calinski_harabasz_index': float(calinski_harabasz) if calinski_harabasz is not None else None  # Calinski-Harabasz Index
    }














# FUNGSI STATISTIK CLUSTER
# ============================================================================
def get_cluster_statistics(data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:  # Fungsi untuk menghitung statistik setiap cluster
    """Get statistics for each cluster"""
    unique_labels = set(labels)  # Ambil semua unique labels
    if -1 in unique_labels:  # Jika ada noise points (label -1)
        unique_labels.remove(-1)  # Hapus dari set (tidak perlu statistik untuk noise)
    
    stats = {}  # Dictionary untuk menyimpan statistik setiap cluster
    for label in sorted(unique_labels):  # Loop untuk setiap cluster (sorted untuk urutan konsisten)
        cluster_data = data[labels == label]  # Ambil semua data points yang termasuk cluster ini
        stats[f'cluster_{int(label + 1)}'] = {  # Buat entry untuk cluster ini dengan key 'cluster_X' (1-based)
            'size': int(len(cluster_data)),  # Jumlah points dalam cluster
            'mean': cluster_data.mean(axis=0).tolist(),  # Rata-rata (mean) untuk setiap fitur
            'std': cluster_data.std(axis=0).tolist(),  # Standar deviasi untuk setiap fitur
            'min': cluster_data.min(axis=0).tolist(),  # Nilai minimum untuk setiap fitur
            'max': cluster_data.max(axis=0).tolist()  # Nilai maksimum untuk setiap fitur
        }
    
    return stats  # Return dictionary statistik

def get_unnormalized_cluster_averages(original_data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Calculate average attributes per cluster from original (unnormalized) data"""
    unique_labels = set(labels)  # Ambil semua unique labels
    if -1 in unique_labels:  # Jika ada noise points (label -1)
        unique_labels.remove(-1)  # Hapus dari set
    
    averages = {}  # Dictionary untuk menyimpan rata-rata per cluster
    for label in sorted(unique_labels):  # Loop untuk setiap cluster
        cluster_data = original_data[labels == label]  # Ambil data asli untuk cluster ini
        averages[f'cluster_{int(label + 1)}'] = {  # Buat entry dengan key 'cluster_X' (1-based)
            'size': int(len(cluster_data)),  # Jumlah points dalam cluster
            'mean': cluster_data.mean(axis=0).tolist()  # Rata-rata untuk setiap atribut (data asli)
        }
    
    return averages  # Return dictionary rata-rata

def calculate_elbow_curve(data: np.ndarray, k_range: range = range(1, 11)) -> Dict[str, Any]:
    """Calculate inertia for different k values to help determine optimal k"""
    inertias = []
    k_values = list(k_range)
    
    for k in k_values:
        kmeans = KMeans(k=k, max_iterations=100)
        result = kmeans.fit(data)
        inertias.append(result['inertia'])
    
    return {
        'k_values': k_values,
        'inertias': inertias
    }
