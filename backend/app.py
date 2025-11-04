"""backend/app.py

Flask-based backend API for the Food Clustering project.

This module exposes endpoints used by the Streamlit frontend:
- /api/upload-csv : parse uploaded CSV files and return numeric data
- /api/upload-excel : parse uploaded Excel files and return numeric data
- /api/kmeans : run KMeans clustering (accepts normalized or raw data)
- /api/dbscan : run DBSCAN clustering (expects normalized data)

The file also contains helper code to safely call functions from
`backend/clustering.py` and to avoid running the backend inside a
Streamlit process.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import csv
import io
import os
import sys
import clustering


def safe_get_unnormalized_cluster_averages(original_data: np.ndarray, labels: np.ndarray):
    """
    Ambil rata-rata atribut per cluster pada skala asli (tidak dinormalisasi).

    Cara kerja:
    - Jika modul `clustering` menyediakan fungsi `get_unnormalized_cluster_averages`,
      panggil fungsi tersebut dan kembalikan hasilnya.
    - Jika tidak tersedia atau terjadi error, hitung secara lokal:
        * Buat set label unik, abaikan label -1 (noise)
        * Untuk setiap label, ambil baris original_data yang sesuai dan hitung ukuran
          cluster dan nilai rata-ratanya (mean) per atribut.

    Return:
        dict dengan format { 'cluster_1': {'size': int, 'mean': [..]}, ... }
    """
    try:
        fn = getattr(clustering, 'get_unnormalized_cluster_averages', None)
        if callable(fn):
            return fn(original_data, labels)
    except Exception:
        pass

    # Fallback implementation (mirror of expected behavior)
    try:
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        averages = {}
        for label in sorted(unique_labels):
            cluster_data = original_data[labels == label]
            averages[f'cluster_{int(label + 1)}'] = {
                'size': int(len(cluster_data)),
                'mean': cluster_data.mean(axis=0).tolist() if len(cluster_data) > 0 else []
            }
        return averages
    except Exception as e:
        # If fallback fails, return empty dict to avoid crashing the API
        print(f"[v0] safe_get_unnormalized_cluster_averages fallback error: {e}")
        return {}

# Check if running inside Streamlit
if 'streamlit' in sys.modules:
    print("\n" + "="*60)
    print("ERROR: Backend should NOT be run with 'streamlit run'")
    print("="*60)
    print("\nCorrect way to run backend:")
    print("  python backend/app.py")
    print("\nCorrect way to run frontend:")
    print("  streamlit run frontend/app.py")
    print("\nRun them in SEPARATE terminals!")
    print("="*60 + "\n")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def read_root():
    """
    Endpoint root sederhana untuk cek kesehatan API.

    Mengembalikan JSON singkat yang menandakan server backend hidup.
    """
    return jsonify({"message": "Food Clustering API is running"})

@app.route('/api/kmeans', methods=['POST'])
def kmeans_cluster():
    """
        Endpoint untuk menjalankan K-means clustering.

        Input (JSON di request body):
            - data: list of list (array 2D). Bisa data mentah atau sudah dinormalisasi.
            - original_data (opsional): list of list, data pra-normalisasi untuk tujuan
                menampilkan nilai 'Asli' pada frontend.
            - k (opsional): jumlah cluster (default 3)
            - max_iterations (opsional): batas iterasi K-means (default 100)

        Alur:
            1. Baca payload dan ubah ke numpy array
            2. Jika original_data disediakan, anggap `data` sudah dinormalisasi.
                 Jika tidak, normalisasikan `data` menggunakan fungsi `clustering.normalize_data`.
            3. Validasi parameter `k`.
            4. Jalankan KMeans.fit pada data ter-normalisasi.
            5. Hitung metrik evaluasi dan statistik cluster, serta rata-rata pada skala asli.
            6. Kembalikan hasil sebagai JSON.

        Response:
            JSON berisi keys: labels, centroids, inertia, iterations, processing_time,
            serta tambahan metrics, statistics, unnormalized_averages.
        """
    try:
        data = request.json.get('data')
        k = request.json.get('k', 3)
        max_iterations = request.json.get('max_iterations', 100)
        
        # The frontend may send either:
        # - raw/original data in the 'data' field (then backend should normalize it), OR
        # - normalized data in the 'data' field together with the original pre-normalized
        #   values in the optional 'original_data' field.
        payload = request.json
        data = np.array(payload.get('data'))

        # If frontend provided original (unnormalized) values, use them for 'Asli' calculations.
        if payload.get('original_data') is not None:
            original_data = np.array(payload.get('original_data'))
            # Assume 'data' is already normalized in this case
            normalized_data = data
        else:
            # No original provided: assume 'data' is raw and normalize here
            original_data = data.copy()
            normalized_data = clustering.normalize_data(data)
        
        if k < 1:
            return jsonify({"error": "k harus minimal 1"}), 400
        if k > len(data):
            return jsonify({"error": f"k tidak boleh lebih besar dari jumlah data ({len(data)})"}), 400
        
        kmeans = clustering.KMeans(k=k, max_iterations=max_iterations)
        result = kmeans.fit(normalized_data)

        labels = np.array(result['labels'])
        metrics = clustering.calculate_evaluation_metrics(normalized_data, labels)
        stats = clustering.get_cluster_statistics(normalized_data, labels)
        unnormalized_averages = safe_get_unnormalized_cluster_averages(original_data, labels)
        
        result['metrics'] = metrics
        result['statistics'] = stats
        result['unnormalized_averages'] = unnormalized_averages
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/kmeans-elbow', methods=['POST'])
def kmeans_elbow():
    """
        Endpoint untuk menghitung kurva elbow (inertia vs k).

        Input (JSON):
            - data: list of list (array 2D)
            - max_k (opsional): batas atas k yang diuji (default 10)

        Alur:
            - Normalisasi data bila perlu (jika original_data tidak diberikan)
            - Batasi max_k ke nilai yang wajar (maks 20 atau jumlah baris)
            - Jalankan KMeans untuk tiap k di range(1, max_k+1) dan kumpulkan inertia
            - Kembalikan dictionary { 'k_values': [...], 'inertias': [...] }
        """
    try:
        data = request.json.get('data')
        max_k = request.json.get('max_k', 10)
        
        payload = request.json
        data = np.array(payload.get('data'))

        # If frontend provided original (unnormalized) values, use them for 'Asli' calculations.
        if payload.get('original_data') is not None:
            original_data = np.array(payload.get('original_data'))
            # Assume 'data' is already normalized in this case
            normalized_data = data
        else:
            original_data = data.copy()
            normalized_data = clustering.normalize_data(data)
        
        # Limit max_k to reasonable value
        max_k = min(max_k, min(20, len(data)))
        
        print(f"[v0] Calculating elbow curve for k=1 to {max_k}")
        
        elbow_result = clustering.calculate_elbow_curve(normalized_data, k_range=range(1, max_k + 1))
        
        return jsonify(elbow_result)
    except Exception as e:
        print(f"[v0] Elbow calculation error: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/dbscan', methods=['POST'])
def dbscan_cluster():
    """
        Endpoint untuk menjalankan DBSCAN clustering.

        Input (JSON):
            - data: list of list (array 2D). Disarankan sudah dinormalisasi.
            - original_data (opsional): list of list untuk perhitungan nilai asli.
            - eps (opsional): radius neighbor (default 0.5)
            - min_samples (opsional): minimal poin untuk core point (default 5)

        Alur:
            - Baca payload, jika perlu normalisasi data dengan clustering.normalize_data
            - Validasi eps dan min_samples
            - Buat instance DBSCAN dan jalankan fit
            - Hitung metrik, statistik, dan rata-rata pada skala asli
            - Jika rasio yang ter-cluster kecil, tambahkan peringatan dalam response
        """
    try:
        # Accept payload and support optional original_data (same as /api/kmeans)
        payload = request.json
        eps = payload.get('eps', 0.5)
        min_samples = payload.get('min_samples', 5)

        # data should be the normalized array (frontend sends normalized 'data')
        data = np.array(payload.get('data'))

        if len(data) > 10000:
            print(f"[v0] Warning: Large dataset ({len(data)} points). DBSCAN will use sampling for efficiency.")
            print(f"[v0] Estimated processing time: {len(data) / 1000:.1f} seconds")

        print(f"[v0] Starting DBSCAN with eps={eps}, min_samples={min_samples}, data_size={len(data)}")

        # If frontend provided original (unnormalized) values, use them for 'Asli' calculations.
        if payload.get('original_data') is not None:
            original_data = np.array(payload.get('original_data'))
            # Assume 'data' is already normalized in this case
            normalized_data = data
        else:
            original_data = data.copy()
            normalized_data = clustering.normalize_data(data)
        
        if eps <= 0:
            return jsonify({"error": "eps harus lebih besar dari 0"}), 400
        if min_samples < 2:
            return jsonify({"error": "min_samples harus minimal 2"}), 400
        
        dbscan = clustering.DBSCAN(eps=eps, min_samples=min_samples)
        result = dbscan.fit(normalized_data)
        
        print(f"[v0] DBSCAN completed in {result.get('processing_time', 'unknown')}s. Clusters: {result['num_clusters']}, Noise: {result['num_noise']}")
        
        labels = np.array(result['labels'])
        metrics = clustering.calculate_evaluation_metrics(normalized_data, labels)
        stats = clustering.get_cluster_statistics(normalized_data, labels)
        unnormalized_averages = safe_get_unnormalized_cluster_averages(original_data, labels)
        
        result['metrics'] = metrics
        result['statistics'] = stats
        result['unnormalized_averages'] = unnormalized_averages
        
        if result['clustering_ratio'] < 10:
            result['warning'] = f"Hanya {result['clustering_ratio']:.1f}% data yang ter-cluster. Coba naikkan eps atau turunkan min_samples."
        
        return jsonify(result)
    except Exception as e:
        print(f"[v0] DBSCAN Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"DBSCAN Error: {str(e)}"}), 400

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """
        Endpoint untuk menerima file CSV via multipart/form-data dan mengembalikan
        data numerik yang siap diproses.

        Strategi parsing (berlapis):
            1. Coba pandas.read_csv auto-detect delimiter
            2. Jika gagal, coba delimiter umum (',', ';', '\t', '|') dan engine 'python'/'c'
            3. Jika masih gagal, coba beberapa encoding
            4. Jika masih gagal, baca sebagai teks dan lakukan parsing manual

        Pembersihan data:
            - Drop kolom yang semuanya NaN
            - Hapus kolom 'Unnamed'
            - Pilih kolom numerik; jika tidak ada, coba konversi kolom string ke numerik
            - Fill NaN dengan 0

        Response:
            JSON dengan keys: success, data (list of lists), rows, columns, column_names,
            dan optional 'nama' (jika kolom 'Nama' tersedia).
        """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        df = None
        errors = []
        
        # Strategy 1: Try pandas with auto-detect and skip bad lines
        try:
            file.seek(0)
            df = pd.read_csv(
                file, 
                sep=None, 
                engine='python',
                on_bad_lines='skip'
            )
            print("[v0] CSV parsed successfully with auto-detect delimiter")
        except Exception as e:
            errors.append(f"Auto-detect failed: {str(e)}")
            print(f"[v0] Auto-detect failed: {str(e)}")
        
        # Strategy 2: Try common delimiters with different engines
        if df is None:
            for delimiter in [',', ';', '\t', '|']:
                for engine in ['python', 'c']:
                    try:
                        file.seek(0)
                        df = pd.read_csv(
                            file, 
                            sep=delimiter,
                            engine=engine,
                            on_bad_lines='skip',
                            encoding='utf-8'
                        )
                        print(f"[v0] CSV parsed with delimiter '{delimiter}' and engine '{engine}'")
                        break
                    except Exception as e:
                        errors.append(f"Delimiter '{delimiter}' with engine '{engine}': {str(e)}")
                        continue
                if df is not None:
                    break
        
        # Strategy 3: Try with different encodings
        if df is None:
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    file.seek(0)
                    df = pd.read_csv(
                        file, 
                        sep=None,
                        engine='python',
                        on_bad_lines='skip',
                        encoding=encoding
                    )
                    print(f"[v0] CSV parsed with encoding '{encoding}'")
                    break
                except Exception as e:
                    errors.append(f"Encoding '{encoding}': {str(e)}")
                    continue
        
        # Strategy 4: Read as string and manually parse
        if df is None:
            try:
                file.seek(0)
                content = file.read().decode('utf-8', errors='ignore')
                lines = content.strip().split('\n')
                
                # Try to detect delimiter by checking first line
                first_line = lines[0]
                delimiter = None
                for delim in [',', ';', '\t', '|']:
                    if delim in first_line:
                        delimiter = delim
                        break
                
                if delimiter:
                    data = []
                    for line in lines:
                        if line.strip():
                            data.append(line.split(delimiter))
                    df = pd.DataFrame(data[1:], columns=data[0])
                    print(f"[v0] CSV parsed manually with delimiter '{delimiter}'")
            except Exception as e:
                errors.append(f"Manual parsing: {str(e)}")
        
        if df is None:
            error_msg = "Could not parse CSV file. Tried multiple strategies:\n" + "\n".join(errors[:3])
            return jsonify({"error": error_msg}), 400
        
        # Drop columns yang semuanya NaN atau kosong
        df = df.dropna(axis=1, how='all')
        
        # Drop kolom yang bernama "Unnamed" (kolom tanpa header)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        print(f"[v0] After cleaning: {len(df.columns)} columns remaining")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        # If no numeric columns found, try to convert string columns to numeric
        if numeric_df.empty:
            print("[v0] No numeric columns found, attempting to convert string columns")
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
            numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return jsonify({"error": "No valid numeric data found in CSV"}), 400
        
        # Remove rows with NaN values
        numeric_df = numeric_df.fillna(0)
        
        if numeric_df.empty:
            return jsonify({"error": "No valid numeric data after cleaning"}), 400
        
        # Convert to list of lists
        data = numeric_df.values.tolist()
        # If the original dataframe had a 'Nama' column, include it aligned with rows
        nama_list = None
        if 'Nama' in df.columns:
            try:
                nama_series = df['Nama'].fillna('').astype(str)
                nama_list = nama_series.tolist()
            except Exception:
                nama_list = None

        resp = {
            "success": True,
            "data": data,
            "rows": len(data),
            "columns": len(data[0]) if data else 0,
            "column_names": numeric_df.columns.tolist()
        }
        if nama_list is not None:
            resp['nama'] = nama_list

        return jsonify(resp)
    except Exception as e:
        print(f"[v0] Unexpected error: {str(e)}")
        return jsonify({"error": f"Error reading file: {str(e)}"}), 400

@app.route('/api/upload-excel', methods=['POST'])
def upload_excel():
    """
        Endpoint untuk menerima file Excel (.xlsx, .xls) dan mengembalikan data numerik.

        Alur:
            - Coba baca dengan engine 'openpyxl', jika gagal coba 'xlrd'
            - Lakukan pembersihan yang sama seperti upload_csv
            - Kembalikan struktur JSON yang sama (success, data, rows, columns, column_names, nama)
        """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        try:
            print(f"[v0] Attempting to read Excel file: {file.filename}")
            df = pd.read_excel(file, engine='openpyxl')
            print(f"[v0] Excel file read successfully with openpyxl")
        except Exception as e:
            print(f"[v0] openpyxl failed: {str(e)}, trying xlrd...")
            try:
                file.seek(0)
                df = pd.read_excel(file, engine='xlrd')
                print(df['Nama'])
                print(f"[v0] Excel file read successfully with xlrd")
            except Exception as e2:
                print(f"[v0] xlrd also failed: {str(e2)}")
                return jsonify({"error": f"Failed to read Excel file: {str(e)} | {str(e2)}"}), 400
        
        # Drop columns yang semuanya NaN atau kosong
        df = df.dropna(axis=1, how='all')
        
        # Drop kolom yang bernama "Unnamed" (kolom tanpa header)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        print(f"[v0] After cleaning: {len(df.columns)} columns remaining")
        print(f"[v0] Columns: {df.columns.tolist()}")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("[v0] No numeric columns found, attempting to convert string columns")
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
            numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return jsonify({"error": "No valid numeric data found in Excel file. Please check your file format."}), 400
        
        numeric_df = numeric_df.fillna(0)
        
        if numeric_df.empty:
            return jsonify({"error": "No valid numeric data after cleaning"}), 400
        
        # Convert to list of lists
        data = numeric_df.values.tolist()
        
        print(f"[v0] Successfully processed Excel file: {len(data)} rows, {len(data[0])} columns")
        
        # If the original dataframe had a 'Nama' column, include it aligned with rows
        nama_list = None
        if 'Nama' in df.columns:
            try:
                nama_series = df['Nama'].fillna('').astype(str)
                nama_list = nama_series.tolist()
            except Exception:
                nama_list = None

        resp = {
            "success": True,
            "data": data,
            "rows": len(data),
            "columns": len(data[0]) if data else 0,
            "column_names": numeric_df.columns.tolist()
        }
        if nama_list is not None:
            resp['nama'] = nama_list

        return jsonify(resp)
    except Exception as e:
        print(f"[v0] Unexpected error in upload_excel: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 400

if __name__ == '__main__':
    # This prevents "signal only works in main thread" error when running in certain environments
    app.run(
        debug=True,
        port=5000,
        use_reloader=False,  # Disable reloader to avoid signal issues
        use_debugger=False   # Disable debugger to avoid signal issues
    )
