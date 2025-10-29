# Food Clustering Thesis - Setup Guide

## Project Structure

This project uses a **Python-only architecture** with Flask backend and Streamlit frontend.

\`\`\`
project/
├── backend/                # Python Flask backend
│   ├── app.py             # Flask application
│   ├── clustering.py      # Clustering algorithms
│   └── requirements.txt    # Python dependencies
├── frontend/               # Streamlit frontend
│   ├── app.py             # Streamlit application
│   └── requirements.txt    # Python dependencies
├── run-all.bat            # Script to run both (Windows)
├── run-all.sh             # Script to run both (Linux/Mac)
├── run-backend.bat        # Script to run backend only (Windows)
├── run-frontend.bat       # Script to run frontend only (Windows)
└── SETUP.md               # This file
\`\`\`

## Quick Start

### Windows
Double-click `run-all.bat` to start both backend and frontend automatically.

### Linux/Mac
\`\`\`bash
chmod +x run-all.sh
./run-all.sh
\`\`\`

Then open your browser to `http://localhost:8501`

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### 1. Backend Setup (Flask)

1. Navigate to the backend directory:
\`\`\`bash
cd backend
\`\`\`

2. Create a virtual environment (recommended):
\`\`\`bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Run the Flask server:
\`\`\`bash
python app.py
\`\`\`

The backend will run on `http://localhost:5000`

### 2. Frontend Setup (Streamlit)

**IMPORTANT: Start the backend FIRST before starting the frontend!**

1. Open a NEW terminal window and navigate to the frontend directory:
\`\`\`bash
cd frontend
\`\`\`

2. Create a virtual environment (recommended):
\`\`\`bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Run the Streamlit app:
\`\`\`bash
streamlit run app.py
\`\`\`

The frontend will run on `http://localhost:8501`

---

## Common Errors & Solutions

### Error: "signal only works in main thread of the main interpreter"

**Cause:** You tried to run `streamlit run app.py` in the backend folder.

**Solution:** 
- Backend (`backend/app.py`) is a Flask API, NOT a Streamlit app
- Frontend (`frontend/app.py`) is a Streamlit app
- Run them in separate terminals:
  - Terminal 1: `cd backend && python app.py`
  - Terminal 2: `cd frontend && streamlit run app.py`

Or use the provided scripts:
- Windows: Double-click `run-all.bat`
- Linux/Mac: Run `./run-all.sh`

### Backend not connecting

- Ensure Flask server is running on port 5000
- Check that CORS is enabled in `backend/app.py`
- Verify the backend URL in `frontend/app.py` (default: http://localhost:5000)
- Make sure backend started BEFORE frontend

### CSV upload fails

- Ensure CSV has numeric data (except header row)
- Check file format is valid CSV
- Verify all columns (except first) contain numbers

### Streamlit connection issues

- Ensure backend is running before starting Streamlit
- Check that both services are on the correct ports
- Try restarting both services

### Clustering takes too long

- Reduce the number of data points
- Reduce `max_iterations` for K-means
- Increase `eps` for DBSCAN

---

## API Endpoints

### K-means Clustering
- **POST** `/api/kmeans`
- Request body:
\`\`\`json
{
  "data": [[1.0, 2.0], [2.0, 3.0], ...],
  "k": 3,
  "max_iterations": 100
}
\`\`\`

### DBSCAN Clustering
- **POST** `/api/dbscan`
- Request body:
\`\`\`json
{
  "data": [[1.0, 2.0], [2.0, 3.0], ...],
  "eps": 0.5,
  "min_samples": 5
}
\`\`\`

### CSV Upload
- **POST** `/api/upload-csv`
- Form data: `file` (CSV file)

---

## Features

### Home Page
- Introduction to the application
- Overview of K-means and DBSCAN algorithms
- Navigation to other pages

### About Page
- Detailed explanation of K-means clustering
- Detailed explanation of DBSCAN clustering
- Advantages and disadvantages of each algorithm
- Evaluation metrics used

### Method Page
- Upload CSV file with food nutrition data
- Configure K-means parameters (k, max_iterations)
- Configure DBSCAN parameters (eps, min_samples)
- Run clustering algorithms

### Results Page
- Visualize K-means clustering results
- Visualize DBSCAN clustering results
- Compare clustering results
- Interactive Plotly charts

---

## CSV Data Format

Your CSV file should have the following format:
- First row: header with column names
- Each subsequent row: one food item
- Columns should contain numeric values (nutrition data)

Example:
\`\`\`
Food,Protein,Fat,Carbs,Fiber,Calories
Chicken,26,3.6,0,0,165
Beef,26,15,0,0,250
Rice,2.7,0.3,28,0.4,130
\`\`\`

---

## Algorithms

### K-means
- Partitioning algorithm that divides data into k clusters
- Minimizes intra-cluster distance
- Iterative and converges quickly
- Parameters: k (number of clusters), max_iterations

### DBSCAN
- Density-based clustering algorithm
- Finds clusters of arbitrary shape
- Can detect outliers
- Parameters: eps (radius), min_samples (minimum points)

---

## Evaluation Metrics

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters
- **Davies-Bouldin Index**: Ratio of average similarity between each cluster and its most similar cluster
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion

---

## Dependencies

### Backend
- Flask 3.0.0
- Flask-CORS 4.0.0
- NumPy 1.24.3

### Frontend
- Streamlit 1.28.1
- Pandas 2.1.1
- NumPy 1.24.3
- Requests 2.31.0
- Plotly 5.17.0
