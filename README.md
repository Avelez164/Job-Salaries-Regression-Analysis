# Job-Salaries-Regression-Analysis

# AI, ML, Data Science 2020–2025 Salary Explorer

An interactive Tkinter GUI for exploring and modeling global AI/ML/Data Science salaries (2020–2025), sourced via Kaggle.

---

## 🚀 Features

- **Salary Distribution**  
  – Overall histogram & KDE plot  
  – Year-by-year KDE overlays  
- **Average Salary by Experience**  
  – Bar chart grouped by `experience_level`  
- **Top Countries by Salary**  
  – Bar chart of the 10 highest-paying countries  
- **Regression Modeling**  
  – Choose among:  
    1. Ridge Regression  
    2. Ordinary Least Squares (Multi-Linear)  
    3. Lasso Regression  
    4. Decision Tree Regression  
  – Set custom test-set percentage  
  – Scatter plot of Actual vs. Predicted & Residuals  
  – Popup summary of R², MSE, and top feature coefficients  

---

## 📦 Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/Avelez164/Job-Salaries-Regression-Analysis.git
   cd Job-Salaries-Regression-Analysis

   ```
   Install system-level dependency for Tkinter
  On Ubuntu/Debian:
    ```bash
    sudo apt update
    sudo apt install python3-tk
    ```
    Install Python packages
    ```bash
    pip install -r requirements.txt
    ```

    ▶️ Running the App
    ```bash
    python3 483_project.py
    ```

