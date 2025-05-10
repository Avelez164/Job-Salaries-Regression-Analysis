import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI, ML, Data Science 2020-2025")
        self.root.geometry("1100x800")

        
        self.dataset = "samithsachidanandan/the-global-ai-ml-data-science-salary-for-2025"
        self.path = kagglehub.dataset_download(self.dataset)
        self.df = pd.read_csv(f"{self.path}/salaries.csv")
        
        self.setup_gui()

    def setup_gui(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(self.root, height=60)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X) 
        
        tk.Button(button_frame, text="Salary Distribution", font=('Times', 18), command = self.salary_dis).pack(side=tk.LEFT, expand=True)
        tk.Button(button_frame, text="Salary By Expirence", font=('Times', 18), command = self.salary_by_experience).pack(side=tk.LEFT, expand=True)
        tk.Button(button_frame, text="Top Countries", font=('Times', 18), command = self.top_countries).pack(side=tk.LEFT, expand=True)
        tk.Button(button_frame, text="Regresssion", font=('Times', 18), command = self.create_and_test_model).pack(side=tk.LEFT, expand=True)

        tk.Label(button_frame, text="Test %", font=('Times', 18)).pack(side=tk.LEFT, expand=True)
        self.test_size = tk.Entry(button_frame, width=7, font=('Times', 18))
        self.test_size.pack(side=tk.LEFT, expand=True)

        tk.Label(button_frame, text="Model #", font=('Times', 18)).pack(side=tk.LEFT, expand=True)
        model_options = [1,2,3,4]
        self.model=tk.StringVar(root)
        self.model.set(model_options[0])
        dropdown = tk.OptionMenu(button_frame, self.model, *model_options)
        dropdown.config(font=('Times', 18))
        dropdown.pack(side=tk.LEFT, expand=True)

        self.salary_dis()
            
    def create_and_test_model(self, test_size=0.2):
        if not self.test_size.get():
            test_size = test_size
        else:
            test_size = float(self.test_size.get())
            
        df = self.df[self.df['salary_in_usd'] < 400000]
        features_df = df.drop(['salary_currency'], axis = 1)
        features_df = pd.get_dummies(features_df, columns=['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size'])

        X = features_df.drop('salary_in_usd', axis=1)
        y = features_df['salary_in_usd']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model_name = "name"
        if not self.model.get():
            model_name = "Ridge Regression"
            model_to_use = Ridge(alpha=1.0)
        else:
            model = int(self.model.get())
            if model == 1:
                model_name = "Ridge Regression"
                model_to_use = Ridge(alpha=1.0)
            if model == 2:
                model_name = "Multi-Linear Regression"
                model_to_use = LinearRegression()
            if model == 3:
                model_name = "Lasso Regression"
                model_to_use = Lasso(alpha=0.1)
            if model == 4:
                model_name = "Decision Tree Regression"
                model_to_use = DecisionTreeRegressor(max_depth=3,min_samples_leaf=10,random_state=42)
        
        model_to_use.fit(X_train, y_train)

        
        y_pred = model_to_use.predict(X_test)
        y_pred = np.clip(y_pred, 20000, 400000)
        y_true = y_test
        
        if int(self.model.get()) != 4:
            coef = pd.Series(model_to_use.coef_, index=X.columns).sort_values(ascending=False)

        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()  # Remove the old canvas widget
            plt.close(self.fig)
        

        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
        
        
        ax1.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Test Data')
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r-', label='Ideal')
        ax1.set_xlabel('Actual Salary (USD)')
        ax1.set_ylabel('Predicted Salary (USD)')
        ax1.set_title('Actual vs. Predicted Salaries')
        #ax1.set_ylim([y_true.min() * 0.9, y_true.max()])
        ax1.legend()

        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals)
        ax2.axhline(0, color='red')
        ax2.set_xlabel('Predicted Salaries')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted Salaries')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        if int(self.model.get()) != 4:     
            messagebox.showinfo(
                f"{model_name} Analysis",
                f'''
                R^2: {r2_score(y_true, y_pred):.3f}
                MSE: {mean_squared_error(y_true, y_pred):,.2f}
                
                Top Factors that Increase Salary:\n{coef.head(10).round(3)}

                Top Factors that Decrease Salary:\n{coef.tail(10).round(3)}''')
        else:
            #cv_r2 = cross_val_score(model_to_use, X, y, cv=5, scoring='r2').mean()
            #print(f"CV R²: {cv_r2:.2f}")
            messagebox.showinfo(
                "Decision Tree Regression Analysis",
                f'''
                R^2: {r2_score(y_true, y_pred):.3f}
                MSE: {mean_squared_error(y_true, y_pred):,.2f}
                ''')    
        
        
            
        
        
            
    def salary_dis(self):
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()  # Remove the old canvas widget
            plt.close(self.fig)
        

        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
        
        df = self.df[self.df['salary_in_usd']< 400000]
        years = sorted(self.df['work_year'].unique())

        
        sns.histplot(self.df['salary_in_usd'], bins=50, kde=True, ax=ax1)
        ax1.set_title('Salary Distribution (in USD)')
        ax1.set_xlabel('Salary (USD)')
        ax1.set_ylabel('Frequency')


        for year in years:
            data = self.df[self.df['work_year'] == year]['salary_in_usd']
            sns.kdeplot(data,label=str(year), fill=True, alpha=0.9, ax=ax2)

        ax2.set_title('Salary Distribution for Each year (USD)')
        ax2.set_xlabel('Salary (usd)')
        ax2.set_ylabel('Frequency')
        ax2.legend(title='Year')
        
            
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        

    def salary_by_experience(self):
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()  # Remove the old canvas widget
            plt.close(self.fig)
            
        self.fig, self.ax = plt.subplots(figsize=(9,7))

        exp_salary = self.df.groupby('experience_level')['salary_in_usd'].mean().sort_values()
        sns.barplot(x=exp_salary.index, y=exp_salary.values, palette='viridis')
        plt.title('Average Salary by Experience Level')
        plt.xlabel('Experience Level')
        plt.ylabel('Average Salary (USD)')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)



    def top_countries(self):
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()  # Remove the old canvas widget
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=(9,7))
        
        country_salary = self.df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False)
        sns.barplot(x=country_salary.head(10).values, y=country_salary.head(10).index, palette='coolwarm')
        plt.title('Top 10 Countries by Average Salary')
        plt.xlabel('Average Salary (USD)')
        plt.ylabel('Country')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

                
root = tk.Tk()
app = GUI(root)
root.mainloop()
