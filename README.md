# *Detailed Explanation of the APS Failure Prediction Code*

This code implements a *machine learning pipeline* to predict failures in *APS (Air Pressure System)* trucks using feature selection with *Genetic Algorithm (GA)* and *Particle Swarm Optimization (PSO)*. Below is a breakdown of each section, including functions, parameters, and their roles.

---

## *1. Import Libraries*
### *Core Libraries*
- **pandas (pd): For data manipulation (loading, cleaning).
- **numpy (np): For numerical operations (handling NaN values).
- **matplotlib.pyplot (plt) & seaborn (sns): For visualizations.
- **random**: Used in GA for random initialization.

### *Scikit-learn (Machine Learning)*
- **train_test_split**: (Not used here, but typically for splitting data).
- **SimpleImputer**: Fills missing values (using mean imputation).
- **StandardScaler**: Standardizes features (mean=0, std=1).
- **LabelEncoder**: Converts class labels (neg/pos) to (0/1).
- **SVC, KNeighborsClassifier, RandomForestClassifier**: ML models.
- **Metrics (accuracy_score, precision_score, etc.): Evaluate model performance.

### *Optimization Libraries*
- **deap**: For Genetic Algorithm (feature selection).
- **pyswarms**: For Particle Swarm Optimization (feature selection).

---

## *2. Data Loading & Preprocessing*
### **load_data(train_path, test_path)**
- *Input*: Paths to training and test CSV files.
- *Output*: DataFrames train_df and test_df.
- *Why skip 20 rows?* The first 20 rows contain metadata, not actual data.

### **preprocess_data(train_df, test_df)**
#### *Steps:*
1. **Replace 'na' with np.nan**  
   - Missing values are marked as 'na' in the dataset.  
   - Convert them to np.nan for proper handling.

2. **Encode Target Variable (class)**  
   - LabelEncoder() converts:  
     - 'neg' (no failure) → 0  
     - 'pos' (failure) → 1  

3. **Split Features (X) and Labels (y)**  
   - X_train, X_test: All columns except 'class' (converted to float).  
   - y_train, y_test: Only the 'class' column (binary labels).

4. *Impute Missing Values*  
   - SimpleImputer(strategy='mean') replaces NaN with the *mean* of each feature.

5. *Standardize Features*  
   - StandardScaler() scales features to have *mean=0* and *std=1* (improves model performance).

#### *Output:*
- X_train, y_train: Processed training data.
- X_test, y_test: Processed test data.

---

## *3. Feature Selection*
### *Why Feature Selection?*
- Many features may be irrelevant or redundant.
- *GA and PSO* select the best subset of features to improve model performance.

### *A. Genetic Algorithm (GA)*
#### **run_genetic_algorithm(X_train, y_train, X_test, y_test)**
- *Goal*: Find the best subset of features using evolution-inspired optimization.

#### *Steps:*
1. *Initialize DEAP Structures*  
   - creator.create("FitnessMax", ...): Maximizes fitness (F1-score).  
   - creator.create("Individual", ...): Binary representation (1=select feature, 0=ignore).  

2. *Define GA Operations*  
   - toolbox.register("attr_bool", random.randint, 0, 1): Each feature is either selected (1) or not (0).  
   - toolbox.register("individual", ...): A binary vector representing a feature subset.  
   - toolbox.register("population", ...): A group of potential solutions.  

3. **Fitness Function (ga_fitness)**  
   - *Input*: A binary vector (selected features).  
   - *Steps*:  
     1. If fewer than 5 features are selected → return 0.0 (invalid solution).  
     2. Train a RandomForestClassifier on selected features.  
     3. Compute *F1-score* on test data.  
   - *Output*: F1-score (higher is better).  

4. *Configure GA Operations*  
   - mate: Crossover (combines two solutions).  
   - mutate: Randomly flips bits (explores new solutions).  
   - select: Tournament selection (keeps best solutions).  

5. *Run GA*  
   - pop = toolbox.population(n=10): Small population (for speed).  
   - algorithms.eaSimple(...): Runs evolution (5 generations).  

6. *Extract Best Solution*  
   - best_ind: The best feature subset found.  
   - selected_indices: Indices of selected features.  

#### *Output:*  
- ga_selected_indices: List of selected feature indices.

---

### *B. Particle Swarm Optimization (PSO)*
#### **run_pso(X_train, y_train, X_test, y_test)**
- *Goal*: Find optimal features using swarm intelligence.

#### *Steps:*
1. **Fitness Function (pso_fitness)**  
   - *Input*: A swarm of particles (each represents a feature subset).  
   - *Steps*:  
     1. If fewer than 5 features → return 0.0.  
     2. Train RandomForestClassifier on selected features.  
     3. Compute *F1-score*.  
   - *Output*: Negative F1-score (PSO minimizes, so we invert it).  

2. *Configure PSO*  
   - options:  
     - c1: Cognitive parameter (how much particles follow their own best).  
     - c2: Social parameter (how much particles follow the swarm's best).  
     - w: Inertia weight (controls exploration).  
     - k: Number of neighbors (for local best).  
     - p: Minkowski distance metric.  

3. *Run PSO*  
   - optimizer = BinaryPSO(...): Binary version (features are on/off).  
   - optimizer.optimize(...): Runs for 10 iterations.  

4. *Extract Best Solution*  
   - pos: Best feature subset (binary vector).  
   - selected_indices: Indices where pos > 0.5.  

#### *Output:*  
- pso_selected_indices: List of selected feature indices.

---

## *4. Model Training & Evaluation*
### **evaluate_model(name, model, X_train, y_train, X_test, y_test)**
- *Input*: Model name, model object, training/test data.  
- *Output*: Dictionary of metrics (Accuracy, Precision, etc.) and confusion matrix.  

#### *Metrics Computed:*
1. *Accuracy*: Overall correctness.  
2. *Precision*: % of predicted failures that were correct.  
3. *Recall*: % of actual failures detected.  
4. *F1-Score*: Balance between precision & recall.  
5. *Confusion Matrix*:  
   - Rows: Actual class.  
   - Columns: Predicted class.  

### **evaluate_all_models(models, ...)**
- *Input*: Dictionary of models (SVM, KNN, Random Forest).  
- *Steps*:  
  1. Train each model on:  
     - All features.  
     - GA-selected features.  
     - PSO-selected features.  
  2. Store results in results_df and conf_matrices.  
- *Output*:  
  - results_df: DataFrame of model performances.  
  - conf_matrices: Confusion matrices for each model.  

---

## *5. Visualization & Results*
### **visualize_results(results_df, conf_matrices)**
1. *Bar Plots for Metrics*  
   - Compares Accuracy, Precision, Recall, F1-Score across models.  
2. *Combined Metrics Plot*  
   - Seaborn bar plot showing all metrics together.  
3. *Confusion Matrices*  
   - Heatmaps showing true vs. predicted classes.  

### **print_top_features(selected_indices, name, feature_names)**
- Prints the names of selected features (for interpretability).

### *Final Output*
- Displays results_df (performance metrics).  
- Shows all visualizations.  

---

## *Key Takeaways*
1. *Data Preprocessing*  
   - Handles missing values (SimpleImputer).  
   - Standardizes features (StandardScaler).  

2. *Feature Selection*  
   - *GA*: Evolutionary approach (good for large feature spaces).  
   - *PSO*: Swarm intelligence (faster convergence).  

3. *Model Evaluation*  
   - Tests SVM, KNN, and Random Forest.  
   - Compares full features vs. selected features.  

4. *Visualization*  
   - Helps compare model performance easily.  

This pipeline is *modular*—you can swap models, tweak GA/PSO parameters, or adjust preprocessing steps easily.
