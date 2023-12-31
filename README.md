# Brain Stroke Prediction with visualization


1. **Data Loading and Exploration:**
   - The dataset is loaded from a CSV file named "healthcare-dataset-stroke-dataset.csv."
   - The first 5 instances of the dataframe are displayed, and the number of missing values in each column is printed and visualized using a bar graph.

2. **Data Cleaning and Transformation:**
   - The 'id' column, which is considered irrelevant to the problem, is dropped.
   - The 'gender' column is examined, and instances labeled as 'Other' are replaced with 'Female' to reduce dimensionality.
   - Various visualizations are used to analyze the distribution of the 'gender,' 'stroke,' 'hypertension,' 'work_type,' 'smoking_status,' 'Residence_type,' 'bmi,' 'age,' and 'avg_glucose_level' attributes.

3. **Handling Missing Data:**
   - The percentage of missing values in the 'bmi' column is calculated, and strategies for handling missing values are explored.
   - The decision is made to impute missing 'bmi' values with the median of the column.

4. **Exploratory Data Analysis (EDA):**
   - Visualizations such as histograms, box plots, and pie charts are used to explore the distribution of various attributes in the dataset.
   - Correlation matrix is visualized to understand the relationships between different attributes.

5. **Data Preprocessing:**
   - Categorical attributes are converted into dummy variables using one-hot encoding.
   - The dataset is highly undersampled, and oversampling (Random Oversampling) is applied to balance the classes.

6. **Model Training:**
   - Decision Tree and Random Forest classifiers are trained on the oversampled data.
   - The models are evaluated for accuracy, and the Random Forest model is chosen for further analysis based on higher accuracy.

7. **Model Evaluation:**
   - K-fold cross-validation is performed on the Random Forest model to assess its performance.
   - Logistic Regression is also applied, and its accuracy is evaluated.

8. **Making Predictions:**
   - The trained Random Forest model is used to make predictions on a manually entered set of health-related features representing an individual.
   - The predicted outcome is printed, indicating whether the individual is likely to have a stroke.<br>

Dataset Description:<br>
• The Txt file dataset is used for predicting brain stroke.<br>
• Size of the dataset: 316.97 KB<br>
• The dataset contains a total of 5110 entries.<br>
• 5110 entries are used.<br>
• The dataset contains 10 fields.<br>
• Training data : 4088 (80%)<br>
• Testing data : 1022 (20%)<br>
