## Binary-Classification

#Introduction
This assignment plans to assess how different supervised learning techniques fare at solving a binary classification problem. Examining their strengths and weaknesses will provide insight into which algorithm works best. Various classification methods will be tested on the task to determine which approach yields the most accurate predictions. By analyzing their behaviors, we can discern which algorithm manages this type of classification challenge most effectively.

The Iris data was carefully studied and visualized using pairplot function of seaborn to learn about flower types. We choose libraries and algorithms carefully to build the best model. Some key parts shaped our process such as using appropriate libraries, classification methods, and strategic hyperparameter tuning.

#Libraries Used: 

 Pandas and NumPy: Essential for effective data handling and mathematical processes, forming the core of our data preparation work.  Matplotlib and Seaborn: Important for creating helpful visualization that helped make sense of and look through the information of dataset.
 Scikit-learn: A versatile machine learning library that played a central role in model training, evaluation, and the fine-tuning of hyperparameters. It helped us adjust settings and measure results as they trained algorithms on dataset.  Graphviz: The graphviz visualization aids help to show how the decision trees arrives at its results.
Classification Methods and Hyperparameter Selection:
 RandomForest: This classification method was chosen for its ability to combine learning from multiple decision trees, using 1000 trees (n_estimators) to create a varied and reliable group of models. The default settings were kept to make it straightforward.
 Logistic Regression: We decided to use this method because it is easy to understand and interpret. Through testing different values for the regularization parameter (C), we found that 1e5 worked best at optimizing the results. Also, maximum iteration has been set to 200 to remove overfitting.
