## Binary-Classification

Introduction:

This assignment plans to assess how different supervised learning techniques fare at solving a binary classification problem. Examining their strengths and weaknesses will provide insight into which algorithm works best. Various classification methods will be tested on the task to determine which approach yields the most accurate predictions. By analyzing their behaviors, we can discern which algorithm manages this type of classification challenge most effectively.

The Iris data was carefully studied and visualized using pairplot function of seaborn to learn about flower types. We choose libraries and algorithms carefully to build the best model. Some key parts shaped our process such as using appropriate libraries, classification methods, and strategic hyperparameter tuning.

Libraries Used: 

Pandas and NumPy: Essential for effective data handling and mathematical processes, forming the core of our data preparation work.

Matplotlib and Seaborn: Important for creating helpful visualization that helped make sense of and look through the information of dataset.

Scikit-learn: A versatile machine learning library that played a central role in model training, evaluation, and the fine-tuning of hyperparameters. It helped us adjust settings and measure results as they trained algorithms on dataset. 

Graphviz: The graphviz visualization aids help to show how the decision trees arrives at its results.

Classification Methods and Hyperparameter Selection:

RandomForest: This classification method was chosen for its ability to combine learning from multiple decision trees, using 1000 trees (n_estimators) to create a varied and reliable group of models. The default settings were kept to make it straightforward.

Logistic Regression: We decided to use this method because it is easy to understand and interpret. Through testing different values for the regularization parameter (C), we found that 1e5 worked best at optimizing the results. Also, maximum iteration has been set to 200 to remove overfitting.

K-Nearest Neighbors (KNN): This method is chosen because of its non-predetermined qualities, and optimal value of k was determined through an exploration using k-fold-cross-validation.

Training and Testing Process: 

The data was split into a training set and test set using an 80:20 ratio to balance model development and assessment. This ensured enough data for learning as well as validating the outcomes. 

The Scikit-learn library has been used to train and evaluate RandomForest and Logistic Regression models. For KNN, we had to take the extra step of finding the best k value from 1 to 31 through cross-validation to make sure we choose the right model setting. This helped improve how well it predicted.

Evaluation:

The confusion matrix is a cornerstone in assessing how well each classification method works. To determine confusion matrix and classification report for all the model, we have used functions of python library Scikit-Learn, which generate and displays confusion matrix and classification report by taking test and predicted data of that particular model.

Evaluation of each model is depends on how much score it gets for TP, TN, FP, and FN. All this values has been visualized by displaying confusion matrix using ConfusionMatrixDisplay function of Scikit-Learn library.

True Positives (TP): The number of instances where setosa, versicolor, and virginica were correctly predicted.
True Negatives (TN): The number of instances correctly identified as not belonging to any of the three classes.
False Positives (FP): Instances incorrectly classified as belonging to a class.
False Negatives (FN): Instances incorrectly classified as not belonging to a class.

The random forest model's confusion matrix provides understanding into how well it can detect complex connections between inputs. 100% true positive values for every category show the model's strong ability to forecast outcomes correctly, whereas 0% false positive and false negative values underline the model's preciseness. Also, the score of accuracy, precision, recall, and f1-score is 1.0, which indicate that model is providing correct output for each and every test data. Decision trees created by the RandomForest classifier can be difficult to understand. However, by using graphviz library we have visualized the five trees to makes the models easier to interpret.

Similar to RandomForest, the Logistic Regression confusion matrix illustrates 100% accurate predictions. The model's simplicity shines through in its ability to make clear and precise classifications, as evident in the distribution of true and false predictions.

Using cross_val_score function of Scikit-Learn library optimal value of k was identified to use in KNN model. The confusion matrix for model with optimal k value shows the model's proficiency in identifying neighbors for precise predictions. A 100% value for true positives and accuracy score showcases KNN's adaptability to handle different types of information well.

All three models gave 100% accuracy score, it is hard to differentiate and compare efficiency of model for iris dataset used in this assignment. However, with larger and complex dataset, each model will have their own pros and cons, which will determine what model will work most efficiently on given dataset.
