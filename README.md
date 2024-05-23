
**Cancer-Type-Classification Project Report: Feature Selection and Classification with Random Forest and XGBoost**

**1. Introduction**

In this project, we aimed to perform feature selection and classification using Random Forest (RF) and XGBoost (XGB) algorithms on a dataset containing accelerometer data. The dataset was obtained from a study conducted by researchers from Harvard University and the University of Southern California to predict heavy drinking episodes among college students during a bar crawl. The primary objective was to select the most relevant features and build robust classification models to detect heavy drinking episodes based on mobile data.

**2. Data Loading and Preprocessing**

We started by loading the dataset, consisting of accelerometer data and corresponding labels indicating heavy drinking episodes. After loading the data, we checked its general information, statistics, and missing values. Additionally, we performed feature selection using Recursive Feature Elimination (RFE) with Linear Discriminant Analysis (LDA) to select the top 10 most relevant features.

**3. Feature Selection and Correlation Analysis**

We applied RFE to select the most informative features from the dataset. Then, we calculated the correlation matrix of the selected features to identify any significant correlations among them. Visualizing the correlation matrix using a heatmap allowed us to observe the relationships between the features.

**4. Model Training and Evaluation**

We divided the dataset into training and testing sets and trained Random Forest (RF) and XGBoost (XGB) classifiers on the selected features. After training, we evaluated the models using confusion matrices and calculated various metrics such as accuracy, sensitivity, and specificity. Both RF and XGB classifiers exhibited promising performance in detecting heavy drinking episodes, as evidenced by their high accuracy and balanced sensitivity and specificity.

**5. Results and Conclusion**

Our results indicate that feature selection combined with RF and XGB classifiers can effectively detect heavy drinking episodes using accelerometer data. The models showed robust performance, demonstrating the potential of mobile data in monitoring and intervening in real-time to promote responsible drinking behavior among college students. Future research could focus on incorporating additional features or optimizing the models further to improve their performance.

**6. Recommendations**

Based on our findings, we recommend deploying the RF and XGB classifiers in a real-world mobile application to monitor and detect heavy drinking episodes among college students. The application could provide timely interventions and alerts to promote responsible drinking habits and enhance student safety during social events like bar crawls.

**7. Acknowledgments**

We would like to thank the researchers from Harvard University and the University of Southern California for providing the dataset used in this project. Their contributions have been instrumental in advancing our understanding of detecting heavy drinking episodes using mobile data.

**8. References**

- Killian, J.A., Passino, K.M., Nandi, A., Madden, D.R., & Clapp, J. (2019). Learning to Detect Heavy Drinking Episodes Using Smartphone Accelerometer Data. Proceedings of the 4th International Workshop on Knowledge Discovery in Healthcare Data co-located with the 28th International Joint Conference on Artificial Intelligence (IJCAI 2019).

- J. Robert Zettl. (2002). The determination of blood alcohol concentration by transdermal measurement. Retrieved from [link](https://www.scramsystems.com/images/uploads/general/research/the-determination-of-blood-alcohol-concentrationby-transdermal-measurement.pdf).

