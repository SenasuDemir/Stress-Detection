# 🧠 Stress Detection NLP Model

## Aim 🎯
The goal of this project is to develop a **Natural Language Processing (NLP)** model that can predict whether a person is experiencing stress based on their textual input. By analyzing user-provided sentences, the model will classify them into two categories: **stress (1)** and **no stress (0)**. This model could help in understanding how individuals express stress in written form and assist in identifying potential signs of stress in various contexts such as social media, forums, or personal communication.

## Data Explanation 📊
The dataset used in this project consists of textual data from various subreddits. Each row includes the following columns:

- **text**: A sentence or paragraph representing the user's feelings or thoughts. This is the main input used to predict stress levels.
- **label**: The target variable, where `0` represents "no stress" and `1` represents "stress". This indicates whether the text is classified as showing signs of stress or not.

The dataset was gathered from various subreddit discussions on topics such as **PTSD**, **relationships**, and other **personal experiences**. The goal is to leverage NLP techniques to analyze how users express emotional or psychological states and classify these expressions as indicating stress or not.

For this project, we will focus on preprocessing the text data (e.g., cleaning, tokenization) and using it to train a machine learning model that can classify new, unseen text into the two categories.

## Results 📈
After evaluating various machine learning algorithms, the **Logistic Regression** model outperformed the others with the highest accuracy score of **74.12%**. Below are the accuracy scores for the different models tested:

| Model                          | Accuracy |
| ------------------------------ | -------- |
| Logistic Regression             | 74.12%   |
| Random Forest Classifier        | 71.48%   |
| Bernoulli Naive Bayes           | 70.60%   |
| Multinomial Naive Bayes         | 70.60%   |
| Ada Boost Classifier            | 69.54%   |
| Gradient Boosting Classifier    | 68.66%   |
| Decision Tree Classifier        | 61.97%   |

### Confusion Matrix 🧩

The confusion matrix for the **Logistic Regression** classifier, which had the highest accuracy, is displayed below. The confusion matrix highlights the performance of the model in terms of true positives, true negatives, false positives, and false negatives:

From the confusion matrix:

- **True positives (229)**: Predicted as stressed and actually stressed.
- **True negatives (192)**: Predicted as not stressed and actually not stressed.
- **False positives (71)**: Predicted as stressed but actually not stressed.
- **False negatives (76)**: Predicted as not stressed but actually stressed.

These results demonstrate that **Logistic Regression** is the most accurate model for stress prediction, with a high number of true positives and true negatives. However, there is a moderate amount of false positives and false negatives, indicating that the model occasionally misclassifies stressed and non-stressed individuals.

The accuracy score and confusion matrix provide an in-depth view of the performance of the **Logistic Regression** model, highlighting its strengths and areas for improvement in stress detection.

## Link to Hugging Face Space 🌐

You can explore the project further on Hugging Face [here](https://huggingface.co/spaces/Senasu/Stress-Detection).


Feel free to contribute to the project and improve the model! Any suggestions or improvements are welcome.
