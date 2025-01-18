import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the data (Entering path for my github containing output in csv format)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/joiya-saad/Data-Analysis-3/refs/heads/main/Extra_Assignment_1/Bias-Variance-Results.csv"  # Replace with your CSV link
    return pd.read_csv(url)

# Load data
results_df = load_data()

# Streamlit app layout
st.title("Bias-Variance Tradeoff Visualization")

st.text("Author: Saad Joiya - 2300715")

st.text("This app is developed for the course DA3 to demonstrate tradeoff between Bias^2 and Variance")

st.header("Introduction")

st.text("The Mean Squared Error (MSE) can be broken down into two key components: \
         bias squared and variance. The bias of a model's predictions refers to the systematic error introduced \
         by approximating a real-world problem with a simplified model. In other words, it reflects how much, \
         on average, the model's predictions differ from the true values. As the output can be a result of the interplay between different variables \
          increasing the number of variables (complexity) in the model, tends to reduce Bias. \
        On the other hand, variance measures the extent to which a model’s predictions fluctuate when trained on different subsets of the data. \
         High variance indicates that the model is sensitive to small changes in the data, while low variance \
         suggests the model is more stable. Addition of extra variables (complexity) leads to higher variance as model tends to become \
        more flexible to changes. There is typically an inverse relationship between bias and variance—reducing \
         one tends to increase the other. This trade-off is crucial to understanding model performance, \
        especially when balancing underfitting (high bias) and overfitting (high variance). \
            The purpose of this application is to visually demonstrate this trade-off by applying \
                  it to a real-world prediction task. By varying the complexity of the model \
                      (e.g., adjusting the polynomial degree in regression), we can observe how bias \
                          and variance evolve, helping to visualize the impact of model choices on prediction error.")

st.header("Data Description and methodology")

st.text("In order to demonstrate the relationship between Bias^2 and Variance, \
we use the Pima Indians Diabetes Dataset. This dataset has Diabetes as the target \
variable with binary values 0 (not present) and 1 (present). There are many explanatory \
variables such as BMI, Glucose, Pregnancies, BloodPressure, Insulin, Age etc. For our purpose \
we pick Glucose as the explanatory variable and use it to predict target variable Diabetes. We increase \
the complexity of the model by increasing the polynomial degree of the explanatory variable Glucose. We split the data into train \
        test by 70-30 ratio and run 100 simulations for each polynomial degree to gather the results and observe \
how the Bias^2 and Variance change respectively.")

# Slicer for Polynomial Degree
min_degree = int(results_df['Degree'].min())
max_degree = int(results_df['Degree'].max())
st.subheader("We can adjust the polynomial degree using the slider below to observe how Bias^2 and Variance change with increasing complexity")
degree_range = st.slider("Select Polynomial Degree Range", min_degree, max_degree, (min_degree, max_degree))

# Filter data based on slider
filtered_df = results_df[(results_df['Degree'] >= degree_range[0]) & (results_df['Degree'] <= degree_range[1])]

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bias² plot
color = 'tab:blue'
ax1.set_xlabel('Degree of Polynomial')
ax1.set_ylabel('Bias²', color=color)
ax1.plot(filtered_df['Degree'], filtered_df['Bias^2'], marker='o', color=color, label='Bias²')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(range(degree_range[0], degree_range[1] + 1))

# Variance plot
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Variance', color=color)
ax2.plot(filtered_df['Degree'], filtered_df['Variance'], marker='s', linestyle='--', color=color, label='Variance')
ax2.tick_params(axis='y', labelcolor=color)

# Title and grid
plt.title('Bias-Variance Tradeoff Across Polynomial Degrees')
fig.tight_layout()
plt.grid(True)

# Show plot
st.pyplot(fig)

st.header("Interpretation")

st.text("The chart illustrates a well-known trade-off between model complexity, bias, and variance. \
        On average, simpler models, with fewer features or lower polynomial degrees, tend to exhibit lower variance \
         but higher bias. This means they are less sensitive to the noise in the data, but they might underfit \
         by failing to capture the underlying relationships. Conversely, more complex models—characterized by \
        higher polynomial degrees or the inclusion of more variables—generally show higher variance but lower bias. \
         These models fit the training data more closely, but they risk overfitting and become sensitive to small \
         variations in the data. Selecting the optimal model involves finding a balance: the sweet spot where both \
         bias and variance are relatively low. It is also to note that in our example, we just used one explanatory \
        variable and we increased complexity by increasing the polynomial degree i.e. adding variations of the same \
        variable to add complexity. Complexity can be added by using additional variables and their polynomials as well.\
        Addition of variables would shift the optimal point for the model. Hence, the balance between Bias^2 and variance \
        depends on the number of features and their variations present in the model.")


st.header("Disclaimer")

st.text("LLMs were used to take guidance on the process of running the simulations for different polynomial degrees while \
         fitting the model. It was also leveraged to understand how to publish apps on streamlit. ")
