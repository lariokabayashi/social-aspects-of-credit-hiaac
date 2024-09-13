## Application of Large Language Models for Explanations in Credit Risk Decisions

### Overview

This project focuses on applying Large Language Models (LLMs) to generate understandable natural language explanations for decisions made in credit risk assessments. The goal is to make credit scoring models more transparent and accessible, particularly to individuals who may lack familiarity with machine learning concepts.

### Objectives

- Develop accessible natural language explanations for credit scoring.
- Utilize machine learning models and interpretability techniques, such as SHAP (SHapley Additive exPlanations) and PDP (Partial Dependence Plot), to explain the model's decisions.
- Employ advanced prompt engineering techniques to enhance model performance.

### Key Components

1. **Dataset**: 
   - Uses the German Credit dataset from UCI, containing 1,000 entries with 20 categorical attributes related to personal and financial information.
   
2. **Machine Learning Models**:
   - The XGBoost model is employed for its robust performance in credit scoring applications.
   - Evaluation metrics include AUC, Balanced Accuracy, Precision, Recall, F1, Specificity, and Sensitivity.

3. **Interpretability Methods**:
   - **SHAP Values**: To explain the contribution of individual features to the model's predictions.
   - **Partial Dependence Plot (PDP)**: To visualize the relationship between specific features and the model's predictions.

4. **Large Language Models**:
   - **Falcon and LLaMA Models**: Used to generate natural language explanations. Techniques like Function Calling and Prompt Engineering were employed to improve the performance of these models.

5. **User Interface**:
   - Developed using Gradio to provide a user-friendly platform for interacting with the model and accessing its explanations.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lariokabayashi/social-aspects-of-credit-hiaac.git
   cd social-aspects-of-credit-hiaac
   ```

2. Install the required dependencies:
   ```bash
   pip install groq requests flask
   ```

### Usage

1. **Running the Model**: 
   - Execute the main script to start the application and access the Gradio UI.
   ```bash
   export GROQ_API_KEY=xxxxxxxxxxx
   python PDP_api.py
   python SHAP_api.py
   python app.py
   ```
   
2. **Interacting with the Interface**:
   - Access the URL indicated on the terminal
   - Use the provided interface to input credit data and receive explanations for credit decisions.

### Limitations

- **Scalability**: Requires considerable computational power for large-scale or real-time analysis.
- **Bias and Fairness**: The performance depends on the quality of the data and models used, which might be subject to bias.
- **Security**: Potential vulnerabilities to adversarial attacks need to be considered.

### Contributing

Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.

### License

This project is licensed under the [MIT License](LICENSE).
