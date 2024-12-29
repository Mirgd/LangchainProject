# LangchainProject
This is a prject for using Langchain as a framework for development LLM application and Hugging Face models as the LLM

# Table of content 
### 1-Overview 
### 2-How to run this project? 

# 1-Overview 
I will compare two methods for development LLM application 
### 1: simple qrery-answer app
### 2: Retrivel Augmented Generation(RAG) app

# simple qrery-answer app
This method is helpful when you asking the LLM of a fact or related to something that LLM has trained on.
example: 

![image](https://github.com/user-attachments/assets/12d05250-8950-4733-9feb-84bd873f074c)

# RAG app
This mehtod is using Retrivel Augmented Generation(RAG) technique to fetch information from your data. Tt can be thought as a costomized LLM trained on your data.
example data:

![image](https://github.com/user-attachments/assets/46735dc3-acb4-4860-9046-ec24674e57b2)

using regular LLM:(it just invents an artificial response):
![image](https://github.com/user-attachments/assets/60c36815-4cea-44b1-9bf7-51163805ab86)

using RAG LLM: (the response is related to data actually):
![image](https://github.com/user-attachments/assets/0e30baba-a11d-4873-87ac-4b38ba1ff9e1)

### another data example:

![image](https://github.com/user-attachments/assets/c238c2e3-6ed6-4a05-8750-da1acba0289d)

using regular LLM:(it just invents an artificial response):

![image](https://github.com/user-attachments/assets/7cc87139-3575-4a47-8c67-dd83510c42f1)


using RAG LLM: (the response is related to data actually):

![image](https://github.com/user-attachments/assets/fdaf1bdf-1931-4d2e-b403-774a502cd510)
![image](https://github.com/user-attachments/assets/0cb0dfcc-3c16-441d-a480-c59bfe1787a9)

# 2-How to run this project?

### 1- clone this repository  

### 2- create a virtual invironment and activate it 
- chromadb dependency: on Mac use `conda install onnxruntime -c conda-forge` For Windows users, install Microsoft Visual C++ Build Tools first, simply using this command:` winget install Microsoft.VisualStudio.2022.BuildTools --force --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended" `

### 3- fill the keys.py with actual keys by create an account in both:
- LangSmith https://smith.langchain.com/ (from settings)
- Hugging Face https://huggingface.co/ (from Access tokens)

### 4- run --> pip install -r requirements.txt

- to run the jupyter notebook: pip install notebook jupyter and choose the virtual environment as the kernal
- to run RAG_app.py: python RAG_app.py
- to run web_RAG.py: streamlit run web_RAG.py








