<div align="center">
  <img width="70" height="70" alt="Text2SQL" src="https://github.com/user-attachments/assets/d2387972-61ef-4b72-8cbf-e4073d8111cb" />
  <h1>Text2SQL-Chinook</h1>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=yellow" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/🤗-Transformers-FF6200?style=for-the-badge&logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</div>

# <div align="center"> • Chinook DB • Streamlit Demo • </div>

<div align="center">
  <img alt="Demo" src="./streamlit/demo.gif" width="800"/>
</div>

**Transform natural language questions into executable SQL** — A production-ready Text2SQL pipeline trained on the Chinook database.

---

# Text2SQL-Chinook

Text2SQL-Chinook is a Streamlit-based web application that translates natural language queries into SQL statements, executes them on the Chinook SQLite database, and displays the results. The application leverages a fine-tuned **T5 transformer model** to enable intuitive and efficient database interactions without requiring users to write SQL manually.

---

## 🚀 Overview

The application allows users to interact with the Chinook database using plain English queries. By harnessing the capabilities of a fine-tuned T5 model, Text2SQL-Chinook converts natural language inputs into executable SQL commands, retrieves the relevant data, and presents it through a clean and user-friendly Streamlit interface.

This project demonstrates how modern transformer models can bridge the gap between natural language and structured query systems, making database exploration more accessible to both technical and non-technical users.

---

## ✨ Features

- ✅ Natural language to SQL translation using SLM-SQL-0.6B
- ✅ Execution of generated SQL queries on the Chinook SQLite database  
- ✅ Interactive results displayed in a Streamlit web interface  
- ✅ Fast, lightweight, and easy-to-deploy architecture  
- ✅ Designed for experimentation, demos, and learning Text2SQL workflows  

---

## 📋 Prerequisites

Before running the project, ensure you have:

- **Python 3.8 or higher**
- **pip** package manager
---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/DwaipayanDutta/Text2SQL-Chinook.git
cd Text2SQL-Chinook


