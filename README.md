<div align="center">
  <img width="128" height="128" alt="Text2SQL" src="https://github.com/user-attachments/assets/d2387972-61ef-4b72-8cbf-e4073d8111cb" />
  <h1>Text2SQL-Chinook 🗄️➡️✨</h1>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=yellow" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/🤗-Transformers-FF6200?style=for-the-badge&logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTM1IiBoZWlnaHQ9IjMyIiB2aWV3Qm94PSIwIDAgMTM1IDMyIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xMzUgMEgxMjEuMDJDMTIwLjYyIDAgMTIwIDEuMTg0IDEyMCAyLjIzNTdWMi4yMzU3QzEyMCAzLjI4NzMgMTIwLjYyIDQuNDcxNCAxMjEuMDIgNS4wMDAySDEzNVYwaFYwWk0xMzUgMzJIMTIxLjAyQzEyMC42MiAzMiAxMjAgMzAuODE2IDEyMCAyOS43NjQzVjI5Ljc2NDNDMTIwIDI4LjcxMTcgMTIwLjYyIDI3LjQ5NzIgMTIxLjAyIDI2Ljk5OThIMTM1VjMyWk0xMjEuMDIgNUgxMzVWNzUuNUgxMjEuMDJDMTE5LjUgNzUuNSAxMTguOTUgNzcuMDUwMiAxMTguOTUgNzguNzY0M1Y4NC4yMzU3QzExOC45NSA4NS45MTg4IDExOS41IDg3LjQ2OSAxMjEuMDIgODcuNUgxMzVWMzJIMTIxLjAyQzEyMy41IDMyIDEyNS4wNSA0MC41IDEyNS4wNSA0My4yMzU3VjQ4Ljc2NDNDMTI1LjA1IDUxLjUwMTMgMTIzLjUgNjAgMTIxLjAyIDYwSDEwMFYxMDBIMzVWMzJIMTIxLjAyWiIgZmlsbD0iIzY2RkY2NiIvPjwvc3ZnPg==" alt="License">
</div>

# <div align="center">**Text-to-SQL ** • **Chinook DB** • **Live Streamlit Demo**</div>

<div align="center">
  <img alt="Demo" src="./streamlit/demo.gif" width="800"/>
</div>

**Transform natural language questions into executable SQL** - Production-ready Text2SQL pipeline trained on Chinook database.

## **Setup**

```bash
# 1. Clone & Install
git clone https://github.com/yourusername/text2sql-chinook.git
cd text2sql-chinook
pip install -r requirements.txt

# 2. Get Chinook DB
curl -L -o data/chinook.db "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip" && \
unzip -o data/chinook.zip -d data/ && mv data/Chinook_Sqlite.sqlite data/chinook.db

# 3. Generate Training Data (optional - 1000 examples included)
python scripts/generate_data.py

# 4. Train Model 
python src/train.py

# 5. Launch Demo ✨
streamlit run streamlit/app.py
