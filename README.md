# AgentML Studio 

**A free and easy to use Agentic AI system for automated data analysis and machine learning**

Built with LangChain + LangGraph + FastAPI + React

---

## What is AgentML Studio?

AgentML Studio is a free agentic AI web application that automatically performs the complete data analysis and machine learning process for any user without requiring any technical knowledge. The user simply uploads a file and the system automatically handles everything from reading the file to selecting the best machine learning model and explaining the results in simple everyday language.

This project was built as a Bachelor's Thesis at HAMK University of Applied Sciences in 2026.

---

## Features

-  **Automated ML Pipeline** — Upload a CSV file and the system automatically cleans the data, selects the best features, trains multiple machine learning models, and deploys the best one
-  **Multi File Support** — Supports CSV, Excel, Word, PDF, Images, and Text files
-  **AutoML** — Automatically tests 10 machine learning algorithms and selects the best one
-  **Explainable AI** — Results are explained in simple everyday language
-  **Live Updates** — Real time pipeline updates through WebSocket connection
-  **Feature Importance** — Visual chart showing which features had the most impact
-  **Model Comparison** — Table showing all tested algorithms and their scores
-  **Auto Deployment** — Models above 82% accuracy are automatically deployed
-  **Completely Free** — No paid APIs or services required

---

## System Architecture

The system consists of three main parts:
<img width="943" height="849" alt="image" src="https://github.com/user-attachments/assets/b65147ea-e3e0-4895-b307-5fcf6065aed3" />


```
Frontend (React + Node.js)
        ↕ REST API + WebSocket
Backend (Python + FastAPI)
        ↕
Orchestrator (LangChain + LangGraph)
        ↕
┌─────────────────────────────────┐
│  Data    Model    File    Deploy │
│  Agent   Agent   Agent   Agent  │
└─────────────────────────────────┘
```

---

## Machine Learning Algorithms

### Classification
| Algorithm | Primary Metric |
|---|---|
| Random Forest | Accuracy |
| Gradient Boosting | Accuracy |
| Logistic Regression | Accuracy |
| Decision Tree | Accuracy |
| Support Vector Machine | Accuracy |

### Regression
| Algorithm | Primary Metric |
|---|---|
| Linear Regression | R Squared |
| Random Forest Regressor | R Squared |
| Gradient Boosting Regressor | R Squared |
| Decision Tree Regressor | R Squared |
| Support Vector Regressor | R Squared |

---

## Technologies Used

### Backend
- **Python** — Main backend language
- **FastAPI** — REST API and WebSocket server
- **LangChain** — Building and defining AI agent tools
- **LangGraph** — Connecting and managing the multi agent pipeline
- **Scikit Learn** — Machine learning model training and selection
- **Pandas** — Data cleaning and processing
- **NumPy** — Numerical computations
- **PyMuPDF** — Reading PDF files
- **Python-docx** — Reading Word documents
- **Pillow** — Processing image files
- **Ollama TinyLlama** — Free local AI model for file summarization
- **Joblib** — Saving and loading trained models

### Frontend
- **React** — User interface
- **Node.js** — Frontend runtime
- **Vite** — Build tool
- **Axios** — HTTP requests

---

## Test Results

| Dataset | Task | Best Model | Accuracy | Decision |
|---|---|---|---|---|
| Titanic | Classification | Random Forest | 95.1% | Auto Deployed |
| Wine Quality | Regression | Random Forest Regressor | 92.2% | Auto Deployed |
| India Hockey Match Data | Regression | Linear Regression | 99.9% | Auto Deployed |

## Screenshot of Home page
<img width="897" height="646" alt="image" src="https://github.com/user-attachments/assets/dace2e01-bdbd-4a4a-976a-a249154e93cb" />
## Result page screenshot
<img width="1063" height="649" alt="image" src="https://github.com/user-attachments/assets/d770b8f1-e955-4fc2-80bc-c40585858735" />

---

## Deployment Decision Logic

| Accuracy | Decision |
|---|---|
| Above 82% | AUTO DEPLOYED |
| Between 65% and 82% | NEEDS HUMAN REVIEW |
| Below 65% | DEFERRED |

---

## Installation and Setup

### Requirements
- Python 3.10 or higher
- Node.js 18 or higher
- Ollama installed locally

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/AgentML-Studio.git
cd AgentML-Studio
```

### Step 2 — Install Ollama and TinyLlama
```bash
# Install Ollama from https://ollama.ai
ollama pull tinyllama
```

### Step 3 — Set up the backend
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4 — Set up the frontend
```bash
cd frontend
npm install
```

### Step 5 — Create environment file
Create a **.env** file in the backend root folder:
```
HUGGINGFACE_TOKEN=your_token_here
```

### Step 6 — Run the backend
```bash
# In the backend folder
uvicorn main:app --reload
```
Backend runs at: **http://localhost:8000**

### Step 7 — Run the frontend
```bash
# In the frontend folder
npm run dev
```
Frontend runs at: **http://localhost:5173**

---

## How to Use

### ML Pipeline Tab
1. Open the application at **http://localhost:5173**
2. Click on the **ML Pipeline** tab
3. Upload any CSV file
4. Watch the pipeline run automatically in real time
5. View the results including accuracy, best model, and feature importance

### File Analyser Tab
1. Click on the **File Analyser** tab
2. Upload any supported file type
3. Optionally type a specific question about the file
4. View the automatic AI analysis of the file

---

## Project Structure

```
AgentML-Studio/
├── agents/
│   ├── data_agent.py       # Data cleaning and feature engineering
│   ├── model_agent.py      # Machine learning model training and selection
│   ├── file_agent.py       # File processing for multiple file types
│   └── orchestrator.py     # LangGraph pipeline management
├── data/                   # Uploaded files (temporary)
├── models/                 # Saved trained models
├── main.py                  # FastAPI backend
├── requirements.txt          # Python dependencies   
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React application
│   │   └── main.jsx        # React entry point
│   ├── package.json
│   └── vite.config.js               
├── .gitignore
└── README.md
```

---

## Requirements.txt

```
fastapi
uvicorn
pandas
scikit-learn
langchain
langgraph
langchain-ollama
langchain-core
python-multipart
python-dotenv
pymupdf
python-docx
pillow
joblib
numpy
huggingface-hub
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /api/run | Upload CSV and run ML pipeline |
| POST | /api/analyse-file | Upload any file for analysis |
| GET | /api/health | Check if backend is running |
| GET | /api/supported-types | List all supported file types |
| WS | /ws | WebSocket for live pipeline updates |

---

## Supported File Types

| File Type | Extension | Feature |
|---|---|---|
| CSV | .csv | ML Pipeline + File Analyser |
| Excel | .xlsx .xls | File Analyser |
| Word | .docx .doc | File Analyser |
| PDF | .pdf | File Analyser |
| Image | .jpg .png .gif | File Analyser |
| Text | .txt .md | File Analyser |

---

## Known Issues

- Logistic Regression may produce a convergence warning on some datasets. This can be fixed by increasing max_iter to 1000
- Support Vector Machine performs better when data is scaled. Data scaling is identified as a future improvement
- Advanced AI image analysis requires a paid HuggingFace account. Basic image metadata is extracted for free

---

## Future Improvements

- Add data scaling for Support Vector Machine
- Increase Logistic Regression max_iter to 1000
- Add support for Excel files in the ML pipeline
- Add real time data streaming
- Add deep learning models using TensorFlow or PyTorch
- Add user account system for saving results
- Deploy to cloud platform such as Railway or Render

---

## License

This project is open source and freely available for anyone to use or improve.

---

## Author

**Neeru Neeru**
Bachelor's Thesis — Degree Programme in Computer Applications
HAMK University of Applied Sciences — Spring 2026

---

## Acknowledgements

This thesis was supervised by **Kevin Cheng** at HAMK University of Applied Sciences.

---

## References

- LangChain Documentation: https://docs.langchain.com
- LangGraph Documentation: https://langchain-ai.github.io/langgraph
- FastAPI Documentation: https://fastapi.tiangolo.com
- Scikit Learn Documentation: https://scikit-learn.org
- React Documentation: https://react.dev
