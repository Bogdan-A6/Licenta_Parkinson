# Leveraging fMRI Data for Early Prediction of Parkinson's Disease

Acest repository conține codul, pipeline-ul de preprocesare și modelele de Machine Learning dezvoltate pentru lucrarea de licență privind predicția timpurie a bolii Parkinson folosind date de imagistică prin rezonanță magnetică funcțională în repaus (*resting-state fMRI* / rs-fMRI).

---

## 1. Configurarea Mediului de Lucru (Reproductibilitate pe orice laptop)

Pentru a evita conflictele între biblioteci și pentru a rula proiectul identic pe orice sistem de operare (Windows, macOS, Linux), folosim un mediu virtual Python (`.venv`).

### Cerințe preliminare
* **Python 3.10 sau 3.11** (recomandat pentru compatibilitate optimă cu bibliotecile de neuroimagistică și ML)
* **Git** (opțional, pentru versionarea codului)

### Pasul 1: Clonarea / Deschiderea proiectului
Deschide un terminal (Command Prompt / PowerShell pe Windows, sau Terminal pe macOS/Linux) în folderul rădăcină al proiectului.

### Pasul 2: Crearea mediului virtual (`.venv`)
Rulează comanda corespunzătoare sistemului tău de operare:

* **Windows:**
  ```bash
  python -m venv .venv
  ```

* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  ```

### Pasul 3: Activarea mediului virtual
După activare, în stânga liniei de comandă trebuie să apară prefixul `(.venv)`.

* **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Notă: Dacă primești o eroare de tip `ExecutionPolicy`, rulează înainte: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

* **Windows (Command Prompt / cmd):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Pasul 4: Instalarea dependențelor
Cu mediul `.venv` activat, actualizează `pip` și instalează pachetele necesare:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Dacă fișierul `requirements.txt` nu este încă creat, rulează comanda de instalare directă de mai jos și apoi generează fișierul)*:

```bash
pip install nibabel nilearn numpy pandas scipy scikit-learn matplotlib seaborn jupyter
pip freeze > requirements.txt
```

### Pasul 5: Dezactivarea mediului (când termini lucrul)
```bash
deactivate
```

---

## 2. Structura Recomandată a Proiectului

Pentru a păstra codul curat și a nu încărca datele medicale brute în Git (au dimensiuni foarte mari), organizăm proiectul astfel:

```text
parkinson-fmri-prediction/
├── .venv/                  # Mediul virtual Python (NU se urcă pe Git)
├── .gitignore              # Ignoră .venv/, datele brute (.nii.gz) și fișierele temporare
├── README.md               # Documentația principală a proiectului (acest fișier)
├── requirements.txt        # Lista exactă a bibliotecilor și versiunilor instalate
│
├── data/                   # Datele descărcate (NU se urcă pe Git)
│   └── ds001907/           # Dataset-ul OpenNeuro (ANT: Healthy aging and Parkinson's disease)
│       ├── demographics.csv
│       ├── derivatives/    # Datele deja preprocesate cu AFNI
│       └── sub-RC4101/ ...
│
├── notebooks/              # Jupyter Notebooks pentru explorare și vizualizări
│   └── 01_data_exploration.ipynb
│
├── src/                    # Codul sursă modular Python
│   ├── __init__.py
│   ├── explore_fmri.py     # Script de verificare a dimensiunilor și vizualizare creier 3D
│   ├── extract_features.py # Extragere serii temporale (Atlas) + Matrici de conectivitate
│   └── train_baseline.py   # Antrenare și evaluare modele ML (SVM, Logistic Regression, RF)
│
└── results/                # Grafice, matrici de confuzie și imagini generate pentru licență
```

---

## 3. Descrierea Setului de Date Curent (`ds001907`)

* **Sursă:** OpenNeuro (`ds001907` — *ANT: Healthy aging and Parkinson's disease*)
* **Structură participanți (46 subiecți total, 2 sesiuni fiecare):**
  * **Grup Control (Sănătoși):** `n = 25`, coduri de tip `sub-RC41*` (ex: `sub-RC4101`)
  * **Grup Parkinson (PD):** `n = 21`, coduri de tip `sub-RC42*` (ex: `sub-RC4201`)
* **Tip date utilizate:** *Resting-state fMRI* (`task-rest`), format NIfTI 4D (`.nii.gz`), atât date brute (`sub-*/ses-*/func/`), cât și date preprocesate (`derivatives/`).

---

## 4. Jurnal de Progres & Roadmap

- [x] **Etapa 1:** Identificarea și descărcarea dataset-ului public (`ds001907`).
- [x] **Etapa 2:** Configurarea mediului reproductibil (`.venv` + `requirements.txt`).
- [ ] **Etapa 3:** Script de explorare a datelor (`src/explore_fmri.py`) – citire `demographics.csv` și vizualizare scanare fMRI medie.
- [ ] **Etapa 4:** Extragerea seriilor temporale pe regiuni cerebrale folosind un atlas (ex. *Harvard-Oxford* / *AAL*) și calcularea matricilor de conectivitate funcțională.
- [ ] **Etapa 5:** Antrenarea unui model baseline (SVM / Logistic Regression) cu *Cross-Validation* și evaluarea metricilor (AUC-ROC, Accuracy, Sensitivity, Specificity).
- [ ] **Etapa 6:** Interpretabilitate (identificarea conexiunilor cerebrale relevante pentru boala Parkinson) și integrarea datelor din **PPMI** (cohorta *Prodromal*).