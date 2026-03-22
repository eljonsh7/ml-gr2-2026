# Hapi 5: Balancimi dhe Mostrimi (Sampling & Balancing)

### **Qëllimi**
Ky hap përgatit të dhënat për trajnimin e modeleve duke krijuar ndarjet **Train/Test** dhe duke u siguruar që modeli nuk do të jetë i anshëm ("biased") ndaj ndonjë klase specifike.

---

### **Veprimet kryesore:**
1.  **Outlier Removal:** Identifikojmë dhe heqim anomalitë duke përdorur sistemin e konsensusit (nga Hapi 4). Rreshtat anomali fshihen automatikisht.
2.  **Preprocessing:** Aplikojmë **StandardScaler** dhe **OneHotEncoder**.
3.  **Train-Test Split:** Të dhënat ndahen në 80% Trajnim dhe 20% Testim.
4.  **Balancimi (SMOTE/ADASYN):** Nëse klasat janë të pabalancuara, shtohen rekordet artificiale.

---

### **Struktura e Skedarëve**
- **Input:** Dataseti me targetin `../Hapi 3 - Diskretizimi/dataset_with_target.csv`.
- **Output-et e gjeneruara:**
    - `train_balanced_features.csv`: Veçoritë (X) për trajnim.
    - `train_balanced_target.csv`: Targeti (y) për trajnim.
    - `test_features.csv` & `test_target.csv`: Seti i pavarur për vlerësimin.
    - `class_distribution_after_balancing.csv`: Konfirmimi i balancës finale.

---

### **Si të ekzekutohet?**
```bash
python step_5.py
```
