# Hapi 2: Pastrimi dhe Trajtimi i vlerave Munguese

### **Qëllimi**
Ky hap realizon pastrimin e datasetit ditor përmes kontrolleve logjike dhe mbushjes së vlerave munguese (imputimit), duke garantuar që të dhënat të jenë matematikisht konsistente për modelim.

---

### **Veprimet kryesore algoritmike:**
1.  **Logical Bounds (Kufijtë Logjikë):** Janë definuar rregulla strikte për kolonat specifike:
    - Kolonat me përqindje (%) duhet të jenë midis **0.0 dhe 100.0**.
    - Kolonat e intensitetit dhe numërimit duhet të jenë **jo-negative**.
    - Datat nuk mund të jenë në të ardhmen.
2.  **Constraint Filtering:** Skripti identifikon çdo rresht që thyen këto rregulla dhe e filtron atë.
3.  **Imputimi Inteligjent:** 
    - Për variablat numerike përdoret **Mediana** (më robuste ndaj outliers).
    - Për variablat kategoriale përdoret **Moda** (vlera më e shpeshtë).
    - *Shënim:* Në datasetin aktual, cilësia ishte shumë e lartë dhe nuk u gjetën vlera Null.

---

### **Struktura e Skedarëve**
- **Input:** Dataseti i agreguar ditor `../Hapi 1/merged_daily_dataset.csv`.
- **Output-et e gjeneruara:**
    - `cleaned_dataset.csv`: Dataseti i pastruar dhe gati për procese më komplekse.
    - `cleaning_log.csv` & `imputation_log.csv`: Regjistrat e të gjitha veprimeve të pastrimit.
    - `logical_bounds.json`: Konfigurimi i kufijve logjikë të përdorur.

---

### **Si të ekzekutohet?**
```bash
python step_2.py
```

