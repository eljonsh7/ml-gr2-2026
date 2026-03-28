# Lënda: Të mësuarit e makinës (Machine Learning)
**Detyra:** Aplikimi i algoritmeve të ML në një domen të zgjedhur.

**Profesor:** Prof. Dr. Lule Ahmedi  
**Asistenti:** Dr. Sc. Mërgim Hoti  
**Semestri:** II - Master, Viti akademik: 2025/26

**Studentët (Grupi 2):**  
- Brahim Sylejmani
- Eljon Shala
- Altin Morina

---

## Faza I: Përgatitja e modelit
Kjo fazë konsiston në përgatitjen e të dhënave, trajtimin e vlerave që mungojnë, balancimin e klasave dhe detektimin e "outliers". 

### Emri dhe Përshkrimi i Datasetit: **XK-CarbonTrace (2021-2025)**
Të dhënat janë nxjerrë nga platforma **Electricity Maps** për zonën e Kosovës (XK) dhe mbulojnë periudhën 2021 - 2025. Ky dataset tregon intensitetin e karbonit dhe energjinë e rinovueshme në intervale orare.

- **Burimi:** [Electricity Maps - Zone XK (Kosovo)](https://app.electricitymaps.com/datasets?zone=XK&year=2025&interval=hourly)
- **Numri i Atributeve:** 11 atribute fillestare të cilat u zgjeruan përmes inxhinierisë së veçorive, dhe më vonë arritën në 30.
- **Numri i Objekteve:** 43,824 rreshta kohorë (orare) -> Të agreguara në 1,826 ditë (rreshta ditorë).
- **Kolonat fillestare të input-it:**
  - `Datetime (UTC)`
  - `Country`
  - `Zone name`
  - `Zone id`
  - `Carbon intensity gCO₂eq/kWh (direct)`
  - `Carbon intensity gCO₂eq/kWh (Life cycle)`
  - `Carbon-free energy percentage (CFE%)`
  - `Renewable energy percentage (RE%)`
  - `Data source`
  - `Data estimated`
  - `Data estimation method`
- **Kolonat e datasetit final të Fazës I:**
  - `day`
  - `month`
  - `year`
  - `Carbon intensity gCO₂eq/kWh (direct)`
  - `Carbon intensity gCO₂eq/kWh (Life cycle)`
  - `Carbon-free energy percentage (CFE%)`
  - `Renewable energy percentage (RE%)`
  - `Data source`
  - `Data estimated`
  - `Data estimation method`
  - `source_file`
  - `estimated_hour_flag`
  - `hourly_row_count`
  - `estimated_hour_share`
  - `hour`
  - `day_of_week`
  - `is_weekend`
  - `carbon_intensity_gap`
  - `cfe_re_gap`
  - `target_quantile_class`

### Arkitektura e Fazës 1 (Modular Pipeline)
Sistemi përbëhet nga 8 Hapa Modularë, të organizuar në folderat përkatës:
1. **[Hapi 1 - Ngarkimi dhe Bashkimi](./Faza%20I%20-%20P%C3%ABrgatitja%20e%20Modelit/Hapi%201%20-%20Ngarkimi%20dhe%20Bashkimi/README.md):** 11 kolona hyrëse orare -> 16 kolona ditore.
2. **[Hapi 2 - Pastrimi i të dhënave](./Faza%20I%20-%20P%C3%ABrgatitja%20e%20Modelit/Hapi%202%20-%20Pastrimi%20i%20t%C3%AB%20dh%C3%ABnave/README.md):** 16 kolona hyrëse -> 16 kolona të pastruara.
3. **[Hapi 3 - Diskretizimi](./Faza%20I%20-%20P%C3%ABrgatitja%20e%20Modelit/Hapi%203%20-%20Diskretizimi/README.md):** 16 kolona hyrëse -> 20 kolona me target dhe feature engineering.
4. **[Hapi 4 - Detektimi i Outliers](./Faza%20I%20-%20P%C3%ABrgatitja%20e%20Modelit/Hapi%204%20-%20Detektimi%20i%20Outliers/README.md):** 20 kolona hyrëse -> 25 kolona me flag-e të anomalive.
5. **[Hapi 5 - Balancimi dhe Mostrimi](./Faza%20I%20-%20P%C3%ABrgatitja%20e%20Modelit/Hapi%205%20-%20Balancimi%20dhe%20Mostrimi/README.md):** 25 kolona hyrëse -> matrica train/test për modelim.
6. **[Hapi 6 - Agregimi](./Faza%20I%20-%20P%C3%ABrgatitja%20e%20Modelit/Hapi%206%20-%20Agregimi/README.md):** 20 kolona hyrëse -> 31 kolona mujore të agreguara.
7. **[Hapi 7 - Finalizimi i Datasetit](./Faza%20I%20-%20P%C3%ABrgatitja%20e%20Modelit/Hapi%207%20-%20Finalizimi%20i%20Datasetit/README.md):** 25 kolona hyrëse -> 20 kolona në datasetin final.
8. **[Hapi 8 - Raporti Final](./Faza%20I%20-%20P%C3%ABrgatitja%20e%20Modelit/Hapi%208%20-%20Raporti%20Final/README.md):** Përmbledhja vizuale dhe raporti PDF/Markdown.

### Arsyetimi i Teknikave të Përdorura (Justification)
- **Cilësia e të dhënave:** Hapi 1 dhe Hapi 2 sigurojnë që dataseti të kalojë nga struktura orare në ditor dhe të kontrollohet me kufij logjikë e imputim.
- **Balancimi i klasave:** Targeti u krijua me diskretizim në tri klasa (`low`, `medium`, `high`). Në run-in aktual klasat dolën tashmë të balancuara, prandaj nuk u shtuan raste sintetike.
- **Drejtimi i Outlier Detection:** Hapi 4 përdor IQR, Z-Score dhe Isolation Forest. Outlier-at flag-ohen me rregull konsensusi 2 nga 3.

### Dataseti final i Fazës I
Dataseti final kryesor është:
- `./Faza I - Përgatitja e Modelit/Hapi 7 - Finalizimi i Datasetit/feature_engineered_dataset.csv`

Datasetet e gatshme për modelim në Fazën II janë:
- `./Faza I - Përgatitja e Modelit/Hapi 5 - Balancimi dhe Mostrimi/train_balanced_features.csv`
- `./Faza I - Përgatitja e Modelit/Hapi 5 - Balancimi dhe Mostrimi/train_balanced_target.csv`
- `./Faza I - Përgatitja e Modelit/Hapi 5 - Balancimi dhe Mostrimi/test_features.csv`
- `./Faza I - Përgatitja e Modelit/Hapi 5 - Balancimi dhe Mostrimi/test_target.csv`

---

## Faza II: Ndërtimi i Modeleve (Në vijim)
Kjo fazë do të fokusohet në trajnimin e modeleve të ndryshme të klasifikimit (p.sh. Random Forest, SVM, Neural Networks) dhe vlerësimin e saktësisë së tyre.

## Faza III: Interpretueshmëria dhe Optimizimi (Në vijim)
Analiza e rëndësisë së veçorive, interpretimi i vendimeve të modelit dhe optimizimi i parametrave finalë.

---
*(Shënim: Grupi duhet të zëvendësojë emrat e tyre më lart.)*
