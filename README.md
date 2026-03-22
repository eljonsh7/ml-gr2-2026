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
**Data e dorëzimit:** 15.03.2026 | **Pikët:** 5

Kjo fazë konsiston në përgatitjen e të dhënave, trajtimin e vlerave që mungojnë, balancimin e klasave dhe detektimin e "outliers". 

### Pershkrimi i Datasetit
Të dhënat janë nxjerrë nga platforma **Electricity Maps** për zonën e Kosovës (XK) dhe mbulojnë periudhën 2021 - 2025. Ky dataset tregon intensitetin e karbonit dhe energjinë e rinovueshme në intervale orare.

- **Burimi:** [Electricity Maps - Zone XK (Kosovo)](https://app.electricitymaps.com/datasets?zone=XK&year=2025&interval=hourly)
- **Numri i Atributeve:** 11 atribute fillestare të cilat u zgjeruan përmes inxhinierisë së veçorive, dhe më vonë arritën në 30.
- **Numri i Objekteve:** 43,824 rreshta kohorë (orare) -> Të agreguara në 1,826 ditë (rreshta ditorë).
- **Atributet Kryesore:**
  - `Datetime (UTC)`: Data dhe ora e matjes.
  - `Carbon intensity gCO₂eq/kWh (direct)`: Intensiteti direkt i karbonit (Kjo është zgjedhur si variabla e targetit për këtë projekt).
  - `Carbon intensity gCO₂eq/kWh (Life cycle)`: Intensiteti i ciklit jetësor.
  - `Carbon-free energy percentage (CFE%)`: Përqindja e energjisë pa karbon.
  - `Renewable energy percentage (RE%)`: Përqindja e energjisë së rinovueshme.
  - `Data estimated` & `Data estimation method` etj.

### Arkitektura e Fazës 1 (Pipeline)
Sistemi përbëhet nga 8 Hapa kryesorë, të automatizuar në kod, me readmes përkatëse paralelisht:
1. **[Step 1: Initial Data Overview](./Faza%20I/Step%201/README.md):** Leximi dhe bashkimi i skedarëve (*hourly-2021.csv ... hourly-2025.csv*), dhe profilizimi i skemës së të dhënave fillestare.
2. **[Step 2: Core Data Cleaning](./Faza%20I/Step%202/README.md):** Trajtimi i vlerave "null", "NaN" dhe pastrimi i atributeve.
3. **[Step 3: Class Definition & Imbalance Metrics](./Faza%20I/Step%203/README.md):** Krijimi i `target_quantile_class` (`low`, `medium`, `high`) nga klasifikimi i `Carbon intensity`. Shpërndarja ishte rreth 33.3% në secilën klasë.
4. **[Step 4: Sampling & Balancing](./Faza%20I/Step%204/README.md):** Balancimi (SMOTE / ADASYN i anashkaluar si i panevojshëm shkaku i hapit 3) dhe ndarja strukturore e test/train set.
5. **[Step 5: Data Aggregation](./Faza%20I/Step%205/README.md):** Agregimi mujor i të dhënave.
6. **[Step 6: Subsets & Transformations](./Faza%20I/Step%206/README.md):** Shtimi i "temporal features" (ora, dita e javës, vikendi, muaji) dhe atributeve të derivuara energjetike.
7. **[Step 7: Multi-Method Outlier Detection](./Faza%20I/Step%207/README.md):** Aplikimi i metodave z-score, IQR, dhe Isolation Forest. Detektim konsensual i anomalisë.
8. **[Step 8: Presentation & Reporting](./Faza%20I/Step%208/README.md):** Gjenerimi terminal i raporteve statistikore the grafikoneve.

### Arsyetimi i Teknikave të Përdorura (Justification)
- **Cilësia e të dhënave:** U përdorën filtrime rigoroze në **Step 1 dhe 2** për të garantuar integritetin rregullator të tipit *timeseries*, pastruar dimensionet dhe përgatitur vizualin.
- **Balancimi i klasave në zgjedhje analitike:** Për shkak se intensiteti i karbonit u vu theks në vlerësimet numerike të shpërndara jo ndryshe nga normalja, përdorimi i parimit *Quantile Cuts* (Step 3) bëri të mundur që 1826 ditë të kenë saktësisht ~608 të tilla në secilën prej ndarjeve (`low`, `medium`, `high`). Kjo pamundëson mbështetjen e tepërt në SMOTE ose gjenerim thellësisht fiktiv artificial (ADASYN), duke i ruajtur vlerat sa më afër realitetit thelbësor shkencor të ambientit në Fazën 2.
- **Drejtimi Anomalo i Outlier Detection:** Në vrojtimet energjetike rregullat standard normal ndonjëherë gabohen gjatë periudhave të ndërprerjes (p.sh defekte blloku B2 vitin e kaluar etj.). Për shkak të kësaj ne kombinuam vizualizimin me standardin e viltë IQR si dhe tekniken e pathyeshme *Isolation Forest*. Një rresht shënohej anomali ekskluzivisht vetëm në momentin ekzekutiv të **Consensus Consensus (p.sh TË PAKTËN 2 ndarje detektive pajtohen)** (Step 7), duke gjetur total prej 310 të tillash.

---
*(Shënim: Grupi duhet të zëvendësojë emrat e tyre më lart. Asnjë *`git commit`* nuk duhet bërë pas datës së dorëzimit 15 Mars pa patur lejen ekskurzive të asistentit për t'u mbrojtur thelbësisht prej rregullave akademike të lëndës.)*
