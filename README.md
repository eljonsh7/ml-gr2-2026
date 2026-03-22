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
- **Atributet Kryesore (Hapi 3-7):**
  - `day`, `month`, `year`: Kolonat kryesore të vendosura në fillim të datasetit.
  - `Carbon intensity gCO₂eq/kWh (direct)`: Variabla e targetit.
  - `hour`, `is_weekend`, `carbon_intensity_gap`: Veçori të inxhinieruara.
  - (Shënim: Kolonat `Country` dhe `Zone` janë fshirë për të evituar zhurmën).

### Arkitektura e Fazës 1 (Modular Pipeline)
Sistemi përbëhet nga 8 Hapa Modularë, të organizuar në folderat përkatës:
1. **[Hapi 1 - Ngarkimi dhe Bashkimi](./Faza%20I/Hapi%201%20-%20Ngarkimi%20dhe%20Bashkimi/README.md):** Bashkimi i skedarëve orarë.
2. **[Hapi 2 - Pastrimi i te dhenave](./Faza%20I/Hapi%202%20-%20Pastrimi%20i%20te%20dhenave/README.md):** Trajtimi i vlerave munguese.
3. **[Hapi 3 - Diskretizimi](./Faza%20I/Hapi%203%20-%20Diskretizimi/README.md):** Inxhinieria e veçorive dhe heqja e kolonave redundante.
4. **[Hapi 4 - Detektimi i Outliers](./Faza%20I/Hapi%204%20-%20Detektimi%20i%20Outliers/README.md):** Auditimi dhe flagging i anomalive.
5. **[Hapi 5 - Balancimi dhe Mostrimi](./Faza%20I/Hapi%205%20-%20Balancimi%20dhe%20Mostrimi/README.md):** **Fshirja e Outliers** (276 rreshta) dhe ndarja Train/Test.
6. **[Hapi 6 - Agregimi](./Faza%20I/Hapi%206%20-%20Agregimi/README.md):** Analiza mujore e trendeve.
7. **[Hapi 7 - Finalizimi i Datasetit](./Faza%20I/Hapi%207%20-%20Finalizimi%20i%20Datasetit/README.md):** Gjenerimi i datasetit të pastër përfundimtar.
8. **[Hapi 8 - Raporti Final](./Faza%20I/Hapi%208%20-%20Raporti%20Final/README.md):** Dokumentimi vizual dhe PDF.

### Arsyetimi i Teknikave të Përdorura (Justification)
- **Cilësia e të dhënave:** U përdorën filtrime rigoroze në **Step 1 dhe 2** për të garantuar integritetin rregullator të tipit *timeseries*, pastruar dimensionet dhe përgatitur vizualin.
- **Balancimi i klasave në zgjedhje analitike:** Për shkak se intensiteti i karbonit u vu theks në vlerësimet numerike të shpërndara jo ndryshe nga normalja, përdorimi i parimit *Quantile Cuts* (Step 3) bëri të mundur që 1826 ditë të kenë saktësisht ~608 të tilla në secilën prej ndarjeve (`low`, `medium`, `high`). Kjo pamundëson mbështetjen e tepërt në SMOTE ose gjenerim thellësisht fiktiv artificial (ADASYN), duke i ruajtur vlerat sa më afër realitetit thelbësor shkencor të ambientit në Fazën 2.
- **Drejtimi i Outlier Detection:** Në vrojtimet energjetike rregullat standard normal ndonjëherë gabojnë (p.sh defekte blloku B2 vitin e kaluar etj.). Për shkak të kësaj ne kombinuam 3 metoda: Z-Score, IQR dhe Isolation Forest. Një rresht shënohet anomali vetëm nëse të paktën **2 metoda** pajtohen. Në versionin final, këto anomali (276 rreshta) janë fshirë për të garantuar modelim më të saktë.

---

## Faza II: Ndërtimi i Modeleve (Në vijim)
Kjo fazë do të fokusohet në trajnimin e modeleve të ndryshme të klasifikimit (p.sh. Random Forest, SVM, Neural Networks) dhe vlerësimin e saktësisë së tyre.

## Faza III: Interpretueshmëria dhe Optimizimi (Në vijim)
Analiza e rëndësisë së veçorive, interpretimi i vendimeve të modelit dhe optimizimi i parametrave finalë.

---
*(Shënim: Grupi duhet të zëvendësojë emrat e tyre më lart.)*
