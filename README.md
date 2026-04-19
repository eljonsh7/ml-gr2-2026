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

## Faza II: Analiza dhe Evaluimi i Modeleve
Kjo fazë shërben për ndërtimin, krahasimin dhe diskutimin e modeleve të Machine Learning për parashikimin e klasës së intensitetit të karbonit. Në këtë fazë nuk është bërë vetëm ekzekutimi i algoritmeve, por edhe arsyetimi i zgjedhjes së tyre, diskutimi i rezultateve të fituara dhe interpretimi i sjelljes së modeleve në raport me natyrën e të dhënave.

### Dataseti hyrës i Fazës II
Inputi i vetëm i Fazës II është dataseti final i përpunuar në Fazën I:
- `./Faza I - Përgatitja e Modelit/Hapi 7 - Finalizimi i Datasetit/feature_engineered_dataset.csv`

Ky dataset përmban:
- 1,550 rreshta
- 20 kolona
- targetin `target_quantile_class`

Kolonat kryesore të përdorura janë:
- `day`, `month`, `year`
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

Ky dataset u zgjodh sepse përfaqëson versionin final dhe më të qëndrueshëm të të dhënave pas gjithë hapave të Fazës I. Kjo ndarje është metodologjikisht e arsyeshme, sepse Faza I trajton përgatitjen e të dhënave, ndërsa Faza II fokusohet ekskluzivisht në modelim dhe evaluim.

### Objektivi i Fazës II
Problemi u formulua si klasifikim multiklasor, ku targeti `target_quantile_class` ndan çdo ditë në tri kategori:
- `High`
- `Medium`
- `Low`

Kjo zgjedhje mundëson analizë më të qartë të modeleve klasifikuese dhe e bën interpretimin e rezultateve më intuitiv sesa një regresion i thjeshtë numerik.

### Hapat metodologjikë të pipeline-it
Në Fazën II janë ndjekur këta hapa:
1. Ngarkimi i datasetit final nga Faza I.
2. Ndarja e veçorive (`X`) nga targeti (`y`).
3. Krijimi i `train_test_split` me `stratify=y`.
4. Aplikimi i preprocessing:
   - `StandardScaler` për kolonat numerike
   - `OneHotEncoder` për kolonat kategoriale
5. Kontrollimi i balancës së klasave vetëm në training split.
6. Aplikimi i SMOTE/ADASYN vetëm nëse do të kishte nevojë.
7. Trajnimi i 6 algoritmeve supervised.
8. Testimi i 2 algoritmeve unsupervised për analizë krahasuese.
9. Gjenerimi i matricës së konfuzionit, learning curves, grafeve krahasuese dhe raporteve përmbledhëse.

Ky organizim ndjek praktikën e mirë të Machine Learning, sepse test set-i ruhet i paprekur, preprocessing mësohet nga train set-i dhe krahasimi i modeleve kryhet mbi të njëjtën bazë testimi.

### Ndarja e të dhënave dhe preprocessing
Në run-in aktual janë përdorur këto dimensione:
- Train raw: 1,240 rreshta x 19 kolona
- Test raw: 310 rreshta x 19 kolona
- Train processed: 1,240 rreshta x 25 veçori

Shpërndarja e klasave në test set:
- `Medium`: 106
- `High`: 105
- `Low`: 99

Përdorimi i `stratify=y` ishte i rëndësishëm për të ruajtur raportin e klasave ndërmjet train dhe test. `StandardScaler` u përdor sepse algoritme si Logistic Regression, SVM dhe MLP janë të ndjeshme ndaj shkallës së kolonave numerike. `OneHotEncoder` u përdor për kolonat kategoriale në mënyrë që ato të transformohen në formë numerike pa krijuar rend artificial ndërmjet kategorive.

Balancimi u kontrollua pas train/test split, por në këtë run u kalua sepse train split ishte tashmë mjaft i balancuar. Kjo është e rëndësishme sepse shmang krijimin e të dhënave sintetike kur nuk ka nevojë reale për to.

### Strategjia e evaluimit
Për secilin model supervised u përdor:
- `GridSearchCV`
- `3-fold cross-validation`
- metrika kryesore e optimizimit: `F1 (macro)`

`GridSearchCV` u zgjodh sepse lejon testimin sistematik të kombinimeve të hiperparametrave dhe zgjedhjen e konfigurimit më të mirë në mënyrë objektive. `F1 (macro)` u përdor sepse është më e përshtatshme sesa vetëm accuracy në probleme multiklasore, pasi trajton të gjitha klasat në mënyrë të barabartë.

### Algoritmet e aplikuara dhe arsyetimi i zgjedhjes
Në këtë fazë u trajnuan 6 algoritme supervised:
- **Logistic Regression:** si model bazë linear dhe i interpretuar lehtë
- **Random Forest:** për kapjen e marrëdhënieve jolineare dhe analizën e rëndësisë së veçorive
- **Gradient Boosting:** si model shumë i fuqishëm për të dhëna tabulare
- **SVM (Linear):** për të testuar ndarjen lineare të klasave
- **SVM (RBF):** për të testuar nëse kufijtë ndërmjet klasave janë më jolinearë
- **Neural Network (MLP):** për të përfshirë qasjen neuronale dhe për të analizuar konvergjencën e trajnimit

Gjithashtu u përdorën 2 algoritme unsupervised:
- **K-Means**
- **Agglomerative Clustering**

Këto u përdorën për të testuar nëse klasat formojnë grupime natyrale edhe pa etiketa.

### Rezultatet kryesore supervised
| Model | Best Params | Accuracy | F1 (macro) |
|---|---|---:|---:|
| Logistic Regression | `{"C": 10}` | 0.9742 | 0.9741 |
| Random Forest | `{"max_depth": 10, "n_estimators": 100}` | 0.9903 | 0.9904 |
| Gradient Boosting | `{"learning_rate": 0.05, "max_depth": 3, "n_estimators": 100}` | 0.9903 | 0.9904 |
| SVM (Linear) | `{"C": 10}` | 0.9710 | 0.9709 |
| SVM (RBF) | `{"C": 10, "gamma": "auto"}` | 0.9645 | 0.9645 |
| Neural Network (MLP) | `{"alpha": 0.0001, "hidden_layer_sizes": [128, 64]}` | 0.9710 | 0.9712 |

### Diskutimi i rezultateve supervised
Nga rezultatet vërehet se të gjithë modelet supervised performuan shumë mirë, me `F1 (macro)` mbi `0.96`. Kjo tregon se veçoritë e ndërtuara në Fazën I janë të fuqishme dhe informative.

Modelet më të mira rezultuan:
- `Random Forest`
- `Gradient Boosting`

Të dy arritën:
- `Accuracy = 0.9903`
- `F1 (macro) = 0.9904`

Kjo sugjeron se të dhënat përmbajnë marrëdhënie jolineare që modelet me pemë i kapin shumë mirë. Nga ana tjetër:
- Logistic Regression dhe SVM Linear performuan shumë mirë, çka tregon se një pjesë e strukturës është lineare
- SVM RBF doli më i dobët se varianti linear, gjë që tregon se fleksibiliteti shtesë nuk solli përfitim real
- MLP performoi mirë, por nuk e kaloi performancën e modeleve me pemë, gjë që është tipike për shumë probleme tabulare

Nga classification reports vërehet se klasa `Medium` është pak më sfiduese se `High` dhe `Low`. Kjo është logjike sepse `Medium` përfaqëson raste kufitare ndërmjet dy klasave të tjera.

### Rezultatet unsupervised
- K-Means silhouette = `0.2110`
- Agglomerative silhouette = `0.1958`

Këto vlera janë relativisht të ulëta dhe tregojnë se klasat nuk formojnë klasterë shumë të fortë në mënyrë natyrale. Kjo nënkupton se problemi është më i përshtatshëm për qasje supervised sesa për clustering pa etiketa.

### Krahasimi i grafeve të gjeneruara
Në Fazën II janë gjeneruar grafe që nuk shërbejnë vetëm si ilustrim, por si pjesë e argumentimit:
- `algorithm_comparison.png`: krahasimi vizual i performancës së modeleve supervised
- `feature_importance_rf.png`: rëndësia e veçorive sipas Random Forest
- `learning_curves.png`: analiza e overfitting/underfitting
- `regularization_effect.png`: efekti i parametrit `C` në Logistic Regression
- `mlp_loss_curve.png` dhe `mlp_validation_curve.png`: konvergjenca dhe stabiliteti i MLP
- `elbow_and_silhouette.png`: vlerësimi i clustering për vlera të ndryshme të `k`
- `pca_clusters_comparison.png`: krahasimi vizual i clustering me etiketat reale
- `confusion_matrix_*.png`: analiza e gabimeve sipas klasave për secilin model

Këto grafe ndihmojnë në mbështetjen vizuale të konkluzioneve dhe e bëjnë analizën më të argumentuar dhe më të kuptueshme.

### Përfundimi i Fazës II
Nga e gjithë analiza mund të nxirren këto përfundime:
1. Dataseti final nga Faza I është i përshtatshëm dhe cilësor për klasifikim.
2. Të gjitha modelet supervised arritën performancë të lartë.
3. `Random Forest` dhe `Gradient Boosting` ishin modelet më të forta.
4. Modelet lineare treguan se ekziston edhe strukturë lineare në problem.
5. MLP dhe SVM RBF nuk sollën përmirësim ndaj modeleve me pemë.
6. Algoritmet unsupervised nuk formuan klasterë të fortë, çka e justifikon qasjen supervised.

Pra, Faza II tregoi jo vetëm se mund të ndërtohen modele shumë të sakta për këtë problem, por edhe pse qasja e zgjedhur është metodologjikisht e qëndrueshme dhe e arsyetuar mirë.

Dokumentimi i plotë i kësaj faze gjendet te:
- `./Faza II - Analiza dhe evaluimi/README.md`

## Faza III: Interpretueshmëria dhe Optimizimi (Në vijim)
Analiza e rëndësisë së veçorive, interpretimi i vendimeve të modelit dhe optimizimi i parametrave finalë.

---
*(Shënim: Grupi duhet të zëvendësojë emrat e tyre më lart.)*
