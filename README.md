# Lënda: Të mësuarit e makinës (Machine Learning)
**Detyra:** Aplikimi i algoritmeve të ML në një domen të zgjedhur.

**Profesor:** Prof. Dr. Lule Ahmedi  
**Asistenti:** Dr. Sc. Mërgim Hoti  
**Semestri:** II - Master, Viti akademik: 2025/26

**Studentët (Grupi 2):**  
- Brahim Sylejmani
- Eljon Shala``

---

## Faza I: Përgatitja dhe Inxhinieria e të Dhënave

Kjo fazë e transformon datasetin orar të karbonit për Kosovën në një dataset ditor të pastruar, të inxhinieruar dhe gati për përdorim analitik dhe modelim.

### Inputi fillestar i Fazës I
Skedarët hyrës:
- `hourly-2021.csv`
- `hourly-2022.csv`
- `hourly-2023.csv`
- `hourly-2024.csv`
- `hourly-2025.csv`

Kolonat fillestare:
```text
Datetime (UTC)
Country
Zone name
Zone id
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
```

### Rrjedha e 8 hapave

#### Hapi 1 - Ngarkimi dhe Bashkimi
- Input: 11 kolona orare
- Output kryesor: `merged_daily_dataset.csv`

Kolonat e output-it:
```text
Country
Zone name
Zone id
date
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
source_file
estimated_hour_flag
hourly_row_count
Datetime (UTC)
estimated_hour_share
```

#### Hapi 2 - Pastrimi i të Dhënave
- Input: `merged_daily_dataset.csv`
- Output kryesor: `cleaned_dataset.csv`

Kolonat e output-it:
```text
Country
Zone name
Zone id
date
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
source_file
estimated_hour_flag
hourly_row_count
Datetime (UTC)
estimated_hour_share
```

#### Hapi 3 - Diskretizimi
- Input: `cleaned_dataset.csv`
- Output kryesor: `dataset_with_target.csv`

Kolonat e output-it:
```text
day
month
year
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
source_file
estimated_hour_flag
hourly_row_count
estimated_hour_share
hour
day_of_week
is_weekend
carbon_intensity_gap
cfe_re_gap
target_quantile_class
```

#### Hapi 4 - Detektimi i Outliers
- Input: `dataset_with_target.csv`
- Output kryesor: `outlier_flags_dataset.csv`

Kolonat shtesë të output-it:
```text
iqr_outlier_count
zscore_outlier_count
isolation_forest_flag
outlier_consensus_count
outlier_consensus_flag
```

#### Hapi 5 - Balancimi dhe Mostrimi
- Input: `outlier_flags_dataset.csv`
- Output-e kryesore:
  - `train_balanced_features.csv`
  - `train_balanced_target.csv`
  - `test_features.csv`
  - `test_target.csv`

Kolonat e matricës së feature-ve:
```text
numeric__day
numeric__month
numeric__year
numeric__Carbon intensity gCO₂eq/kWh (direct)
numeric__Carbon intensity gCO₂eq/kWh (Life cycle)
numeric__Carbon-free energy percentage (CFE%)
numeric__Renewable energy percentage (RE%)
numeric__Data estimated
numeric__estimated_hour_flag
numeric__hourly_row_count
numeric__estimated_hour_share
numeric__hour
numeric__day_of_week
numeric__is_weekend
numeric__carbon_intensity_gap
numeric__cfe_re_gap
categorical__Data source_Electricity Maps Estimation
categorical__Data source_entsoe.eu
categorical__Data source_kostt.com
categorical__Data estimation method_Unknown
```

Kolona e target-it:
```text
target_quantile_class
```

#### Hapi 6 - Agregimi
- Input: `feature_engineered_dataset.csv`
- Output kryesor: `aggregated_dataset.csv`

Kolonat e output-it:
```text
year
month
day_mean
day_sum
Carbon intensity gCO₂eq/kWh (direct)_mean
Carbon intensity gCO₂eq/kWh (direct)_sum
Carbon intensity gCO₂eq/kWh (Life cycle)_mean
Carbon intensity gCO₂eq/kWh (Life cycle)_sum
Carbon-free energy percentage (CFE%)_mean
Carbon-free energy percentage (CFE%)_sum
Renewable energy percentage (RE%)_mean
Renewable energy percentage (RE%)_sum
Data estimated_mean
Data estimated_sum
estimated_hour_flag_mean
estimated_hour_flag_sum
hourly_row_count_mean
hourly_row_count_sum
estimated_hour_share_mean
estimated_hour_share_sum
hour_mean
hour_sum
day_of_week_mean
day_of_week_sum
is_weekend_mean
is_weekend_sum
carbon_intensity_gap_mean
carbon_intensity_gap_sum
cfe_re_gap_mean
cfe_re_gap_sum
row_count_in_month_count
```

#### Hapi 7 - Finalizimi i Datasetit
- Input: `outlier_flags_dataset.csv`
- Output kryesor: `feature_engineered_dataset.csv`

Kolonat e output-it:
```text
day
month
year
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
source_file
estimated_hour_flag
hourly_row_count
estimated_hour_share
hour
day_of_week
is_weekend
carbon_intensity_gap
cfe_re_gap
target_quantile_class
```

#### Hapi 8 - Raporti Final
- Input: rezultatet nga Hapi 1, Hapi 3, Hapi 5 dhe Hapi 7
- Output: `null_heatmap.png`, `class_balance_comparison.png`, `phase1_report.md`, `phase1_report.pdf`

Ky hap nuk prodhon dataset të ri CSV.

### Cili është dataseti final?
Dataseti final kryesor i Fazës I është:
- `./Faza I - Përgatitja e Modelit/Hapi 7 - Finalizimi i Datasetit/feature_engineered_dataset.csv`

Ky është master dataset-i ditor i përgatitur për analiza të mëtejshme dhe si referencë kryesore e Fazës I.

Datasetet e gatshme për modelim në Fazën II janë:
- `./Faza I - Përgatitja e Modelit/Hapi 5 - Balancimi dhe Mostrimi/train_balanced_features.csv`
- `./Faza I - Përgatitja e Modelit/Hapi 5 - Balancimi dhe Mostrimi/train_balanced_target.csv`
- `./Faza I - Përgatitja e Modelit/Hapi 5 - Balancimi dhe Mostrimi/test_features.csv`
- `./Faza I - Përgatitja e Modelit/Hapi 5 - Balancimi dhe Mostrimi/test_target.csv`

---

## Faza II: Analiza dhe Evaluimi i Modeleve

## Qëllimi i Fazës II
Faza II i jep përgjigje pyetjes kryesore të projektit tonë: **a mund të parashikohet me saktësi niveli i intensitetit të karbonit për një ditë të caktuar në Kosovë si `High`, `Medium` ose `Low`, duke u bazuar në veçoritë energjetike dhe kohore të ndërtuara në Fazën I?**

Në këtë fazë nuk është bërë vetëm trajnim mekanik i algoritmeve. Qëllimi ka qenë:
- të krahasohen modele të ndryshme supervised dhe unsupervised,
- të argumentohet pse secili model u përfshi në analizë,
- të interpretohen rezultatet e fituara,
- dhe të diskutohet se çfarë nënkuptojnë ato për natyrën e problemit dhe për hapat e ardhshëm të projektit.

Në këtë kuptim, Faza II është faza ku dataseti final i Fazës I testohet praktikisht si bazë për modelim.

---

## Struktura e Fazës II
```text
Faza II - Analiza dhe evaluimi/
├── README.md
├── STUDY_GUIDE_AL.md
├── STUDY_GUIDE_EN.md
├── phase2_pipeline.py
└── output/
    ├── model_results.csv
    ├── clustering_results.csv
    ├── classification_reports.txt
    ├── results_summary.md
    ├── phase2_train_raw_features.csv
    ├── phase2_train_raw_target.csv
    ├── phase2_test_raw_features.csv
    ├── phase2_test_target.csv
    ├── phase2_train_processed_features.csv
    ├── phase2_test_processed_features.csv
    ├── train_balanced_features.csv
    ├── train_balanced_target.csv
    ├── algorithm_comparison.png
    ├── feature_importance_rf.png
    ├── learning_curves.png
    ├── regularization_effect.png
    ├── mlp_loss_curve.png
    ├── mlp_validation_curve.png
    ├── elbow_and_silhouette.png
    ├── pca_clusters_comparison.png
    └── confusion_matrix_*.png
```

Kjo strukturë e ndan qartë:
- kodin e pipeline-it,
- dokumentimin ndihmës për mbrojtje,
- dhe output-et numerike e vizuale.

---

## Dataseti hyrës dhe arsyetimi i përdorimit
Inputi i vetëm i Fazës II është dataseti final i përpunuar në Fazën I:

- `./Faza I - Përgatitja e Modelit/Hapi 7 - Finalizimi i Datasetit/feature_engineered_dataset.csv`

Ky dataset përmban:
- `1550` rreshta
- `20` kolona
- targetin `target_quantile_class`

Kolonat hyrëse janë:
```text
day
month
year
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
source_file
estimated_hour_flag
hourly_row_count
estimated_hour_share
hour
day_of_week
is_weekend
carbon_intensity_gap
cfe_re_gap
target_quantile_class
```

Ky dataset u përdor sepse përfaqëson versionin më të pastër dhe më të qëndrueshëm të të dhënave pas:
- bashkimit të të dhënave 2021–2025,
- agregimit nga orar në ditor,
- pastrimit logjik dhe imputimit,
- inxhinierisë së veçorive,
- dhe finalizimit të datasetit në Fazën I.

Pra, në Fazën II nuk punohet me të dhëna bruto, por me datasetin final të përgatitur posaçërisht për modelim. Kjo është zgjedhje metodologjikisht korrekte, sepse ndan qartë:
- Fazën I si fazë të përgatitjes së të dhënave,
- Fazën II si fazë të trajnimit, evaluimit dhe interpretimit.

---

## Objektivi i parashikimit
Problemi është formuluar si **klasifikim multiklasor**. Targeti `target_quantile_class` ndan çdo ditë në tri klasa:
- `High`
- `Medium`
- `Low`

Kjo zgjedhje është e rëndësishme sepse:
- e kthen problemin në një formë të qartë klasifikimi,
- e bën krahasimin e algoritmeve më të kuptueshëm,
- dhe mundëson interpretim më intuitiv të rezultateve sesa një vlerë e vazhdueshme regresioni.

---

## Pipeline-i i aplikuar në Fazën II
Pipeline-i i Fazës II ndjek këta hapa:

1. Ngarkohet dataseti final nga Faza I.
2. Ndahen veçoritë (`X`) nga targeti (`y`).
3. Krijohet `train_test_split` me `stratify=y`.
4. Bëhet preprocessing:
   - `StandardScaler` për kolonat numerike
   - `OneHotEncoder` për kolonat kategoriale
5. Kontrollohet balanca e klasave vetëm në training split.
6. Nëse është e nevojshme, aplikohet SMOTE ose ADASYN vetëm në train.
7. Trajnohen 6 algoritme supervised me `GridSearchCV`.
8. Ekzekutohen 2 algoritme unsupervised për analizë krahasuese.
9. Gjenerohen matricat e konfuzionit, grafet krahasuese, learning curves dhe raporte përmbledhëse.

Ky organizim ndjek praktikën e mirë të Machine Learning sepse:
- test set-i ruhet i paprekur deri në fund,
- preprocessing mësohet nga train set-i,
- balancimi, nëse do të aplikohej, do të prekte vetëm train set-in,
- dhe të gjitha modelet testohen mbi të njëjtin test set për krahasim të drejtë.

---

## Ndarja e të dhënave dhe preprocessing
Në run-in aktual janë përdorur këto dimensione:
- Train raw: `1240` rreshta x `19` kolona
- Test raw: `310` rreshta x `19` kolona
- Train processed: `1240` rreshta x `25` veçori

Shpërndarja e klasave në test set:
- `Medium`: 106
- `High`: 105
- `Low`: 99

### Pse u përdor `train_test_split` me `stratify=y`?
Stratifikimi siguron që raporti i klasave të ruhet afërsisht i njëjtë në train dhe test. Kjo është e rëndësishme sepse:
- e bën evaluimin më të drejtë,
- shmang një test set me shpërndarje të shtrembëruar,
- dhe ruan stabilitetin e krahasimit ndërmjet modeleve.

### Pse u përdor `StandardScaler`?
Standardizimi ishte i domosdoshëm për kolonat numerike sepse algoritmet si:
- Logistic Regression,
- SVM,
- dhe MLP

janë të ndjeshme ndaj shkallës së veçorive. Pa standardizim, kolonat me vlera më të mëdha numerike do të dominonin optimizimin.

### Pse u përdor `OneHotEncoder`?
Kolonat kategoriale si `Data source` dhe `Data estimation method` nuk mund të përdoren drejtpërdrejt nga shumica e algoritmeve. One-hot encoding i kthen në forma numerike binare pa krijuar rend artificial ndërmjet kategorive.

### Pse nuk u përdor SMOTE në këtë run?
Balancimi u kontrollua pasi u bë train/test split. Në këtë run, train split rezultoi tashmë mjaft i balancuar, prandaj pipeline-i e kaloi këtë hap me:

- `Skipped (already balanced)`

Kjo është e rëndësishme sepse shmang krijimin e të dhënave sintetike kur nuk ka nevojë reale për to.

---

## Algoritmet e përdorura

| # | Algoritmi | Lloji | Qëllimi |
|---|-----------|-------|----------|
| 1 | Logistic Regression | Supervised – Klasifikim | Baseline linear për të testuar ndarjen lineare të klasave |
| 2 | Random Forest | Supervised – Klasifikim | Model jo-linear për të dhëna tabulare dhe analizë të rëndësisë së veçorive |
| 3 | Gradient Boosting | Supervised – Klasifikim | Model boosting për performancë të lartë në të dhëna tabulare |
| 4 | SVM (Linear) | Supervised – Klasifikim | Testim i kufijve linearë të vendimmarrjes |
| 5 | SVM (RBF) | Supervised – Klasifikim | Testim i kufijve jolinearë të vendimmarrjes |
| 6 | Neural Network (MLP) | Supervised – Klasifikim | Model neuronik për krahasim me algoritmet klasike |
| 7 | K-Means | Unsupervised | Zbulimi i klasterëve natyralë pa etiketa |
| 8 | Agglomerative Clustering | Unsupervised | Krahasim i clustering hierarkik me K-Means |

---

## Arsyetimi për zgjedhjen e algoritmeve

### Algoritmet supervised

**1. Logistic Regression**
- U përfshi si model bazë linear.
- Është i interpretuar lehtë dhe shërben si pikë reference.
- Ndihmon për të testuar nëse klasat ndahen mirë edhe me kufij linearë.

**2. Random Forest**
- Është shumë i përshtatshëm për të dhëna tabulare.
- Kap marrëdhënie jolineare mes veçorive.
- Është më robust ndaj ndërveprimeve të ndërlikuara dhe jep `feature importance`, çka është e dobishme për interpretim.

**3. Gradient Boosting**
- U përdor si model boosting i fuqishëm për të dhëna tabulare.
- Ndërton pemë sekuencialisht duke korrigjuar gabimet e mëparshme.
- Shërben si krahasim i drejtpërdrejtë me Random Forest.

**4. SVM (Linear)**
- U përdor për të testuar nëse problemi ka strukturë lineare të mjaftueshme.
- Jep krahasim të mirë me Logistic Regression.

**5. SVM (RBF)**
- U përfshi për të testuar nëse kufijtë ndërmjet klasave janë më shumë jolinearë.
- Është më fleksibil se varianti linear, prandaj përdoret si krahasim metodologjik.

**6. Neural Network (MLP)**
- U përfshi sepse rrjetat neurale janë pjesë qendrore e lëndës.
- Shërben për të testuar nëse një model më fleksibil neuronik sjell përmirësim.
- Gjeneron edhe loss curve dhe validation curve për diskutim teorik.

### Algoritmet unsupervised

**7. K-Means**
- U përdor për të parë nëse të dhënat formojnë grupe natyrale pa etiketa.
- Është efikas dhe i interpretuar lehtë në të dhëna tabulare.

**8. Agglomerative Clustering**
- U përdor si alternativë hierarkike ndaj K-Means.
- Ndihmon për të parë nëse një qasje tjetër grupimi jep strukturë më të qartë.

---

## Strategjia e evaluimit
Për secilin model supervised u përdor:
- `GridSearchCV`
- `3-fold cross-validation`
- metrika kryesore e optimizimit: `F1 (macro)`

### Pse `GridSearchCV`?
Grid search teston kombinime të ndryshme hiperparametrash dhe zgjedh konfigurimin më të mirë sipas një metrike të caktuar. Kjo e bën krahasimin:
- më objektiv,
- më sistematik,
- dhe më të besueshëm sesa zgjedhja manuale.

### Pse `F1 (macro)` dhe jo vetëm accuracy?
Accuracy është metrikë e dobishme, por nuk jep gjithmonë pamje të plotë në klasifikim multiklasor. `F1 (macro)`:
- llogaritet veçmas për secilën klasë,
- pastaj i trajton klasat në mënyrë të barabartë,
- dhe është më e përshtatshme kur duam performancë të balancuar në të gjitha klasat.

---

## Rezultatet e evaluimit

### Modelet supervised – krahasimi kryesor
| Model | Best Params | CV F1 (macro) | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | `{"C": 10}` | 0.9847 | 0.9742 | 0.9742 | 0.9744 | 0.9741 |
| Random Forest | `{"max_depth": 10, "n_estimators": 100}` | 0.9968 | 0.9903 | 0.9903 | 0.9905 | 0.9904 |
| Gradient Boosting | `{"learning_rate": 0.05, "max_depth": 3, "n_estimators": 100}` | 0.9984 | 0.9903 | 0.9904 | 0.9905 | 0.9904 |
| SVM (Linear) | `{"C": 10}` | 0.9863 | 0.9710 | 0.9712 | 0.9713 | 0.9709 |
| SVM (RBF) | `{"C": 10, "gamma": "auto"}` | 0.9599 | 0.9645 | 0.9646 | 0.9649 | 0.9645 |
| Neural Network (MLP) | `{"alpha": 0.0001, "hidden_layer_sizes": [128, 64]}` | 0.9766 | 0.9710 | 0.9716 | 0.9712 | 0.9712 |

### Diskutimi i rezultateve supervised
Nga rezultatet vërehen disa përfundime të rëndësishme:
- Të gjithë modelet supervised performuan mirë, me `F1 (macro)` mbi `0.96`.
- Kjo tregon se veçoritë e krijuara në Fazën I janë të fuqishme dhe informative.
- Dy modelet më të mira ishin:
  - `Random Forest`
  - `Gradient Boosting`

Të dy arritën:
- `Accuracy = 0.9903`
- `F1 (macro) = 0.9904`

Ky rezultat sugjeron se të dhënat kanë strukturë jolineare që modelet me pemë e kapin shumë mirë.

Nga ana tjetër:
- Logistic Regression dhe SVM Linear performuan shumë mirë, çka tregon se ekziston edhe komponent linear në problem.
- SVM RBF ishte më i dobëti ndër modelet supervised, gjë që tregon se fleksibiliteti shtesë nuk solli përmirësim real.
- MLP performoi mirë, por nuk e kaloi Random Forest dhe Gradient Boosting. Kjo është tipike për shumë probleme tabulare ku modelet me pemë shpesh dalin më të forta sesa rrjetat neurale.

Nga classification reports vërehet se klasa `Medium` është më sfiduese se `High` dhe `Low`. Kjo është logjike sepse përfaqëson raste kufitare ndërmjet dy klasave të tjera.

### Raportet sipas klasave
Nga `classification_reports.txt` dallohet:
- `High` dhe `Low` parashikohen pak më lehtë
- `Medium` ka më shumë konfuzion në pothuajse të gjitha modelet

Kjo tregon se gabimet nuk janë të rastësishme, por lidhen me strukturën e problemit dhe afërsinë ndërmjet klasave.

### Modelet unsupervised
- K-Means silhouette = `0.2110`
- Agglomerative silhouette = `0.1958`

### Diskutimi i rezultateve unsupervised
Këto vlera janë relativisht të ulëta dhe tregojnë se klasat nuk formojnë klasterë shumë të fortë natyralë pa etiketa. Kjo do të thotë se:
- qasja supervised është më e përshtatshme për këtë problem,
- ndërsa clustering përdoret më tepër për analizë ndihmëse dhe jo si zgjidhje kryesore.

---

## Diskutimi i grafeve të gjeneruara
Në Fazën II janë gjeneruar grafe që mbështesin argumentimin:
- `algorithm_comparison.png`: krahasimi i performancës së modeleve supervised
- `feature_importance_rf.png`: rëndësia e veçorive sipas Random Forest
- `learning_curves.png`: analiza e overfitting/underfitting
- `regularization_effect.png`: efekti i parametrit `C` te Logistic Regression
- `mlp_loss_curve.png` dhe `mlp_validation_curve.png`: konvergjenca dhe stabiliteti i MLP
- `elbow_and_silhouette.png`: sjellja e clustering për vlera të ndryshme të `k`
- `pca_clusters_comparison.png`: krahasimi vizual i clustering me etiketat reale
- `confusion_matrix_*.png`: analiza e gabimeve për secilin model

Këto grafe janë të rëndësishme sepse e bëjnë diskutimin më të argumentuar dhe më të kuptueshëm vizualisht.

---

## Përfundimi i Fazës II
Nga e gjithë analiza mund të nxirren këto përfundime:
1. Dataseti final i Fazës I është i përshtatshëm dhe cilësor për klasifikim.
2. Të gjitha modelet supervised arritën performancë të lartë.
3. `Random Forest` dhe `Gradient Boosting` ishin modelet më të forta.
4. Modelet lineare treguan se ekziston edhe strukturë lineare në problem.
5. MLP dhe SVM RBF nuk sollën përmirësim ndaj modeleve me pemë.
6. Algoritmet unsupervised nuk formuan klasterë të fortë, çka e justifikon qasjen supervised.

Pra, Faza II tregoi jo vetëm se mund të ndërtohen modele shumë të sakta për këtë problem, por edhe pse qasja e zgjedhur është metodologjikisht e qëndrueshme dhe e arsyetuar mirë.

Dokumentimi i plotë i kësaj faze gjendet te:
- `./Faza II - Analiza dhe evaluimi/README.md`

## Faza III: Optimizimi dhe Fine-Tuning i Modeleve

## Qëllimi i Fazës III
Faza III merr pesë modelet më të mira nga Faza II, zgjeron hapësirat e hiperparametrave dhe identifikon një model të vetëm fitues me mbështetje statistikore. SVM (RBF) hiqet plotësisht — ishte modeli më i dobët i Fazës II me CV F1 = 0.9599 dhe fleksibiliteti i tij shtesë nuk solli asnjë përfitim real.

Ndryshimet kryesore metodologjike:
- `GridSearchCV` → `RandomizedSearchCV` me grida të gjera parametrash
- 3-fold CV → **5-fold CV** për vlerësim më të besueshëm
- **Zgjedhja e veçorive** me pragje importancë RF (25 → 9 veçori)
- Shtim i metrikës **ROC-AUC (macro, OvR)**
- **Testi Wilcoxon signed-rank** për konfirmim statistikor të fituesit
- Deklarim i **modelit të vetëm final**

---

## Struktura e Fazës III
```text
Faza III/
├── README.md
├── phase3_pipeline.py
└── output/
    ├── model_results_phase3.csv
    ├── comparison_phase2_vs_phase3.csv
    ├── classification_reports_phase3.txt
    ├── wilcoxon_results.txt
    ├── final_report_phase3.md
    ├── algorithm_comparison_phase3.png
    ├── phase2_vs_phase3_comparison.png
    ├── feature_selection.png
    ├── feature_importance_phase3.png
    ├── learning_curves_phase3.png
    ├── roc_auc_curves_phase3.png
    ├── calibration_curves_phase3.png
    └── confusion_matrix_*.png
```

---

## Dataseti hyrës dhe zgjedhja e veçorive

Inputi mbetet i njëjtë me Fazën II:
- `./Faza I - Përgatitja e Modelit/Hapi 7 - Finalizimi i Datasetit/feature_engineered_dataset.csv`

Megjithatë, Faza III aplikon **zgjedhje veçorish** para trajnimit:
- Trajnohet RF i shpejtë (100 pemë) mbi train set-in e procesuar
- Llogariten importancat e veçorive bazuar në uljen mesatare të papastërtisë Gini
- Ruhen vetëm veçoritë me importancë ≥ 5% × mesatarja e importancave
- **Rezultati: 25 → 9 veçori të mbajtura**

Kjo heq veçoritë me kontribut gati-zero, redukton zhurmën dhe shpejton trajnimin pa humbur performancë.

### Ndarja e të dhënave
- `RANDOM_STATE = 42` — e njëjtë me Fazën II, garanton ndarje identike train/test
- Train: `1,240` rreshta | Test: `310` rreshta
- Pas preprocessing: `25` veçori → pas zgjedhjes: `9` veçori

---

## Algoritmet e përdorura

| # | Algoritmi | Faza II | Faza III | Arsyeja |
|---|---|---|---|---|
| 1 | Logistic Regression | ✅ | ✅ | Baseline linear i ruajtur |
| 2 | Random Forest | ✅ | ✅ | Performancë e lartë, ruhet |
| 3 | Gradient Boosting | ✅ | ✅ | Performancë e lartë, ruhet |
| 4 | SVM (Linear) | ✅ | ✅ | Ruhet, rangu i C zgjerohet ndjeshëm |
| 5 | SVM (RBF) | ✅ | ❌ | **Hiqet** — CV F1=0.9599, më i dobëti |
| 6 | Neural Network (MLP) | ✅ | ✅ | Ruhet, arkitektura zgjerohet |

---

## Strategjia e kërkimit të hiperparametrave

### RandomizedSearchCV
Në vend të testimit exhaustiv (GridSearchCV), RandomizedSearchCV kampionon $n\_iter$ kombinime rastësisht nga hapësira e plotë e parametrave:

- Gradient Boosting: 1,200 kombinime të mundshme → testohen 30
- Random Forest: 540 kombinime → testohen 30
- MLP: 60 kombinime → testohen 30
- Logistic Regression: 11 kombinime → testohen të gjitha (exhaustiv)
- SVM Linear: 9 kombinime → testohen të gjitha (exhaustiv)

Kjo qasje shploron hapësira shumë herë më të mëdha sesa Faza II brenda kohës së ngjashme të ekzekutimit.

### Gridi i zgjeruar i parametrave

| Modeli | Parametri (Faza II) | Parametri (Faza III) |
|---|---|---|
| Logistic Regression | `C: [0.01, 0.1, 1, 10]` | `C: [0.001…100]` (11 vlera) |
| Random Forest | `n_estimators: [100,200]` | `[100,150,200,300,500]` + `min_samples_leaf`, `max_features` |
| Gradient Boosting | `lr: [0.05,0.1]` | `[0.01,0.03,0.05,0.1,0.2]` + `subsample` |
| SVM Linear | `C: [0.1,1,10]` | `C: [0.001…100]` (9 vlera) |
| MLP | `hidden: [(64,32),(128,64)]` | + `(256,128)`, `(128,64,32)`, `(256,128,64)` + `learning_rate_init` |

### 5-fold Cross-Validation
Upgraduar nga 3-fold për vlerësim me bias më të ulët dhe besueshmëri më të lartë — i njëjti objekt `StratifiedKFold` përdoret për të gjitha modelet, duke garantuar fold-et identike dhe krahasim të drejtë.

---

## Rezultatet e Fazës III

### Krahasimi kryesor

| Model | CV F1 (macro) | Accuracy | Precision | Recall | F1 (macro) | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **0.9992** | 0.9903 | 0.9904 | 0.9905 | 0.9904 | **0.9999** |
| Random Forest | 0.9984 | 0.9903 | 0.9904 | 0.9905 | 0.9904 | 0.9999 |
| SVM (Linear) | 0.9904 | 0.9839 | 0.9837 | 0.9840 | 0.9838 | 0.9995 |
| Logistic Regression | 0.9872 | 0.9806 | 0.9806 | 0.9807 | 0.9806 | 0.9993 |
| Neural Network (MLP) | 0.9871 | 0.9742 | 0.9745 | 0.9746 | 0.9743 | 0.9989 |

### Parametrat më të mirë të gjetur

| Model | Parametrat më të mirë |
|---|---|
| Gradient Boosting | `n_estimators=500, lr=0.2, max_depth=5, subsample=0.9, min_samples_split=5` |
| Random Forest | `n_estimators=100, max_depth=8, min_samples_leaf=4, max_features="log2"` |
| SVM (Linear) | `C=50` |
| Logistic Regression | `C=100` |
| Neural Network (MLP) | `hidden=(64,32), alpha=0.001, lr_init=0.005` |

### Raportet sipas klasave — modeli fitues (Gradient Boosting)

| Klasa | Precision | Recall | F1 | Mbështetja |
|---|---|---|---|---|
| High | 1.00 | 0.98 | 0.99 | 105 |
| Low | 0.99 | 1.00 | 0.99 | 99 |
| Medium | 0.98 | 0.99 | 0.99 | 106 |

---

## Testimi Statistikor — Wilcoxon Signed-Rank

Testi Wilcoxon (njëanësor, α = 0.05) mbi 5 fold-et CV konfirmon nëse Gradient Boosting është **statistikisht superior** ndaj modeleve të tjera:

| Krahasimi | W+ | p-vlerë | Domethënës? |
|---|---|---|---|
| GB vs Logistic Regression | 15.0 | 0.0312 | **PO** |
| GB vs Neural Network (MLP) | 15.0 | 0.0312 | **PO** |
| GB vs SVM (Linear) | 10.0 | 0.0625 | Jo (kufitar) |
| GB vs Random Forest | 1.0 | 0.5000 | Jo (barabar praktikisht) |

**Gradient Boosting** deklarohet modeli final pasi ka CV F1 më të lartë (0.9992) dhe është statistikisht superior ndaj dy modeleve të dobëta.

---

## Krahasimi Faza II vs Faza III

| Model | Ph2 F1 | Ph3 F1 | Delta | Ph2 CV F1 | Ph3 CV F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9741 | 0.9806 | **+0.0065** | 0.9847 | 0.9872 |
| Random Forest | 0.9904 | 0.9904 | +0.0000 | 0.9968 | 0.9984 |
| Gradient Boosting | 0.9904 | 0.9904 | +0.0000 | 0.9984 | **0.9992** |
| SVM (Linear) | 0.9709 | 0.9838 | **+0.0129** | 0.9863 | 0.9904 |
| Neural Network (MLP) | 0.9712 | 0.9743 | **+0.0031** | 0.9766 | 0.9871 |
| SVM (RBF) | 0.9645 | — *hiqet* | — | 0.9599 | — |

**SVM Linear pati përfitimin më të madh (+0.0129)** — optimumi C=50 ishte jashtë gridit të Fazës II (max C=10) plotësisht. Asnjë model nuk u keqësua.

---

## Diskutimi i grafeve të gjeneruara

- `algorithm_comparison_phase3.png`: krahasimi i 5 modeleve me 4 metrika
- `phase2_vs_phase3_comparison.png`: shtylla krah-për-krah me delta-t të shënuara
- `feature_selection.png`: importancat e veçorive (blu = mbajtur, kuq = hequr) me vijën e pragut
- `feature_importance_phase3.png`: top veçori sipas RF pas zgjedhjes
- `learning_curves_phase3.png`: kurba trajnimi vs. validimi (parametrat finalë të RF)
- `roc_auc_curves_phase3.png`: kurba ROC makro-mesatare për të 5 modelet
- `calibration_curves_phase3.png`: probabiliteti i parashikuar vs. fraksioni aktual sipas klasës
- `wilcoxon_results.txt`: raporti i plotë i testit statistikor
- `confusion_matrix_*.png`: 5 matrica konfuzioni (një për model)

---

## Përfundimi i Fazës III

```
Modeli Final   : Gradient Boosting
Parametrat     : n_estimators=500, learning_rate=0.2, max_depth=5,
                 subsample=0.9, min_samples_split=5
CV F1 (macro)  : 0.9992
Accuracy       : 0.9903  (307/310 të sakta)
F1 (macro)     : 0.9904
ROC-AUC (macro): 0.9999
```

Faza III konfirmoi se:
1. Zgjerimi i hapësirës së hiperparametrave solli përmirësim të matur për çdo model
2. Gradient Boosting është fitues i qartë me CV F1 = 0.9992 dhe superioriteti i tij mbi dy modele është statistikisht i konfirmuar
3. Random Forest dhe GB mbeten praktikisht të barabartë në test set — dallimi shihet vetëm në CV
4. Dataseti i Fazës I dhe inxhinieria e veçorive ishin të cilësisë së lartë — konfirmohet përfundimisht

Dokumentimi i plotë i kësaj faze gjendet te:
- `./Faza III/README.md`

---
