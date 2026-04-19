# Faza II: Analiza dhe Evaluimi i Modeleve

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

- `../Faza I - Përgatitja e Modelit/Hapi 7 - Finalizimi i Datasetit/feature_engineered_dataset.csv`

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
9. Gjenerohen matricat e konfiyionit, grafet krahasuese, learning curves dhe raporte përmbledhëse.

Ky organizim ndjek praktikën e mirë të Machine Learning sepse:
- test set-i ruhet i paprekur deri në fund,
- preprocessing mësohet nga train set-i,
- balancimi, nëse do të aplikohej, do të prekte vetëm train set-in,
- dhe të gjitha modelet testohen mbi të njëjtin test set për krahasim të drejtë.

---

## Ndarja e të dhënave dhe preprocessin``g
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
