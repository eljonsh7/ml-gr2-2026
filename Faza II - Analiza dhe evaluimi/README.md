# Faza II: Analiza dhe Evaluimi i Modeleve

## Qëllimi i Fazës II
Qëllimi i kësaj faze është të trajnojmë, krahasojmë dhe diskutojmë modele të ndryshme të Machine Learning për parashikimin e klasës së intensitetit të karbonit në Kosovë. Në këtë fazë nuk mjafton vetëm të paraqiten rezultatet numerike, por duhet të argumentohet:
- pse u zgjodhën teknikat e caktuara,
- çfarë roli luan secila teknikë në problem,
- çfarë tregojnë rezultatet,
- dhe çfarë nënkuptojnë ato për vazhdimin e analizës.

Në këtë mënyrë, Faza II shërben si ura midis përgatitjes së të dhënave në Fazën I dhe interpretimit/analizës më të thellë në fazat pasuese.

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

Ky dataset u përdor sepse përfaqëson versionin më të pastër dhe më të qëndrueshëm të të dhënave, pas:
- bashkimit të të dhënave 2021–2025,
- agregimit nga orar në ditor,
- pastrimit,
- inxhinierisë së veçorive,
- dhe finalizimit të datasetit në Fazën I.

Kjo do të thotë se në Fazën II nuk u nisëm nga të dhëna bruto, por nga një dataset i përgatitur posaçërisht për modelim. Ky është një vendim i arsyeshëm metodologjik, sepse ndan qartë:
- Fazën I si fazë të përgatitjes së të dhënave,
- Fazën II si fazë të ndërtimit dhe evaluimit të modeleve.

---

## Objektivi i parashikimit
Problemi ynë është formulua si **klasifikim multiklasor**. Targeti `target_quantile_class` ndan çdo ditë në tri klasa:
- `High`
- `Medium`
- `Low`

Kjo zgjedhje është e rëndësishme sepse:
- e kthen problemin në një detyrë klasifikimi më të kuptueshme,
- lehtëson krahasimin ndërmjet algoritmeve klasifikuese,
- dhe mundëson diskutim më intuitiv të rezultateve sesa një regresion i pastër numerik.

---

## Hapat metodologjikë të pipeline-it
Pipeline-i i Fazës II ndjek këtë rrjedhë:

1. Ngarkohet dataseti final nga Faza I.
2. Ndahen veçoritë (`X`) nga targeti (`y`).
3. Krijohet një `train_test_split` me `stratify=y`.
4. Kryhet preprocessing:
   - `StandardScaler` për kolonat numerike
   - `OneHotEncoder` për kolonat kategoriale
5. Kontrollohet balanca e klasave vetëm në training split.
6. Nëse është e nevojshme, aplikohet SMOTE ose ADASYN vetëm në train.
7. Trajnohen gjashtë algoritme supervised.
8. Testohen dy algoritme unsupervised për analizë krahasuese.
9. Gjenerohen grafikë, confusion matrices dhe përmbledhje për interpretim.

Ky organizim është i arsyeshëm sepse ndjek praktikën e mirë të Machine Learning:
- test set-i ruhet i paprekur deri në fund,
- preprocessing mësohet nga training set-i,
- balancimi kontrollohet vetëm në train,
- dhe krahasimi i modeleve bëhet mbi të njëjtin test set.

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
- dhe mban krahasimin e modeleve më të qëndrueshëm.

### Pse u përdor `StandardScaler`?
Standardizimi u aplikua për kolonat numerike sepse algoritme si:
- Logistic Regression,
- SVM,
- dhe MLP

janë të ndjeshme ndaj shkallës së ndryshme të veçorive. Pa standardizim, kolonat me vlera numerikisht më të mëdha do të dominonin procesin e optimizimit. Standardizimi e bën krahasimin mes veçorive më të drejtë dhe ndihmon konvergjencën e algoritmeve.

### Pse u përdor `OneHotEncoder`?
Kolonat kategoriale si `Data source` dhe `Data estimation method` nuk mund të futen drejtpërdrejt në shumicën e algoritmeve. One-hot encoding i transformon ato në forma numerike binare pa imponuar rend artificial ndërmjet kategorive.

### Pse nuk u përdor SMOTE në këtë run?
Balancimi u kontrollua vetëm pas ndarjes train/test, siç është praktikë korrekte. Në këtë run, train split rezultoi tashmë mjaft i balancuar, prandaj pipeline-i e kaloi këtë hap me:

- `Skipped (already balanced)`

Ky është një vendim i rëndësishëm metodologjik, sepse shmang krijimin e të dhënave sintetike kur ato nuk janë të nevojshme.

---

## Strategjia e evaluimit
Për secilin model supervised u përdor:
- `GridSearchCV`
- `3-fold cross-validation`
- metrika e optimizimit: `F1 (macro)`

### Pse `GridSearchCV`?
Grid search teston kombinime të ndryshme hiperparametrash dhe zgjedh konfigurimin më të mirë sipas një metrike të caktuar. Kjo është më e fortë sesa të zgjidhen parametrat me hamendje, sepse:
- e bën krahasimin më objektiv,
- e redukton bias-in e zgjedhjes manuale,
- dhe rrit besueshmërinë e rezultateve.

### Pse `F1 (macro)` dhe jo vetëm accuracy?
Accuracy është e dobishme, por nuk tregon gjithmonë balancën e performancës ndërmjet klasave. `F1 (macro)`:
- llogarit F1 për secilën klasë veçmas,
- pastaj i trajton të gjitha klasat në mënyrë të barabartë,
- dhe është më e përshtatshme për klasifikim multiklasor ku duam performancë të mirë në të gjitha klasat.

Kjo është arsyeja pse `F1 (macro)` u përdor si kriter kryesor i optimizimit dhe krahasimit.

---

## Algoritmet supervised të aplikuara

### 1. Logistic Regression
**Pse u zgjodh:** Logistic Regression u përfshi si model bazë linear. Ai ndihmon për të parë nëse klasat janë të dallueshme kryesisht me kufij linearë. Gjithashtu është model i interpretuar lehtë dhe shpesh shërben si pikë referimi.

**Çfarë u testua:**  
- `C = [0.01, 0.1, 1, 10]`

**Rezultati optimal:**  
- `C = 10`

**Rezultatet:**
- Accuracy: `0.9742`
- Precision (macro): `0.9742`
- Recall (macro): `0.9744`
- F1 (macro): `0.9741`

**Diskutimi:**  
Ky rezultat është shumë i lartë për një model linear dhe tregon që një pjesë e mirë e strukturës së klasave është e ndarshme edhe pa modele shumë komplekse. Nga classification report vërehet se klasa `Medium` është pak më e vështirë se `High` dhe `Low`, gjë që ka kuptim sepse klasa e mesme zakonisht kufizohet nga të dy anët dhe është më e paqartë.

---

### 2. Random Forest
**Pse u zgjodh:** Random Forest është algoritëm shumë i përshtatshëm për të dhëna tabulare, sidomos kur marrëdhëniet mes veçorive janë jolineare. Përveç kësaj, ai jep edhe rëndësinë e veçorive, çka e bën të vlefshëm jo vetëm për performancë, por edhe për interpretim.

**Çfarë u testua:**  
- `n_estimators = [100, 200]`
- `max_depth = [10, 20, None]`

**Rezultati optimal:**  
- `n_estimators = 100`
- `max_depth = 10`

**Rezultatet:**
- Accuracy: `0.9903`
- Precision (macro): `0.9903`
- Recall (macro): `0.9905`
- F1 (macro): `0.9904`

**Diskutimi:**  
Random Forest ishte ndër modelet më të mira. Kjo sugjeron se të dhënat përmbajnë marrëdhënie jolineare dhe kombinime veçorish që një model linear nuk i kap plotësisht. Fakti që modeli arrin performancë kaq të lartë pa pasur nevojë për thellësi shumë të madhe (`max_depth = 10`) tregon se struktura e të dhënave është e pasur, por jo kaotike.

Nga classification report shihet se të tria klasat trajtohen pothuajse në mënyrë perfekte, çka e bën këtë model shumë të besueshëm për këtë problem.

---

### 3. Gradient Boosting
**Pse u zgjodh:** Gradient Boosting është ndër algoritmet më të fuqishme për të dhëna tabulare, sepse ndërton modelin në mënyrë sekuenciale duke korrigjuar gabimet e pemëve paraprake. Kjo e bën shumë të fortë për kapjen e strukturave të ndërlikuara.

**Çfarë u testua:**  
- `n_estimators = [100, 200]`
- `learning_rate = [0.05, 0.1]`
- `max_depth = [3, 5]`

**Rezultati optimal:**  
- `n_estimators = 100`
- `learning_rate = 0.05`
- `max_depth = 3`

**Rezultatet:**
- Accuracy: `0.9903`
- Precision (macro): `0.9904`
- Recall (macro): `0.9905`
- F1 (macro): `0.9904`

**Diskutimi:**  
Gradient Boosting u barazua me Random Forest në performancë. Ky është një tregues shumë i fortë që struktura e të dhënave mbështet shumë mirë modelet me pemë. Parametrat optimalë tregojnë që performanca e lartë u arrit me pemë relativisht të cekëta dhe `learning_rate` të ulët, pra me qasje më konservatore dhe më stabile.

Kjo është e rëndësishme për diskutim, sepse tregon se modeli nuk ka nevojë për kompleksitet ekstrem për të performuar mirë.

---

### 4. SVM (Linear)
**Pse u zgjodh:** SVM Linear u përdor për të testuar nëse klasat mund të ndahen me kufij linearë në hapësirën e transformuar të veçorive. Është një krahasim i vlefshëm me Logistic Regression dhe SVM RBF.

**Çfarë u testua:**  
- `C = [0.1, 1, 10]`

**Rezultati optimal:**  
- `C = 10`

**Rezultatet:**
- Accuracy: `0.9710`
- Precision (macro): `0.9712`
- Recall (macro): `0.9713`
- F1 (macro): `0.9709`

**Diskutimi:**  
SVM Linear performoi shumë mirë, por pak më dobët se Logistic Regression dhe dukshëm më poshtë se modelet me pemë. Kjo sugjeron që ka ndarje lineare të konsiderueshme në të dhëna, por jo të mjaftueshme për të kapur plotësisht të gjitha kufijtë ndërmjet klasave.

Si te Logistic Regression, klasa `Medium` mbetet disi më sfiduese.

---

### 5. SVM (RBF)
**Pse u zgjodh:** SVM me kernel RBF u përfshi për të testuar nëse kufijtë e klasave janë më tepër jolinearë sesa linearë. Në teori, ky model duhet të performojë më mirë se varianti linear nëse marrëdhëniet janë fort jolineare.

**Çfarë u testua:**  
- `C = [0.1, 1, 10]`
- `gamma = [scale, auto]`

**Rezultati optimal:**  
- `C = 10`
- `gamma = auto`

**Rezultatet:**
- Accuracy: `0.9645`
- Precision (macro): `0.9646`
- Recall (macro): `0.9649`
- F1 (macro): `0.9645`

**Diskutimi:**  
Ky model rezultoi më i dobëti nga algoritmet supervised. Kjo është interesante, sepse teorikisht RBF është më fleksibil. Në praktikë, kjo nënkupton që fleksibiliteti shtesë nuk solli përfitim real dhe mund të ketë krijuar kufij më të ndërlikuar sesa duhej.

Pra, jo çdo model më kompleks jep domosdoshmërisht rezultat më të mirë. Kjo është pikërisht arsyeja pse krahasimi i modeleve është i domosdoshëm.

---

### 6. Neural Network (MLP)
**Pse u zgjodh:** MLP u përfshi sepse rrjetat neurale janë pjesë e rëndësishme e syllabusi-it dhe ofrojnë një qasje të ndryshme nga modelet klasike. Ky model teston nëse një strukturë më fleksibile neuronale sjell përmirësim në performancë.

**Çfarë u testua:**  
- `hidden_layer_sizes = [(64, 32), (128, 64)]`
- `alpha = [0.0001, 0.001]`

**Rezultati optimal:**  
- `hidden_layer_sizes = (128, 64)`
- `alpha = 0.0001`

**Rezultatet:**
- Accuracy: `0.9710`
- Precision (macro): `0.9716`
- Recall (macro): `0.9712`
- F1 (macro): `0.9712`

**Diskutimi:**  
MLP performoi mirë, por nuk e kaloi Random Forest dhe Gradient Boosting. Kjo është shumë tipike në të dhëna tabulare: rrjetat neurale jo gjithmonë dominojnë, sidomos kur dimensioni i problemit është i moderuar dhe veçoritë janë të strukturuara mirë. Pra, fakti që MLP nuk fitoi nuk është dobësi e analizës; përkundrazi, është gjetje e rëndësishme që tregon se modelet me pemë janë më të përshtatshme për këtë rast.

---

## Tabela krahasuese e modeleve supervised
| Model | Best Params | CV F1 (macro) | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | `{"C": 10}` | 0.9847 | 0.9742 | 0.9742 | 0.9744 | 0.9741 |
| Random Forest | `{"max_depth": 10, "n_estimators": 100}` | 0.9968 | 0.9903 | 0.9903 | 0.9905 | 0.9904 |
| Gradient Boosting | `{"learning_rate": 0.05, "max_depth": 3, "n_estimators": 100}` | 0.9984 | 0.9903 | 0.9904 | 0.9905 | 0.9904 |
| SVM (Linear) | `{"C": 10}` | 0.9863 | 0.9710 | 0.9712 | 0.9713 | 0.9709 |
| SVM (RBF) | `{"C": 10, "gamma": "auto"}` | 0.9599 | 0.9645 | 0.9646 | 0.9649 | 0.9645 |
| Neural Network (MLP) | `{"alpha": 0.0001, "hidden_layer_sizes": [128, 64]}` | 0.9766 | 0.9710 | 0.9716 | 0.9712 | 0.9712 |

### Diskutim i përgjithshëm
Nga kjo tabelë vërehen disa gjëra të rëndësishme:
- Të gjithë modelet supervised performuan shumë mirë, me F1 macro mbi `0.96`.
- Kjo tregon se veçoritë e krijuara në Fazën I janë vërtet informative.
- Modelet me pemë (`Random Forest`, `Gradient Boosting`) dominuan qartë.
- Diferencat mes modeleve lineare dhe jo-lineare tregojnë se problemi përmban strukturë jolineare, por jo aq të çrregullt sa të kërkojë domosdoshmërisht rrjeta neurale të thella.

---

## Analiza e klasifikimit sipas klasave
Nga `classification_reports.txt` vërehet se:
- klasat `High` dhe `Low` parashikohen pak më lehtë,
- ndërsa `Medium` është klasa më sfiduese pothuajse në të gjithë algoritmet.

Kjo është logjike, sepse `Medium` ndodhet midis dy klasave të tjera dhe shpesh përfaqëson raste kufitare. Kjo e bën më të vështirë për modelet të vendosin nëse një ditë është vërtet `Medium` apo më afër `High` ose `Low`.

Ky është një diskutim i rëndësishëm, sepse tregon se gabimet e modeleve nuk janë të rastësishme, por burojnë nga struktura e vet problemit.

---

## Analiza e algoritmeve unsupervised
Në këtë fazë janë aplikuar edhe dy algoritme unsupervised:
- K-Means
- Agglomerative Clustering

Rezultatet:
- K-Means silhouette score: `0.2110`
- Agglomerative silhouette score: `0.1958`

### Pse u përdorën?
Këto algoritme u përdorën jo për të zëvendësuar klasifikimin supervised, por për të testuar nëse në të dhëna ekzistojnë grupime natyrale të forta pa përdorur etiketat.

### Çfarë nënkuptojnë këto rezultate?
Silhouette score mat sa mirë është ndarë secili grup nga grupet e tjera. Vlera më afër `1` do të thotë grupim i fortë dhe i pastër. Vlera afër `0` tregon mbivendosje të konsiderueshme. Rezultatet rreth `0.20` janë relativisht të dobëta dhe tregojnë se:
- klasat `High`, `Medium`, `Low` nuk formojnë klasterë shumë të qartë në mënyrë natyrale,
- dhe se etiketat e targetit janë më të kuptueshme përmes qasjes supervised sesa clustering.

Ky është një rezultat i rëndësishëm teorik, sepse justifikon pse problemi ynë duhet trajtuar si problem klasifikimi i mbikëqyrur.

---

## Diskutimi i grafikëve të gjeneruar

### 1. `algorithm_comparison.png`
Ky grafik jep krahasimin vizual të modeleve supervised në metrikat kryesore. Është i rëndësishëm sepse bën menjëherë të dukshme diferencat mes modeleve dhe konfirmon epërsinë e Random Forest dhe Gradient Boosting.

### 2. `feature_importance_rf.png`
Ky grafik paraqet rëndësinë relative të veçorive në Random Forest. Kjo ndihmon në interpretimin e modelit dhe tregon cilat karakteristika kontribuojnë më shumë në vendimmarrje. Për raport akademik, ky grafik është me vlerë sepse lidh modelimin me kuptimin e të dhënave.

### 3. `learning_curves.png`
Ky grafik shërben për të diskutuar overfitting dhe underfitting. Nëse training score është shumë më i lartë se validation score, kemi shenjë mbipërshtatjeje. Në run-in aktual, diferenca mbetet e kontrolluar, çka sugjeron generalizim të mirë.

### 4. `regularization_effect.png`
Ky grafik tregon si ndryshon performanca e Logistic Regression me vlera të ndryshme të `C`. Ai ilustron në praktikë idenë e kompromisit bias-variance dhe është i rëndësishëm për të justifikuar përdorimin e regularizimit.

### 5. `mlp_loss_curve.png` dhe `mlp_validation_curve.png`
Këto dy grafikë demonstrojnë si neural network mëson gjatë epokave. Rënia e loss-it tregon konvergjencë, ndërsa validation curve tregon stabilitetin e modelit gjatë trajnimit.

### 6. `elbow_and_silhouette.png`
Ky grafik është i rëndësishëm për pjesën unsupervised, sepse tregon se cilat vlera të `k` u testuan dhe sa i arsyeshëm është grupimi me `k=3`.

### 7. `pca_clusters_comparison.png`
Ky grafik ndihmon në krahasimin vizual të clustering me etiketat reale. Ai është shumë i dobishëm në diskutim, sepse e bën të qartë pse unsupervised learning nuk i kap mirë klasat reale.

### 8. `confusion_matrix_*.png`
Matrica e konfuzionit për secilin model është shumë e rëndësishme sepse nuk tregon vetëm sa gabime ka modeli, por edhe **ku** gabon. Në këtë projekt, gabimet priren të ndodhin më shumë rreth klasës `Medium`, gjë që përputhet me interpretimin tonë teorik.

---

## Përfundimi i Fazës II
Nga të gjitha eksperimentet e zhvilluara, mund të nxirren këto përfundime:

1. Dataseti i përpunuar në Fazën I është mjaft cilësor dhe shumë informativ për detyrën e klasifikimit.
2. Të gjitha modelet supervised arritën performancë të lartë, çka tregon se problemi është i modelueshëm mirë.
3. `Random Forest` dhe `Gradient Boosting` arritën performancën më të mirë me:
   - `Accuracy = 0.9903`
   - `F1 (macro) = 0.9904`
4. Modelet lineare si Logistic Regression dhe SVM Linear performuan gjithashtu shumë mirë, duke treguar se një pjesë e strukturës së problemit është lineare.
5. SVM RBF dhe MLP nuk sollën përmirësim mbi modelet me pemë, çka sugjeron se për këtë problem, modelet e bazuara në pemë janë më të përshtatshme.
6. Algoritmet unsupervised dhanë silhouette score të ulët, gjë që tregon se klasat nuk formojnë klasterë të fortë natyralë pa etiketa.

Pra, Faza II jo vetëm që tregoi se mund të ndërtohen modele shumë të sakta për këtë problem, por gjithashtu tregoi qartë **pse** qasja supervised është më e përshtatshme dhe **si** duhet interpretuar performanca e modeleve në raport me natyrën e të dhënave.

---

## Output-et kryesore

### Skedarët tabelorë
- `output/model_results.csv`
- `output/clustering_results.csv`
- `output/classification_reports.txt`
- `output/results_summary.md`
- `output/phase2_train_raw_features.csv`
- `output/phase2_train_raw_target.csv`
- `output/phase2_test_raw_features.csv`
- `output/phase2_test_target.csv`
- `output/phase2_train_processed_features.csv`
- `output/phase2_test_processed_features.csv`
- `output/train_balanced_features.csv`
- `output/train_balanced_target.csv`

### Grafikët
- `output/algorithm_comparison.png`
- `output/feature_importance_rf.png`
- `output/learning_curves.png`
- `output/regularization_effect.png`
- `output/mlp_loss_curve.png`
- `output/mlp_validation_curve.png`
- `output/elbow_and_silhouette.png`
- `output/pca_clusters_comparison.png`
- `output/confusion_matrix_*.png`

---

## Ekzekutimi
```bash
python phase2_pipeline.py
```
