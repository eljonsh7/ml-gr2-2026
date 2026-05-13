# Faza III - Ritrajnimi: Optimizimi dhe Fine-Tuning i Modeleve

## Tabela e Përmbajtjes
1. [Përmbledhje](#1-përmbledhje)
2. [Çfarë Ndryshoi nga Faza II](#2-çfarë-ndryshoi-nga-faza-ii)
3. [Arkitektura e Pipeline-it](#3-arkitektura-e-pipeline-it)
4. [Hapi 0 - Ngarkimi dhe Ndarja e të Dhënave](#4-hapi-0--ngarkimi-dhe-ndarja-e-të-dhënave)
5. [Hapi 1 - Paraprocesimi](#5-hapi-1--paraprocesimi)
6. [Hapi 2 - Kontrolli i Balancës së Klasave](#6-hapi-2--kontrolli-i-balancës-së-klasave)
7. [Hapi 3 - Zgjedhja e Veçorive](#7-hapi-3--zgjedhja-e-veçorive)
8. [Hapi 4 - Strategjia e Kërkimit të Hiperparametrave](#8-hapi-4--strategjia-e-kërkimit-të-hiperparametrave)
9. [Algoritmet - Trajtim i Plotë Matematikor](#9-algoritmet--trajtim-i-plotë-matematikor)
   - [9.1 Regresioni Logjistik](#91-regresioni-logjistik)
   - [9.2 Isolation Forest](#92-pylli-i-rastit)
   - [9.3 Gradient Boosting](#93-gradient-boosting)
   - [9.4 SVM Linear](#94-makina-me-vektora-mbështetës--bërthama-lineare)
   - [9.5 Rrjeti Nervor (MLP)](#95-rrjeti-nervor--perceptroni-shumështresor)
10. [Metrikat e Vlerësimit - Formula të Plota](#10-metrikat-e-vlerësimit--formula-të-plota)
11. [Rëndësia Statistikore - Testi Wilcoxon Signed-Rank](#11-rëndësia-statistikore--testi-wilcoxon-signed-rank)
12. [Mjetet Shtesë të Analizës](#12-mjetet-shtesë-të-analizës)
    - [12.1 Testi McNemar - Faza II vs Faza III](#121-testi-mcnemar--faza-ii-vs-faza-iii)
    - [12.2 SHAP - Interpretueshmëria e Modelit](#122-shap--interpretueshmëria-e-modelit)
    - [12.3 Kurba e Validimit - Ndjeshmëria ndaj Hiperparametrave](#123-kurba-e-validimit--ndjeshmëria-ndaj-hiperparametrave)
13. [Rezultatet](#13-rezultatet)
14. [Krahasimi Faza II vs Faza III](#14-krahasimi-faza-ii-vs-faza-iii)
15. [Skedarët e Prodhuar](#15-skedarët-e-prodhuar)
16. [Konkluzionet dhe Impakti i Projektit](#16-konkluzionet-dhe-impakti-i-projektit)

---

## 1. Përmbledhje

**Qëllimi:** Të merren pesë algoritmet supervised me performancën më të lartë nga Faza II, të zgjerohet hapësira e kërkimit të parametrave dhe të identifikohet një model i vetëm statistikisht superior për parashikimin e klasës ditore të intensitetit të karbonit në rrjetin elektrik të Kosovës.

**Variabla e synuar:** `target_quantile_class` - tre klasa: `High`, `Medium`, `Low`

**Inputi:** `feature_engineered_dataset.csv` nga Faza I (1,550 rreshta × 20 kolona)

**Vendimi i hequr nga Faza II:** SVM (RBF) - CV F1 = 0.9599 në Fazën II, më i ulëti ndër të gjashtë modelet. Fleksibiliteti i tij shtesë nuk solli asnjë fitim ndaj bërthamës lineare, duke konfirmuar se struktura dominante në këtë dataset nuk kapet mirë nga një funksion bazë radial.

---

## 2. Çfarë Ndryshoi nga Faza II

| Aspekti | Faza II | Faza III                                                                   |
|---|---|----------------------------------------------------------------------------|
| Modelet | 6 (me SVM RBF) | 5 (SVM RBF hiqet)                                                          |
| Metoda e kërkimit | `GridSearchCV` | `RandomizedSearchCV`                                                       |
| Fold-et CV | 3 | 5                                                                          |
| Hapësirat e parametrave | Të ngushta | Të gjera (3–5× më shumë vlera)                                             |
| Zgjedhja e veçorive | Asnjë | Pragu i nevojës RF (25 → 9)                                            |
| Metrikat | Accuracy, Precision, Recall, F1 | + ROC-AUC (macro, OvR)                                                     |
| Testet statistikore | Asnjë | Wilcoxon (krahasim fold-to-fold) + McNemar (krahasim gabimesh në test set) |
| Interpretueshmëria e modelit | Asnjë | SHAP - kontributi për-veçori i modelit më të mirë                          |
| Diagnostikimi i hiperparametrave | Asnjë | Kurba e validimit (sweep i learning_rate)                                  |
| Raporti final | RF dhe GB barabar | Fitues i vetëm me provë statistikore                                       |

---

## 3. Arkitektura e Pipeline-it

```
Dataset i Fazës I (1,550 × 20)
        │
        ▼
  train_test_split (stratifikuar, 80/20, RANDOM_STATE=42)
        │
        ├── X_train_raw (1,240 × 19) ── StandardScaler + OneHotEncoder ──▶ X_train_proc (1,240 × 25)
        │                                                                            │
        │                                                                balance_training_split
        │                                                                  (tashmë balancuar → kalohet)
        │                                                                            │
        │                                                                Zgjedhja e Veçorive (RF)
        │                                                                   25 → 9 veçori
        │                                                                            │
        │                                                        ┌───────────────────┤
        │                                                        │  RandomizedSearchCV│
        │                                                        │  5-fold Stratified │
        │                                                        │  KFold, F1 macro   │
        │                                                        ├────────────────────┤
        │                                                        │ Regresion Logjistik│
        │                                                        │ Isolation Forest     │
        │                                                        │ Gradient Boosting  │
        │                                                        │ SVM (Linear)       │
        │                                                        │ MLP                │
        │                                                        └────────┬───────────┘
        │                                                                 │
        └── X_test_raw (310 × 19) ──── transform ──── X_test_sel ────────▶ Vlerëso
                                                                           │
                              ┌────────────────────────────────────────────┤
                              │  Metrikat Kryesore                         │
                              │    Accuracy, Precision, Recall, F1, AUC   │
                              │    Matrica Konfuzioni (5 modele)           │
                              │    Kurba ROC-AUC (5 modele, macro OvR)    │
                              │    Kurba Kalibrimi                         │
                              │    Kurbat e të Mësuarit (sklearn)          │
                              ├────────────────────────────────────────────┤
                              │  Testet Statistikore                       │
                              │    Wilcoxon: krahasim fold-to-fold CV      │
                              │    McNemar: Ph2 GB vs Ph3 Fituesi (test)   │
                              ├────────────────────────────────────────────┤
                              │  Interpretueshmëria dhe Diagnostika        │
                              │    SHAP: kontributet për-veçori (fituesi)  │
                              │    Kurba Validimi: sweep i learning_rate   │
                              └────────────────────────────────────────────┘
```

---

## 4. Hapi 0 - Ngarkimi dhe Ndarja e të Dhënave

### Ndarja e Stratifikuar Train/Test

Dataseti ndahet 80% trajnim / 20% testim duke përdorur `stratify=y`:

```
RANDOM_STATE = 42   (garanton riprodhueshmëri dhe ndarje identike me Fazën II)
test_size    = 0.2

Trajnim: 1,240 rreshta
Testim:    310 rreshta
```

**Pse stratifikim?**
Pa stratifikim, një ndarje rastësore mund të vendosë rastësisht më shumë mostra `High` në test set dhe më pak në trajnim. Stratifikimi siguron që proporcionet e klasave në të dy set-et të përputhen me datasetin origjinal. Formalisht, nëse klasa $k$ ka proporcion $p_k$ në datasetin e plotë, stratifikimi garanton:

$$\frac{n_k^{\text{trajnim}}}{n^{\text{trajnim}}} \approx \frac{n_k^{\text{testim}}}{n^{\text{testim}}} \approx p_k$$

**Shpërndarja e klasave në test set (aktuale):**
| Klasa | Numri | Përqindja |
|---|---|---|
| Medium | 106 | 34.2% |
| High | 105 | 33.9% |
| Low | 99 | 31.9% |

---

## 5. Hapi 1 - Paraprocesimi

Paraprocesimi aplikohet **vetëm mbi set-in e trajnimit** dhe më pas aplikohet mbi test set-in. Aplikimi mbi datasetin e plotë para ndarjes është rrjedhje e të dhënave (data leakage) - modeli do të kishte njohuri indirekte mbi statistikat e test set-it.

### 5.1 StandardScaler (Kolonat Numerike)

Çdo veçori numerike $x_j$ transformohet në mesatare zero dhe variancë njësi:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

Ku:
- $\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}$ - mesatarja e veçorisë $j$ mbi set-in e trajnimit
- $\sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2}$ - devijimi standard

**Pse?** Algoritmet si Regresioni Logjistik, SVM dhe MLP optimizojnë një funksion humbje me metoda të bazuara në gradient. Një veçori me vlera në [0, 10,000] do të prodhonte gradienta 1,000× më të mëdha se një veçori me vlera në [0, 10], duke bërë optimizuesin të lëvizë në mënyrë disproporcionale. Standardizimi eliminon këtë varësi nga shkalla.

Modelet e bazuara në pemë (Isolation Forest, Gradient Boosting) janë invariante nga shkalla - ato kujdesen vetëm për renditjen e veçorive, jo për madhësinë. StandardScaler nuk i dëmton as ato.

### 5.2 OneHotEncoder (Kolonat Kategorike)

Çdo veçori kategorike me $K$ kategori unike zëvendësohet me $K$ kolona binare:

$$x_j \in \{c_1, c_2, \ldots, c_K\} \longrightarrow [0, 0, \ldots, 1, \ldots, 0] \in \{0,1\}^K$$

**Parametrat e përdorur:**
- `handle_unknown="ignore"` - nëse test set-i përmban një kategori që nuk është parë gjatë trajnimit, të gjitha kolonat e saj tregues vendosen në 0 në vend që të ngrihet një gabim
- `sparse_output=False` - kthen një array të dendur NumPy për përputhshmëri me të gjithë estimatorët vijues

**Pse jo kodimi ordinal?** Kodimi ordinal (p.sh., `Low=0, Medium=1, High=2`) nënkupton një renditje që nuk ekziston për variablat kategorike arbitrare. Kodimi one-hot nuk bën supozime të tilla.

**Rezultati pas paraprocesimit:** 25 veçori (nga 19 të papërpunuara; kolona shtesë nga zgjerimi one-hot i variablave kategorike).

---

## 6. Hapi 2 - Kontrolli i Balancës së Klasave

Vlerësohet **vetëm mbi ndarjen e trajnimit** (kontrolli i test set-it do të ishte rrjedhje - modelet vlerësohen mbi test set-in siç është).

**Rregulli i pragut:**
$$\text{Nëse } \min_k \left(\frac{n_k}{n}\right) \geq 0.20 \Rightarrow \text{i balancuar, kalohet risamplimet}$$

**Rezultati në këtë ekzekutim:** Ndarja e trajnimit ishte natyrshëm e balancuar → `"Skipped (already balanced)"`

Nëse është i pabalancuar, pipeline-i aplikon:

### SMOTE - Teknika e shtimit të Minoritetit Sintetik

Për çdo mostër të minoritetit $\mathbf{x}_i$, zgjidhet njëri nga $k$ fqinjët më të afërt $\mathbf{x}_{nn}$ rastësisht dhe krijohet:

$$\mathbf{x}_{\text{e re}} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_{nn} - \mathbf{x}_i), \quad \lambda \sim \text{Uniform}(0, 1)$$

Kjo gjeneron mostra sintetike **përgjatë segmentit të vijës** midis pikave reale të klasës së minoritetit.

**Kushti:** Përdoret kur $\min(n_k) \geq 6$ (fqinjë të mjaftueshëm për interpolim të qëndrueshëm).

### ADASYN - Mostrat Sintetike Adaptive

ADASYN përmirëson SMOTE duke gjeneruar **më shumë mostra sintetike pranë kufirit të vendimit** - domethënë, rreth pikave të klasës minoritare të rrethuara nga pika të klasës shumicë dhe prandaj më të vështira për t'u klasifikuar saktë.

Për çdo mostër të minoritetit $\mathbf{x}_i$, llogaritet:

$$r_i = \frac{\Delta_i}{k}, \quad \Delta_i = \text{numri i mostrave të klasës shumicë midis } k \text{ fqinjëve më të afërt të } \mathbf{x}_i$$

Pastaj normalizohet: $\hat{r}_i = r_i / \sum_j r_j$

Numri i mostrave sintetike të gjeneruara për $\mathbf{x}_i$ është $G \cdot \hat{r}_i$, ku $G$ është numri total i mostrave sintetike të nevojshme. Mostrat pranë kufirit (me $r_i$ të lartë) marrin më shumë fqinjë sintetikë.

**Kushti:** Përdoret kur $\min(n_k) < 6$ (SMOTE ka nevojë për të paktën 6 fqinjë).

---

## 7. Hapi 3 - Zgjedhja e Veçorive

Një Pyll i Rastit i parapërgatitur (100 pemë) trajnohet mbi set-in e trajnimit të balancuar për të llogaritur **rëndësitë e veçorive të bazuara në Gini**.

### Formula e Nevojës së Veçorisë (Ulja Mesatare e Papastërtisë)

Për një pemë të vetme vendimi, rëndësia e veçorisë $j$ është:

$$I(j) = \sum_{t \in \text{nyjet ku përdoret } j} \frac{n_t}{n} \cdot \Delta \text{Gini}(t)$$

Ku:
- $n_t$ = numri i mostrave të trajnimit që arrijnë nyjen $t$
- $n$ = numri total i mostrave trajnuese
- $\Delta \text{Gini}(t) = \text{Gini}(t) - \frac{n_L}{n_t}\text{Gini}(t_L) - \frac{n_R}{n_t}\text{Gini}(t_R)$ - ulja e papastërtisë në ndarjen $t$
- $\text{Gini}(t) = 1 - \sum_{k} p_k^2$ - papastërtia Gini në nyjen $t$

Për një Pyll Rasti, rëndësia mesatarizohet mbi të gjitha pemët dhe normalizohet në shumë 1.

### Pragu

$$\text{prag} = \bar{I} \times 0.05, \quad \bar{I} = \frac{1}{p} \sum_{j=1}^{p} I(j)$$

Faktori 0.05 heq vetëm veçoritë me nevojë nën 5% të mesatares - ky është një kufi konservativ që eliminon veçoritë gati-zero duke mbajtur shumicën dërrmuese.

**Rezultati:** 25 veçori → **9 veçori të mbajtura**

Kjo redukton zhurmën nga veçoritë e parëndësishme, shpejton trajnimin dhe mund të përmirësojë gjeneralizimin. I njëjti maskë aplikohet në mënyrë identike mbi test set-in (pa ri-aplikim mbi të dhënat e testit).

---

## 8. Hapi 4 - Strategjia e Kërkimit të Hiperparametrave

### RandomizedSearchCV

Në vend të testimit lodhës të çdo kombinimi (GridSearchCV), RandomizedSearchCV kampionon $n\_iter$ kombinime në mënyrë uniforme rastësisht nga shpërndarjet e parametrave:

$$\theta^* = \arg\max_{\theta \in S} \mathbb{E}[\text{Rezultati CV}(\theta)], \quad S \subset \Theta, \; |S| = n\_iter$$

**Pse RandomizedSearchCV dhe jo GridSearchCV?**
- GridSearchCV me grida të gjera do të kërkonte mijëra trajnime modeli (p.sh., Gradient Boosting ka $5 \times 5 \times 4 \times 4 \times 3 = 1200$ kombinime - me 5 fold-e secila, kjo janë 6,000 trajnime për model)
- RandomizedSearchCV kampionon 30 kombinime (150 trajnime për model) dhe shploron ende të gjithë gamën
- Empirikisht, kërkimi rastësor gjen hiperparametra po aq të mirë sa kërkimi me grid në një fraksion të kohës *(Bergstra & Bengio, 2012)*

**n_iter kufizohet për çdo model:**
```
Regresion Logjistik : min(30, 11)  = 11   ← lodhës (vetëm 11 kombinime)
Isolation Forest    : min(30, 540) = 30   ← nënbashkësi rastësore
Gradient Boosting   : min(30,1200) = 30   ← nënbashkësi rastësore
SVM (Linear)        : min(30,   9) =  9   ← lodhës (vetëm 9 kombinime)
MLP                 : min(30,  60) = 30   ← nënbashkësi rastësore
```

### Validimi i Kryqëzuar StratifiedKFold (5 fold-e)

Set-i i trajnimit ndahet në 5 fold-e jo-të-mbivendosura, të stratifikuara:

```
Fold 1: [████░░░░░░░░░░░░░░░░]  validim / trajnim mbi 4 të tjerat
Fold 2: [░░░░████░░░░░░░░░░░░]
Fold 3: [░░░░░░░░████░░░░░░░░]
Fold 4: [░░░░░░░░░░░░████░░░░]
Fold 5: [░░░░░░░░░░░░░░░░████]
```

Rezultati CV për konfigurimin e parametrave $\theta$:

$$\widehat{\text{CV}}(\theta) = \frac{1}{K} \sum_{k=1}^{K} \text{F1}_{\text{macro}}(M_\theta^{(-k)}, D_k)$$

Ku $M_\theta^{(-k)}$ është modeli i trajnuar mbi të gjitha fold-et përveç $k$, i vlerësuar mbi fold-in $k$.

**Pse 5-fold dhe jo 3-fold (Faza II)?**
Më shumë fold-e → më shumë të dhëna trajnuese për fold → bias më i ulët në vlerësimin e rezultatit, dhe mesatarizon mbi 5 set-e të mbajtura redukton variancën. 5-fold është standardi në vlerësimin akademik të ML.

**Metrika e vlerësimit:** `f1_macro` - F1 makro-mesatare trajton të gjitha klasat në mënyrë të barabartë pavarësisht mbështetjes, gjë që është e përshtatshme për një problem me 3 klasa ku asnjë klasë e vetme nuk duhet të dominojë optimizimin.

---

## 9. Algoritmet - Trajtim i Plotë Matematikor

### 9.1 Regresioni Logjistik

**Lloji:** Klasifikues linear probabilistik (model linear i përgjithësuar)

**Hiperparametrat më të mirë të gjetur:** `C = 100`

#### Formulimi Multinomial (Softmax)

Për $K = 3$ klasa, probabiliteti i klasës $k$ për input-in $\mathbf{x} \in \mathbb{R}^p$ është:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^\top \mathbf{x} + b_j}}$$

Kjo është **funksioni softmax** - prodhon një shpërndarje të duhur probabiliteti mbi $K$ klasa (të gjitha vlerat pozitive, shuma 1).

#### Funksioni i Humbjes - Entropia e Kryqëzuar Multinomiale + Regularizimi L2

Modeli trajnohet duke minimizuar:

$$\mathcal{L}(\mathbf{W}) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbf{1}[y_i = k] \log P(y_i = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=1}^{K} \|\mathbf{w}_k\|_2^2$$

Ku:
- Termi i parë: **humbja nga entropia e kryqëzuar** - ndëshkon probabilitetin e ulët të parashikuar për klasën e saktë
- Termi i dytë: **regularizimi L2** - tkurr peshat drejt zeros për të parandaluar mbi-përshtatjen
- $C$ = **forca inverse e regularizimit**: $C$ i madh → regularizim i dobët → model më fleksibël; $C$ i vogël → regularizim i fortë → model më i thjeshtë

#### Efekti i Regularizimit

$$C \to 0 \Rightarrow \mathbf{W} \to \mathbf{0} \quad \text{(nën-përshtatje)}$$
$$C \to \infty \Rightarrow \text{pa penalizim, MLE i pastër} \quad \text{(rrezik mbi-përshtatjeje)}$$

**C = 100 më i miri** (regularizim i dobët) tregon se dataseti ka zhurmë të ulët dhe modeli përfiton nga përshtatja e ngushtë me shpërndarjen e trajnimit.

#### Rregulli i Vendimit

$$\hat{y} = \arg\max_{k} \; P(y = k \mid \mathbf{x})$$

#### Rezultatet e Fazës III

| Metrika | Vlera |
|---|---|
| CV F1 (macro) | 0.9872 |
| Accuracy | 0.9806 |
| F1 (macro) | 0.9806 |
| ROC-AUC (macro) | 0.9993 |

**Sipas klasës:**
| Klasa | Precision | Recall | F1 | Mbështetja |
|---|---|---|---|---|
| High | 1.00 | 0.99 | 1.00 | 105 |
| Low | 0.97 | 0.98 | 0.97 | 99 |
| Medium | 0.97 | 0.97 | 0.97 | 106 |

---

### 9.2 Isolation Forest

**Lloji:** Ansambël pemësh vendimi duke përdorur Agregimin Bootstrap (Bagging)

**Hiperparametrat më të mirë të gjetur:** `n_estimators=100, max_depth=8, min_samples_split=2, min_samples_leaf=4, max_features="log2"`

#### Pema e Vendimit - Kriteri i Ndarjes (Papastërtia Gini)

Në çdo nyje $t$, pema zgjedh veçorinë $j^*$ dhe pragun $\tau^*$ që maksimizojnë uljen e papastërtisë:

$$j^*, \tau^* = \arg\max_{j, \tau} \; \Delta\text{Gini}(t, j, \tau)$$

$$\Delta\text{Gini}(t, j, \tau) = \text{Gini}(t) - \frac{n_L}{n_t}\text{Gini}(t_L) - \frac{n_R}{n_t}\text{Gini}(t_R)$$

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} \hat{p}_{tk}^2, \quad \hat{p}_{tk} = \frac{\text{mostrat e klasës } k \text{ në nyjen } t}{n_t}$$

Një nyje e pastër ($\hat{p}_{tk} = 1$ për një $k$) ka $\text{Gini} = 0$ (minimum). Një nyje plotësisht e përzier ka $\text{Gini} = 1 - \frac{1}{K}$ (maksimum).

#### Bagging (Agregimi Bootstrap)

Për çdo pemë $b = 1, \ldots, B$:
1. Nxirret një mostër bootstrap $D_b$ me madhësi $n$ me zëvendësim nga set-i i trajnimit
2. Në çdo ndarje, konsiderohet vetëm një nënbashkësi rastësore e $m = \lfloor\log_2(p)\rfloor$ veçorish (`max_features="log2"`)
3. Pema rritet deri në thellësinë maksimale ose derisa të plotësohen kushtet e pastërtisë së gjethes

**Pse nënbashkësi rastësore veçorish?** Nëse një veçori është shumë parashikuese, të gjitha pemët në një ansambël bagging standard do ta përdornin atë në rrënjë, duke i bërë pemët shumë të korreluara. Nënbashkësitë rastësore de-korrelojnë pemët, duke reduktuar variancën e ansamblit pa rritur biast.

#### Parashikimi i Ansamblit (Votimi me Shumicë)

$$\hat{y} = \text{mode}\left(\hat{y}_1(\mathbf{x}), \hat{y}_2(\mathbf{x}), \ldots, \hat{y}_B(\mathbf{x})\right)$$

#### Dekompozimi Bias-Variancë

$$\text{Gabimi} = \text{Bias}^2 + \text{Varianca} + \text{Zhurmë e Pareduktuesme}$$

Pemët e thella individuale kanë bias të ulët por variancë të lartë (mbi-përshtaten me mostrën e tyre bootstrap). Mesatarizon $B$ pemë të de-korreluara redukton variancën me afërsisht $\frac{1}{B}$ duke mbajtur biasit konstant.

#### Kuptimet e Hiperparametrave

| Parametri | Vlera | Efekti                                                                        |
|---|---|-------------------------------------------------------------------------------|
| `n_estimators` | 100 | Numri i pemëve - më shumë pemë → variancë më e ulët por kthesa zbriste        |
| `max_depth` | 8 | Thellësia maksimale e pemës - parandalon mbi-përshtatjen e pemëve individuale |
| `min_samples_leaf` | 4 | Gjethja duhet të ketë ≥ 4 mostra - regularizim i pemës, redukton zhurmën      |
| `min_samples_split` | 2 | Nyja duhet të ketë ≥ 2 mostra për t'u ndarë                                   |
| `max_features` | log2 | $m = \lfloor\log_2(9)\rfloor = 3$ veçori konsiderohen për çdo ndarje          |

#### Rezultatet e Fazës III

| Metrika | Vlera |
|---|---|
| CV F1 (macro) | 0.9984 |
| Accuracy | 0.9903 |
| F1 (macro) | 0.9904 |
| ROC-AUC (macro) | 0.9999 |

**Sipas klasës:**
| Klasa | Precision | Recall | F1 | Mbështetja |
|---|---|---|---|---|
| High | 1.00 | 0.98 | 0.99 | 105 |
| Low | 0.99 | 1.00 | 0.99 | 99 |
| Medium | 0.98 | 0.99 | 0.99 | 106 |

---

### 9.3 Gradient Boosting

**Lloji:** Ansambël additiv i pemëve të cekëta vendimi të ndërtuara sekuencialisht duke përdorur zbritjen e gradientit në hapësirën e funksionit

**Hiperparametrat më të mirë të gjetur:** `n_estimators=500, learning_rate=0.2, max_depth=5, subsample=0.9, min_samples_split=5`

#### Modeli Additiv

Parashikimi është shuma e $M$ nxënësve të dobët (pemëve):

$$F_M(\mathbf{x}) = F_0(\mathbf{x}) + \sum_{m=1}^{M} \eta \cdot h_m(\mathbf{x})$$

Ku:
- $F_0(\mathbf{x})$ = parashikimi fillestar (p.sh., log-gjasët e klasës më të shpeshtë)
- $h_m(\mathbf{x})$ = pema e $m$-të, e trajnuar për të parashikuar **gradientin negativ** (pseudo-mbetjet) të humbjes ndaj modelit aktual
- $\eta$ = `learning_rate` (parametri i tkurrjes)

#### Zbritja e Gradientit në Hapësirën e Funksionit

Në çdo raund boosting-u $m$, llogariten pseudo-mbetjet:

$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{m-1}}$$

Për humbjen nga entropia e kryqëzuar multinomiale mbi $K$ klasa:

$$\mathcal{L} = -\sum_{i=1}^{n} \sum_{k=1}^{K} \mathbf{1}[y_i = k] \log p_{ik}$$

Pseudo-mbetja për klasën $k$ në mostrën $i$ është:

$$r_{imk} = \mathbf{1}[y_i = k] - p_{ik,m-1}$$

Kjo është thjesht diferenca midis etiketës reale one-hot dhe parashikimit aktual të probabilitetit - modeli mëson të reduktojë këtë mbetje në çdo hap.

#### Tkurrja (learning_rate)

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x}), \quad \eta = 0.2$$

$\eta$ i vogël → hapa të shumë të vegjël (nevojiten më shumë pemë, regularizim më i mirë).
$\eta$ i madh → hapa më pak, më të mëdha (më i shpejtë por mund të tejkalojë).

#### Gradient Boosting Stokastik (subsample)

Me `subsample=0.9`, çdo pemë $h_m$ trajnohet mbi 90% mostrë rastësore të të dhënave trajnuese (pa zëvendësim), të nxjerra sërishmi çdo raund. Kjo fut rastësi që:
- Redukton korrelacionin midis pemëve konsekutive
- Vepron si regularizim implicit
- Shpesh përmirëson gjeneralizimin (Friedman, 1999)

#### Dallimi nga Isolation Forest

| | Isolation Forest | Gradient Boosting |
|---|---|---|
| Ndërtimi i pemës | **Paralel** (i pavarur) | **Sekuencial** (secila korrigjon të mëparshmen) |
| Synimi | Etiketa origjinale | Pseudo-mbetjet (gradienti i humbjes) |
| Rastësia | Bootstrap + nënbashkësi veçorish | Nën-samplimet (stokastike) |
| Regularizimi | Thellësia, madhësia e gjethes | Shkalla e mësimit, thellësia, nën-samplimet |
| Pemë tipike | Të thella | **Të cekëta** (max_depth=5) |

#### Kuptimet e Hiperparametrave

| Parametri | Vlera | Efekti                                                                       |
|---|---|------------------------------------------------------------------------------|
| `n_estimators` | 500 | Raundet e boosting-ut - më shumë raunde përshtatin mbetjet më saktë          |
| `learning_rate` | 0.2 | Tkurrja - shkallëzon kontributin e secilës pemë                              |
| `max_depth` | 5 | Pemët e cekëta janë nxënës të dobët - thellësia 5 lejon ndërveprime moderate |
| `subsample` | 0.9 | 90% e të dhënave trajnuese për pemë - regularizim stokastik                  |
| `min_samples_split` | 5 | Nyja duhet të ketë ≥5 mostra për t'u ndarë                                   |

#### Rezultatet e Fazës III (Modeli Fitues)

| Metrika | Vlera |
|---|---|
| CV F1 (macro) | **0.9992** ← më i larti |
| Accuracy | 0.9903 |
| F1 (macro) | 0.9904 |
| ROC-AUC (macro) | **0.9999** |

**Sipas klasës:**
| Klasa | Precision | Recall | F1 | Mbështetja |
|---|---|---|---|---|
| High | 1.00 | 0.98 | 0.99 | 105 |
| Low | 0.99 | 1.00 | 0.99 | 99 |
| Medium | 0.98 | 0.99 | 0.99 | 106 |

---

### 9.4 Makina me Vektora Mbështetës - Bërthama Lineare

**Lloji:** Klasifikues linear me margin maksimal

**Hiperparametrat më të mirë të gjetur:** `C = 50`

**Shënim:** `probability=True` aktivizon shkallëzimin Platt kështu që `predict_proba` është i disponueshëm për llogaritjen e ROC-AUC.

#### Formulimi Binar SVM (zgjeruar te shumë-klasësh nëpërmjet One-vs-Rest)

Për një problem binar me etiketa $y \in \{-1, +1\}$, SVM gjen hiperplanin $\mathbf{w}^\top \mathbf{x} + b = 0$ që maksimizon marginin:

$$\max_{\mathbf{w}, b} \frac{2}{\|\mathbf{w}\|} \quad \text{kushtëzuar nga} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \; \forall i$$

Ekuivalent (forma primare me variabla slack):

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$
$$\text{kushtëzuar nga} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

Ku:
- $\frac{1}{\|\mathbf{w}\|}$ = **gjerësia e marginit** - maksimizimi i saj minimizon $\|\mathbf{w}\|^2$
- $\xi_i \geq 0$ = **variablat slack** - lejojnë keq-klasifikime (margin i butë)
- $C$ = **parametri i regularizimit**: $C$ i madh → ndëshko keq-klasifikimet rëndë → margin i vogël; $C$ i vogël → lejo më shumë slack → margin më i madh

#### Interpretimi i Humbjes Hinge

Objektivi SVM është ekuivalent me:

$$\min_{\mathbf{w}} \frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac{1}{n}\sum_{i=1}^{n} \max(0, 1 - y_i \mathbf{w}^\top \mathbf{x}_i)$$

Termi $\max(0, 1 - y_i f(\mathbf{x}_i))$ është **humbja hinge** - zero për pikat e klasifikuara saktë jashtë marginit, lineare për pikat brenda ose përtej marginit.

#### Zgjerimi Shumë-Klasësh (One-vs-Rest)

Për $K = 3$ klasa, trajnohen tre SVM binare:
- $f_1$: High vs. {Low, Medium}
- $f_2$: Low vs. {High, Medium}
- $f_3$: Medium vs. {High, Low}

$$\hat{y} = \arg\max_{k} \; f_k(\mathbf{x})$$

#### Shkallëzimi Platt (probability=True)

Funksioni i vendimit $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ konvertohet në probabilitet nëpërmjet sigmoid:

$$P(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{Af(\mathbf{x}) + B}}$$

Parametrat $A$ dhe $B$ përshtaten me maximum likelihood mbi një set validimi (validim i kryqëzuar 5-fold brenda). Kjo mundëson llogaritjen e ROC-AUC.

#### Kuptimet e Hiperparametrave

| Parametri | Vlera | Efekti |
|---|---|---|
| `C` | 50 | C i lartë → margin i vogël, ndëshkim i rëndë i keq-klasifikimeve |
| `kernel` | linear | Kufiri i vendimit është një hiperplan në hapësirën origjinale të veçorive |

#### Rezultatet e Fazës III

| Metrika | Vlera |
|---|---|
| CV F1 (macro) | 0.9904 |
| Accuracy | 0.9839 |
| F1 (macro) | 0.9838 |
| ROC-AUC (macro) | 0.9995 |

**Sipas klasës:**
| Klasa | Precision | Recall | F1 | Mbështetja |
|---|---|---|---|---|
| High | 1.00 | 0.99 | 1.00 | 105 |
| Low | 0.97 | 0.99 | 0.98 | 99 |
| Medium | 0.98 | 0.97 | 0.98 | 106 |

---

### 9.5 Rrjeti Nervor - Perceptroni Shumështresor

**Lloji:** Rrjet nervor i plotë-lidhur feedforward

**Hiperparametrat më të mirë të gjetur:** `hidden_layer_sizes=(64, 32), alpha=0.001, learning_rate_init=0.005`

**Arkitektura:**

```
Shtresa Hyrëse:        9 neurone   (një për çdo veçori të zgjedhur)
Shtresa e Fshehur 1:  64 neurone + aktivizimi ReLU
Shtresa e Fshehur 2:  32 neurone + aktivizimi ReLU
Shtresa Dalëse:        3 neurone + Softmax (një për çdo klasë)
```

#### Kalimi Përpara (Forward Pass)

Për një rrjet me $L$ shtresa, dalja e shtresës $l$ është:

$$\mathbf{a}^{(l)} = g\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

Ku:
- $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ - matrica e peshave të shtresës $l$
- $\mathbf{b}^{(l)} \in \mathbb{R}^{d_l}$ - vektori i shtesave
- $g(\cdot)$ - funksioni i aktivizimit

**Shtresat e fshehura - aktivizimi ReLU:**

$$g(z) = \max(0, z) = \begin{cases} z & \text{nëse } z > 0 \\ 0 & \text{nëse } z \leq 0 \end{cases}$$

ReLU preferohet ndaj sigmoid/tanh sepse:
- Nuk ka problem me zhdukjen e gradientit për aktivizime pozitive ($g'(z) = 1$ për $z > 0$)
- Aktivizime të rralla (shumë neurone dalin 0) → regularizim implicit
- Llogaritisht efikas

**Shtresa dalëse - Softmax:**

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

#### Funksioni i Humbjes - Entropia e Kryqëzuar + Regularizimi L2

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} \mathbf{1}[y_i = k] \log P(y_i = k \mid \mathbf{x}_i) + \frac{\alpha}{2} \sum_{l} \|\mathbf{W}^{(l)}\|_F^2$$

Ku:
- $\alpha = 0.001$ - forca e regularizimit L2 (ndëshkon peshat e mëdha, redukton mbi-përshtatjen)
- $\|\mathbf{W}\|_F^2 = \sum_{i,j} W_{ij}^2$ - norma Frobenius (shuma e peshave në katror)

#### Propagimi Prapa (Backpropagation)

Gradientët llogariten nëpërmjet rregullit të zinxhirit prapa nëpër rrjet:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(l)}} \cdot \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{W}^{(l)}}$$

Për ReLU: $\frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} = \mathbf{1}[z_j^{(l)} > 0]$

#### Optimizuesi - Adam (Vlerësimi Adaptive i Momentit)

MLP përdor Adam, i cili mban mesatare të lëvizshme eksponenciale të gradientit $m_t$ dhe gradientit në katror $v_t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

Vlerësimet e korrigjuara nga bias: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$

Përditësimi i peshave:
$$\mathbf{W}_t = \mathbf{W}_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t, \quad \eta = \text{learning\_rate\_init} = 0.005$$

Parazgjedhja: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

#### Kuptimet e Hiperparametrave

| Parametri | Vlera | Efekti                                                              |
|---|---|---------------------------------------------------------------------|
| `hidden_layer_sizes` | (64, 32) | Gjerësia e rrjetit - 64 neurone pastaj 32; thellohet por ngushtohet |
| `alpha` | 0.001 | Forca e penalizimit L2 - regularizim moderate                       |
| `learning_rate_init` | 0.005 | Madhësia fillestare e hapit Adam                                    |
| `max_iter` | 1000 | Epokat maksimale të trajnimit                                       |

**Shënim mbi early_stopping:** Çaktivizuar për shkak të një papërputhshmërie `numpy.isnan` me etiketat e klasave si vargje në këtë version sklearn/numpy. `max_iter=1000` kompenson duke lejuar epoka trajnimi të mjaftueshme.

#### Rezultatet e Fazës III

| Metrika | Vlera |
|---|---|
| CV F1 (macro) | 0.9871 |
| Accuracy | 0.9742 |
| F1 (macro) | 0.9743 |
| ROC-AUC (macro) | 0.9989 |

**Sipas klasës:**
| Klasa | Precision | Recall | F1 | Mbështetja |
|---|---|---|---|---|
| High | 0.96 | 0.99 | 0.98 | 105 |
| Low | 0.98 | 0.99 | 0.98 | 99 |
| Medium | 0.98 | 0.94 | 0.96 | 106 |

---

## 10. Metrikat e Vlerësimit - Formula të Plota

Për çdo klasë $k$, definohet:
- $TP_k$ = pozitivë të vërtetë për klasën $k$
- $FP_k$ = pozitivë të rremë për klasën $k$ (klasa të tjera parashikuara si $k$)
- $FN_k$ = negativë të rremë për klasën $k$ ($k$ parashikuar si klasë tjetër)

### Saktësia (Accuracy)

$$\text{Accuracy} = \frac{\sum_k TP_k}{n} = \frac{\text{të klasifikuara saktë}}{\text{mostrat totale}}$$

### Precizioni (Precision)

$$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}$$

"Nga të gjitha mostrat e parashikuara si klasë $k$, çfarë fraksioni vërtet është klasa $k$?"

### Ndjeshmëria (Recall)

$$\text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

"Nga të gjitha mostrat e vërteta të klasës $k$, çfarë fraksioni parashikuam saktë?"

### Rezultati F1

$$\text{F1}_k = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k} = \frac{2 \cdot TP_k}{2 \cdot TP_k + FP_k + FN_k}$$

Mesatarja harmonike e precizionit dhe ndjeshmërisë - ndëshkon modelet që sakrifikojnë njërin për tjetrin.

### F1 Makro-Mesatare

$$\text{F1}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{F1}_k$$

Çdo klasë kontribuon në mënyrë të barabartë, pavarësisht numrit të mostrave. Kjo është metrika kryesore e renditjes - siguron që modeli të mos gjykohet vetëm mbi klasën e shumicës.

### F1 e Peshuar

$$\text{F1}_{\text{peshuar}} = \frac{\sum_{k=1}^{K} n_k \cdot \text{F1}_k}{\sum_{k=1}^{K} n_k}$$

Peshohet sipas mbështetjes së klasës - më informuese kur klasat janë të pabalancuara.

### ROC-AUC (Macro, One-vs-Rest)

Për çdo klasë $k$, trajnohet një klasifikues binar (klasa $k$ vs. të gjithë të tjerët). Kurba ROC paraqet:

$$\text{TPR}_k(\tau) = \frac{TP_k(\tau)}{TP_k(\tau) + FN_k(\tau)}, \quad \text{FPR}_k(\tau) = \frac{FP_k(\tau)}{FP_k(\tau) + TN_k(\tau)}$$

ndërsa pragu i vendimit $\tau$ varion nga 0 në 1.

$$\text{AUC}_k = \int_0^1 \text{TPR}_k(\text{FPR}) \; d(\text{FPR})$$

Makro-mesatare:

$$\text{ROC-AUC}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{AUC}_k$$

AUC = 1.0 do të thotë ndarje perfekte; AUC = 0.5 do të thotë klasifikues rastësor.

**Të gjitha modelet e Fazës III arritën ROC-AUC > 0.998** - kalibrimi gati-perfekt i probabilitetit dhe ndarja e klasave.

---

## 11. Rëndësia Statistikore - Testi Wilcoxon Signed-Rank

Për të konfirmuar se Gradient Boosting është **statistikisht** superior (jo vetëm numerikisht), kryhet një test Wilcoxon signed-rank mbi rezultatet F1 të CV-së për-fold.

### Konfigurimi

- I njëjti objekt `StratifiedKFold(n_splits=5)` përdoret për të gjitha modelet → fold-e identike → **krahasim i çiftëzuar**
- Referenca: **Gradient Boosting** (CV F1 më i lartë = 0.9992)
- Testi: njëanësor, $H_1$: F1 e Gradient Boosting > F1 e modelit tjetër

### Procedura Wilcoxon Signed-Rank

Për $n = 5$ diferenca të çiftëzuara $d_i = \text{GB}_i - \text{Modeli}_i$:

1. Llogaritet $|d_i|$ dhe renditen nga më i vogli tek më i madhi
2. Çdo renditje i caktohet shenja e $d_i$
3. Llogaritet $W^+ = \sum_{\{i: d_i > 0\}} R_i$ (shuma e renditjeve të diferencave pozitive)
4. Nën $H_0$, $W^+$ ndjek një shpërndarje diskrete të njohur; llogaritet p-vlera

$$W^+ \geq W^+_{\text{kritike}} \Rightarrow \text{refuzohet } H_0$$

**Pse Wilcoxon dhe jo t-testi?**
T-testi supozon diferenca të shpërndara normalisht. Me $n=5$ vëzhgime të çiftëzuara, normaliteti nuk mund të verifikohet. Wilcoxon është një test jo-parametrik që kërkon vetëm simetrinë e diferencave - një supozim më i dobët dhe më i mbrojtueshëm.

**P-vlera minimale e arritshme me n=5 (njëanësor):**

$$P(W^+ = 15) = \frac{1}{2^5} = \frac{1}{32} = 0.03125 < 0.05$$

(arrihet kur të 5 diferenrcat janë pozitive dhe në drejtimin e pritur)

### Rezultatet Aktuale

| Modeli | Mesatare F1 | Statistika (W+) | p-vlera | Domethënës? |
|---|---|-----------------|---------|-------------|
| **Gradient Boosting** | 0.9992 | - (referenca)   | -       | -           |
| Isolation Forest | 0.9984 | 1.000           | 0.5000  | Jo          |
| SVM (Linear) | 0.9904 | 10.000          | 0.0625  | Jo          |
| Regresion Logjistik | 0.9872 | 15.000          | 0.0312  | **PO**      |
| Rrjeti Nervor (MLP) | 0.9871 | 15.000          | 0.0312  | **PO**      |

**Interpretimi:**
- Gradient Boosting është **statistikisht superior** ndaj Regresionit Logjistik dhe MLP (p < 0.05)
- Dallimi midis Gradient Boosting dhe Pyllit të Rastit **nuk është statistikisht domethënës** (p = 0.50) - ato performojnë esencialisht identikisht mbi të 5 fold-et
- Dallimi midis Gradient Boosting dhe SVM Linear është kufitar (p = 0.0625, pak mbi α = 0.05)

**Përfundimi:** Gradient Boosting shpallet fitues bazuar në CV F1 më të lartë (0.9992) dhe superioritetin statistikisht të konfirmuar ndaj modeleve më të dobëta.

---

## 12. Mjetet Shtesë të Analizës

Tre mjete u shtuan në Fazën III specifikisht për të shkuar përtej metrikave skalare dhe për t'iu përgjigjur pyetjeve më të thella: *Pse vendos modeli kështu? A është vërtet superior ndaj Fazës II në parashikime individuale? Sa i ndjeshëm është modeli më i mirë ndaj hiperparametrit kryesor?*

---

### 12.1 Testi McNemar - Faza II vs Faza III

#### Çfarë është

Testi McNemar është një **test i çiftëzuar jo-parametrik** që krahason dy klasifikues mbi *të njëjtën test set*. Ndërsa Wilcoxon krahason rezultatet e fold-eve të validimit të kryqëzuar (vlerësim i kohës së trajnimit), McNemar krahason gabimet aktuale të parashikimit mbi të dhënat e mbajtura - një krahasim më i fortë dhe më i drejtpërdrejtë.

#### Pse jo thjesht të krahasojmë accuracy-n?

Nëse Faza II ka Accuracy=99.03% dhe Faza III ka Accuracy=99.03%, nuk mund të dallohet nga metrika skalare nëse *të njëjtat* 3 mostra keq-klasifikohen ose nëse Faza III ka të drejtë aty ku Faza II gaboi (dhe anasjelltas). Testi McNemar e provon drejtpërdrejt këtë.

#### Konfigurimi në Fazën III

1. Një model **Gradient Boosting ekuivalent Fazës II** ri-trajnohet mbi të njëjtin set trajnimi me zgjedhje veçorish (`X_train_sel`) duke përdorur parametra të afërt me Fazën II:
   ```
   n_estimators=200, learning_rate=0.1, max_depth=3,
   subsample=1.0, min_samples_split=2
   ```
2. Të dy modelet vlerësohen mbi **të njëjtën** `X_test_sel` (310 mostra)

#### Tabela e Kushtëzuar

|  | **Ph3 saktë** | **Ph3 gabim** |
|---|---|---|
| **Ph2 saktë** | $a$ (të dy saktë) | $c$ (Ph2 saktë, Ph3 gabim) |
| **Ph2 gabim** | $b$ (Ph2 gabim, Ph3 saktë) | $d$ (të dy gabim) |

Nën $H_0$: $b = c$ (të dy klasifikuesit bëjnë të njëjtin numër gabimesh diskordante).

#### Statistika e Testit (Binomiale Ekzakte)

Testi ekzakt McNemar përdor shpërndarjen binomiale mbi çiftet diskordante $(b, c)$:

$$p\text{-vlera} = 2 \sum_{k=0}^{\min(b,c)} \binom{b+c}{k} \left(\frac{1}{2}\right)^{b+c}$$

Kjo është probabiliteti i vëzhgimit të një ndarjeje po aq ekstreme si $(b, c)$ rastësisht nëse të dy klasifikuesit vërtet kanë të njëjtën shpërndarje gabimesh. Kur $b > c$ dhe $p < 0.05$, Faza III bën dukshëm më pak gabime.

#### Pse McNemar dhe jo z-testi?

Testi McNemar merr parasysh natyrën **të çiftëzuar** (të korreluar) të parashikimeve - të dy modelet shohin të njëjtën mostër testimi. Një test jo-çiftëzuar si z-testi për proporcionet do të ishte i gabuar sepse supozimi i pavarësisë do të shkelej (e njëjta mostër parashikohet nga të dy modelet).

---

### 12.2 SHAP - Interpretueshmëria e Modelit

#### Çfarë është SHAP

**SHapley Additive exPlanations** (Lundberg & Lee, 2017) i cakton një vlerë kontributi $\phi_j$ çdo veçorie $j$ për një parashikim specifik. Mbështetet në **teorinë e lojërave kooperative** - çdo veçori trajtohet si "lojtar" dhe vlera e saj SHAP është pjesa e saj e drejtë e parashikimit total.

#### Formula e Vlerës Shapley

Për një model $f$, vlera SHAP e veçorisë $j$ për mostrën $\mathbf{x}$ është:

$$\phi_j(\mathbf{x}) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \left[f(S \cup \{j\}) - f(S)\right]$$

Ku:
- $F$ = bashkësia e plotë e veçorive
- $S$ = një nënbashkësi veçorish **pa përfshirë** $j$
- $f(S)$ = dalja e modelit duke përdorur vetëm veçoritë në $S$ (të tjerat zëvendësohen me pritshmërinë e tyre marxhinale)
- Fraksioni është një **faktor peshimi** që mesatarizon mbi të gjitha renditjet e mundshme të veçorive

Kjo është llogaritimisht e parealizueshme për $|F|$ të madh - llogaritja ekzakte kërkon $2^{|F|}$ vlerësime modeli.

#### TreeSHAP (Llogaritje Ekzakte Efikase)

Për ansambël pemësh (Isolation Forest, Gradient Boosting), **TreeSHAP** llogarit vlera ekzakte Shapley në **kohë polinomiale** $O(TLD^2)$ duke shfrytëzuar strukturën e pemës:
- $T$ = numri i pemëve
- $L$ = gjethet maksimale për pemë
- $D$ = thellësia maksimale

Njohja kryesore: kalimi nëpër një shteg të vetëm në pemën e vendimit tashmë kushtezon mbi një nënbashkësi veçorish. TreeSHAP ripërdor këto kalime për të llogaritur vlera ekzakte Shapley pa asnjë aproksimim.

#### Çfarë Tregojnë Grafet

**Grafiku i shtylave të nevojës globale** (`shap_feature_importance.png`):

$$\text{Rëndësia}(j) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{K} \sum_{k=1}^{K} |\phi_j^{(k)}(\mathbf{x}_i)|$$

Vlera mesatare absolute SHAP e mesatarizuar mbi mostra dhe klasa. Veçoritë me vlera të larta këtu i shtyjnë parashikimet larg bazës më fort, pavarësisht drejtimit.

**Grafiku beeswarm** (`shap_beeswarm.png`):

Për një klasë (p.sh., `High`), çdo pikë është një mostër testimi. Boshti x është vlera SHAP (pozitive = shtyn drejt `High`, negative = shtyn larg). Ngjyra kodon vlerën aktuale të veçorisë (e kuqe = e lartë, blu = e ulët). Ky grafik tregon:
- Cilat veçori janë më diskriminuese
- Nëse vlerat e larta të veçorive nxisin parashikime më të larta apo më të ulëta
- Mostrat outlier që janë të vështira për t'u klasifikuar

#### Pse SHAP dhe jo rëndësia standarde e veçorive?

| | Rëndësia e Ginit | Rëndësia e Permutacionit | SHAP |
|---|------------------|--------------------------|---|
| Merr parasysh ndërveprimet e veçorive | Jo               | Pjesërisht               | Po |
| Shpjegim për-mostër | Jo               | Jo                       | Po |
| Konsistent midis llojeve të modelit | Jo               | Po                       | Po |
| Ekzakt (pa aproksimim) për pemët | N/A              | Jo                       | **Po** |
| Trajton veçoritë e korreluara | Dobët            | Dobët                    | Më mirë |

Rëndësia standarde Gini numëron sa shpesh përdoret një veçori dhe ulja mesatare e papastërtisë - mbivlerëson veçoritë e korreluara dhe nuk jep informacion mbi drejtimin. Vlerat SHAP janë unike dhe plotësojnë aksiomët e **efikasitetit**, **simetrisë**, **dummyt** dhe **additivitetit** nga teoria e lojërave.

---

### 12.3 Kurba e Validimit - Ndjeshmëria ndaj Hiperparametrave

#### Çfarë tregon

Kurba e validimit paraqet rezultatin F1 të trajnimit dhe validimit të modelit ndërsa varion një hiperparametër i vetëm, duke mbajtur të tjerët fikse. I përgjigjet pyetjes:

> *"Sa i ndjeshëm është performanca e modelit ndaj këtij hiperparametri? Ku është optimumi? A mbi-përshtatet modeli për vlera të mëdha?"*

#### Pse learning_rate për Gradient Boosting?

`learning_rate` (tkurrja $\eta$) është hiperparametri më ndikues i vetëm i Gradient Boosting. Kontrollon drejtpërdrejt:
- **Nën-përshtatja** ($\eta$ i vogël, pemë të pakta): bias i lartë, të dy kurbat të ulëta
- **Përshtatje e mirë** ($\eta$ moderate): trajnimi ≈ validimi, të dyja të larta
- **Mbi-përshtatja** ($\eta$ i lartë): kurba e trajnimit qëndron e lartë, validimi bie

Kjo e bën atë boshtin më informues për një grafik diagnostik.

#### Implementimi

Sweepingu përdor `sklearn.model_selection.validation_curve` (fallback sklearn, pasi Yellowbrick është i papërputhshëm me Python 3.14 për shkak të heqjes së modulit `distutils` në Python 3.12+):

```python
param_range = [0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]

train_scores, val_scores = validation_curve(
    estimator,
    X_train, y_train,
    param_name="learning_rate",
    param_range=param_range,
    cv=StratifiedKFold(n_splits=5),
    scoring="f1_macro",
)
```

Për çdo vlerë në `param_range`, estimatori trajnohet 5 herë (një herë për fold). Mesatarja ± std e rezultateve të trajnimit dhe validimit paraqiten në grafik. Boshti x përdor shkallë logaritmike pasi `learning_rate` shtrihet mbi dy rende magnitude.

#### Kurba e të Mësuarit (sklearn)

Gjithashtu gjenerohet: një **kurbë të mësuari** që tregon si varion F1 me madhësinë e set-it të trajnimit (10% → 100%). Kjo diagnostikon:
- **Bias i lartë** (nën-përshtatja): të dyja kurbat plafonohen nën 1.0
- **Variancë e lartë** (mbi-përshtatja): kurba e trajnimit >> kurba e validimit me hendek të madh
- **Përshtatje e mirë**: kurbat konvergjojnë pranë 1.0 me rritjen e madhësisë trajnuese

Modeli më i mirë i Fazës III (Gradient Boosting) tregon kurbat që konvergjojnë shpejt → modeli nuk është i varfër nga të dhënat dhe nuk mbi-përshtatet.

---

## 13. Rezultatet

### Tabela e Plotë e Rezultateve të Fazës III

| Modeli | CV F1 | Accuracy | Precision | Recall | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.9992** | 0.9903 | 0.9904 | 0.9905 | 0.9904 | **0.9999** |
| Isolation Forest | 0.9984 | 0.9903 | 0.9904 | 0.9905 | 0.9904 | 0.9999 |
| SVM (Linear) | 0.9904 | 0.9839 | 0.9837 | 0.9840 | 0.9838 | 0.9995 |
| Regresion Logjistik | 0.9872 | 0.9806 | 0.9806 | 0.9807 | 0.9806 | 0.9993 |
| Rrjeti Nervor (MLP) | 0.9871 | 0.9742 | 0.9745 | 0.9746 | 0.9743 | 0.9989 |

### Fituesi: Gradient Boosting

```
Parametrat Më të Mirë : n_estimators=500, learning_rate=0.2, max_depth=5,
                        subsample=0.9, min_samples_split=5
CV F1 (macro)         : 0.9992
Accuracy              : 0.9903  (307/310 të sakta)
F1 (macro)            : 0.9904
ROC-AUC (macro)       : 0.9999
```

---

## 14. Krahasimi Faza II vs Faza III

| Modeli | F1 Ph2 | F1 Ph3 | Delta F1 | CV F1 Ph2 | CV F1 Ph3 |
|---|---|---|---|---|---|
| Regresion Logjistik | 0.9741 | 0.9806 | **+0.0065** | 0.9847 | 0.9872 |
| Isolation Forest | 0.9904 | 0.9904 | +0.0000 | 0.9968 | 0.9984 |
| Gradient Boosting | 0.9904 | 0.9904 | +0.0000 | 0.9984 | **0.9992** |
| SVM (Linear) | 0.9709 | 0.9838 | **+0.0129** | 0.9863 | 0.9904 |
| Rrjeti Nervor (MLP) | 0.9712 | 0.9743 | **+0.0031** | 0.9766 | 0.9871 |

**Vëzhgimet kryesore:**
1. **SVM Linear pati përmirësimin më të madh (+0.0129)** - gama më e gjerë e C (deri në 100) gjeti C=50 që gridi i Fazës II (max C=10) e kishte plotësisht jashtë
2. **Regresioni Logjistik u përmirësua me +0.0065** - C=100 ishte jashtë gridit të Fazës II
3. **F1 e testit të RF dhe GB nuk ndryshoi** - ato ishin tashmë gati-optimale; Faza III e konfirmoi me besueshmëri më të lartë CV
4. **CV F1 e GB u përmirësua nga 0.9984 → 0.9992** - hapësira më e madhe e parametrave (500 pemë, subsample=0.9) nxori variancën e mbetur
5. **Të gjitha modelet u përmirësuan ose qëndruan** - asnjë model nuk u degradua, duke konfirmuar se kërkimi i gjerë ishte i dobishëm

---

## 15. Skedarët e Prodhuar

### Raporte CSV / Tekst

| Skedari | Përshkrimi |
|---|---|
| `model_results_phase3.csv` | Të gjitha metrikat e 5 modeleve + hiperparametrat më të mirë |
| `comparison_phase2_vs_phase3.csv` | Tabela delta Faza II vs III (F1, CV F1, Accuracy) |
| `classification_reports_phase3.txt` | Precision/Recall/F1 i plotë për-klasë për të gjitha modelet |
| `wilcoxon_results.txt` | Testi Wilcoxon signed-rank: GB vs çdo model tjetër (5 fold-e) |
| `mcnemar_results.txt` | Testi McNemar: GB Faza II vs fituesi Faza III (gabimet e test set-it) |
| `final_report_phase3.md` | Raporti ekzekutiv përmbledhës |

### Vizualizimet Kryesore

| Skedari | Çfarë tregon                                                             | Pse ka rëndësi                                                     |
|---|--------------------------------------------------------------------------|--------------------------------------------------------------------|
| `algorithm_comparison_phase3.png` | Grafiku me shtylla grupore: Accuracy, Precision, Recall, F1 për 5 modelet | Krahasim vizual i shpejtë me shikim të parë                        |
| `phase2_vs_phase3_comparison.png` | Shtylla krah-për-krah F1 (Ph2 vs Ph3) me shënime Δ                       | Tregon drejtpërdrejt cilat modele u përmirësuan dhe me sa          |
| `feature_selection.png` | Shtylla rëndësish (blu=mbajtur, kuq=hequr) + vija e pragut               | Tregon cilat veçori u eliminuan dhe pse                            |
| `feature_importance_phase3.png` | Veçoritë kryesore sipas nevojës Gini të RF pas zgjedhjes                 | Cilat veçori nxisin ndarjet e ansamblit                            |
| `roc_auc_curves_phase3.png` | Kurbat ROC makro-mesatare për 5 modelet                                  | Vlerëson cilësinë e probabilitetit, jo vetëm parashikimet e ngurta |
| `calibration_curves_phase3.png` | Probabiliteti i parashikuar vs. fraksioni aktual për-klasë               | Kontrollon nëse 0.8 i parashikuar do të thotë 80% e herëve         |
| `confusion_matrix_*.png` | Heatmap 3×3 për çdo model (5 skedarë)                                    | Modeli i gabimit për-klasë - cilat klasa ngatërrohen               |

### Grafet Diagnostike dhe të Interpretueshmërisë

| Skedari | Çfarë tregon                                                | Pse ka rëndësi                                                                 |
|---|-------------------------------------------------------------|--------------------------------------------------------------------------------|
| `learning_curves_phase3.png` | F1 trajnimit vs. validimit me rritjen e madhësisë trajnuese | Diagnostikon bias-variancën; provon se modeli nuk mbi-përshtatet               |
| `shap_feature_importance.png` | Mean \|SHAP\| për-veçori (global) - modeli më i mirë        | Nevojë e bazuar në teorinë e modelit; merr parasysh ndërveprimet dhe drejtimin |
| `shap_beeswarm.png` | Vlerat SHAP për-mostër për një klasë - modeli më i mirë     | Tregon si shpjegohen mostrat individuale; zbulon outliers                      |
| `yellowbrick_validation_curve.png` | F1 vs sweep i `learning_rate` për GB                        | Zbulon shkallën optimale të mësimit dhe ndjeshmërinë ndaj këtij hiperparametri |

---

## 16. Konkluzionet dhe Impakti i Projektit

### Konkluzionet në lidhje me rezultatet
Pas tre fazave të eksperimentimit intensiv, përfundojmë se modeli Gradient Boosting (dhe ngjashëm Isolation Forest) arrin të parashikojë me saktësi pothuajse perfekte (Accuracy 99.03%, F1 0.9992) varësinë thelbësore mes variablave (energjia e rinovueshme, koha, etj.) dhe nivelit ditor të karbonit në Kosovë. Modelet dëshmuan se zgjedhja e targetit në tre klasa dhe inxhinieria e veçorive në Fazën I kanë qenë jashtëzakonisht efikase, duke filtruar zhurmën (prej 25 veçori në 9 themelore). Faza III plotësoi me sukses këtë detyrë, jo thjesht numerikisht, por duke e lidhur atë me rëndësi statistikore.

### Kontributi ynë unik (që të tjerët nuk e kanë dhënë)
Kontributi ynë i drejtpërdrejtë është lidhja e ndërtimit të një master-dataseti të personalizuar të Kosovës (agreguar nga miliona rreshta të dhënash orare nga 2021-2025 në një regjistër ditor, të pastruar e funksional) dhe validimi i modeleve përmes rigorozitetit real statistikor (Testi Wilcoxon dhe Testi McNemar). Rrallëherë datasetet publike për sektorin e energjisë së Kosovës janë të kthyera në probleme standarde të klasifikimit multiklasor me inxhinieri të tillë të detajuar (si p.sh. carbon_intensity_gap). Ne nuk kemi zbatuar vetëm modelin; ne kemi krijuar terrenin dhe problemin, e kemi vërtetuar se modeli e zgjidh, pastaj dhe e kemi provuar ashpër këtë përmes testeve statistikore dhe analizës SHAP.

### Si t'i lexojmë rezultatet, kujt i ndihmojnë dhe si?
Këto rezultate tregojnë probabilitetin se cila normë karboni ditor (`High`, `Medium`, `Low`) pritet bazuar në inputet ditore.
- **Kujt i ndihmojnë:** Ndihmojnë drejtpërdrejt institucionet e menaxhimit të rrjetit (si KOSTT), politikë-bërësit si dhe konsumatorët e mëdhenj (industritë) apo prosumer-at e energjisë solare në Kosovë.
- **Si i ndihmojnë:** Mundësojnë gjenerimin e alarmeve të automatizuara ditore; për shembull aktivizimin e incentiva financiare (tarifave të ulëta) kur mjedisi theksohet si `Low` ose dërgimin e sinjaleve për kufizime në prodhimet ndotëse kur parashikohet `High` intensity, duke u lidhur drejtpërdrejt me planin dhe axhendën e gjelbërt të reduktimit të emetimeve.

### Çka mund të bëhet në të ardhmen? (Future Work)
Si zhvillime të së ardhmes, modeli ditor mund të ngushtohet në një version predikues 'Real-Time' që jep target për orën e radhës në vend të ditës së radhës. Modele të kalibruara mund të shtohen me analiza të kohës (Time-Series Forecasting me LSTM), dhe puna mund të pasurohet më shumë duke lidhur një API meteorologjike për të marrë atributet si era, ndriçimi i diellit apo rreshjet - faktorë që padiskutim do i jepnin përgjigje origjinës së variacionit të gjetur. Aplikimi i menjëhershëm pastaj mund të bëhej integrimi i modeleve të fitura nga Faza III në ndonjë web/mobile-app ose dashboard ku mund të kyçen palët e prekura drejtpërdrejt në këtë domen.

---

## Referenca

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189–1232.
- Friedman, J. H. (1999). *Stochastic Gradient Boosting*. Computational Statistics & Data Analysis, 38(4), 367–378.
- Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. JMLR, 13, 281–305.
- Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273–297.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines*. Advances in Large Margin Classifiers.
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. ICLR 2015.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 16, 321–357.
- He, H. et al. (2008). *ADASYN: Adaptive Synthetic Sampling Approach*. IJCNN 2008.
- Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods*. Biometrics Bulletin, 1(6), 80–83.
- McNemar, Q. (1947). *Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages*. Psychometrika, 12(2), 153–157.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.
- Lundberg, S. M. et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees*. Nature Machine Intelligence, 2, 56–67.
