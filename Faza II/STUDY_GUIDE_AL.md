# 🎓 Faza 2 — Udhëzues i Plotë Studimi (Për Mbrojtjen e Projektit)

> **Çka është ky dokument?** Ky është udhëzues studimi që sqaron gjithçka që projekti ynë bën në Fazën 1 dhe Fazën 2 në mënyrën më të thjeshtë të mundshme. Lexojeni përpara mbrojtjes (prezantimit) që të mund t'i përgjigjeni me besim pyetjeve të profesorit.

---

## 🔰 Pamja e Përgjithshme: Për Çka Bëhet Fjalë?

Kemi të dhëna të elektricitetit nga rrjetet energjetike (2021–2025). Çdo ditë, dimë gjëra si: sa energji solare u përdor, sa qymyr u dogj, sa përqind ishte e rinovueshme, etj.

Gjithashtu dimë **intensitetin e karbonit** — sa "e ndotur" ose "e pastër" ishte energjia atë ditë (e matur në gram CO₂ për kilowatt-orë).

**Qëllimi ynë:** Të ndërtojmë një program kompjuterik që shikon të dhënat energjetike të një dite dhe automatikisht parashikon nëse ndotja e karbonit ishte **E ULËT**, **MESATARE**, ose **E LARTË**.

Mendone si parashikimin e motit, por për ndotjen e karbonit.

---

## 📦 Faza 1: Përgatitja e të Dhënave (Çka kemi bërë tashmë)

Para se makina të mësojë diçka, të dhënat duhet të jenë të pastra dhe të organizuara. Si kur gatuan — duhet t'i lash, t'i presësh, dhe t'i përgatitësh përbërësit para se të gatuan.

### Hapi 1 — Ngarkuam 5 skedarë CSV (një për çdo vit, 2021–2025) dhe i bashkuam në një tabelë të madhe.

### Hapi 2 — Mesatarizuam të dhënat orare në ditore. Në vend të 24 rreshtave për ditë, morëm 1 rresht me vlerat mesatare.

### Hapi 3 — Pastruam të dhënat:
- Hoqëm vlerat e pamundshme (si përqindje -50%)
- Plotësuam vlerat mungese: numrat morën **medianën** (vlera e mesit, jo mesatarja — sepse mesatarja prishet nga vlerat ekstreme), teksti mori **modën** (vlera më e shpeshtë)

### Hapi 4 — Krijuam "çelësin e përgjigjeve" (variabla target):
- Morëm numrin e intensitetit të karbonit dhe e ndamë në **3 grupe të barabarta**
- E treta e poshtme → **"low" (e ulët)**
- E treta e mesme → **"medium" (mesatare)**
- E treta e sipërme → **"high" (e lartë)**
- Kjo kolonë e re quhet `target_quantile_class`

### Hapi 5 — Krijuam kolona të reja (feature engineering):
- **Bazuar në kohë:** muaji, dita, dita e javës, a është fundjavë?
- **Të llogaritura:** hendeku mes dy llojeve të matjeve të karbonit, përqindja e energjisë së rinovueshme brenda energjisë totale të pastër

### Hapi 6 — Gjetëm outlier-ët (pika shumë të pazakonta) me 3 metoda dhe i shënuam.

### Hapi 7 — Shkallëzuam gjithçka. Kolona të ndryshme kanë shkallë të ndryshme. I bëmë të gjitha me mesatare=0 dhe shtrirje=1 që asnjë kolonë të mos dominojë.

### Hapi 8 — Ndamë të dhënat:
- **80% → Set trajnimi** (makina mëson nga kjo)
- **20% → Set testimi** (e fshehim dhe e përdorim më vonë për të kontrolluar sa mirë mësoi)

### Hapi 9 — Balancuam klasat me **SMOTE**:
SMOTE krijon pika të reja artificiale për klasat me më pak shembuj. Zgjedh një pikë reale, gjen fqinjët e saj, dhe krijon një pikë të re mes tyre. Kështu makina nuk bëhet dembele duke parashikuar gjithmonë klasën më të madhe.

> ⚠️ SMOTE prek vetëm të dhënat e trajnimit. Të dhënat e testimit mbeten 100% reale.

---

## 🤖 Faza 2: Mësimi i Makinës (Çka bëmë tani)

Mësuam 6 algoritme "supervised" (të mbikëqyrura — japim përgjigjet korrekte) dhe 2 algoritme "unsupervised" (jo të mbikëqyrura — NUK japim përgjigje).

Për çdo algoritëm, përdorëm **GridSearchCV** — mjet që automatikisht provon kombinime të ndryshme cilësimesh dhe zgjedh më të mirën. Mendone si të provosh 20 palë këpucë dhe të mbash ato që përshtaten më mirë.

---

## 🟢 6 Algoritmet Supervised (Mbikëqyrëse)

### 1. Logistic Regression — "Vizatuesi i Vijave të Drejta"

**Çka bën me fjalë të thjeshta:**
Imagjino 366 ditë testimi si pika në letër. Disa pika janë ditë LOW (gjelbra), disa MEDIUM (verdha), disa HIGH (kuqe). Logistic Regression provon të vizatojë **vija të drejta** në letër për t'i ndarë tri ngjyrat në zonat e veta.

**Si mëson:**
1. Fillon me hamendje rastësore
2. Kontrollon sa gaboi (ky "rezultat gabimi" quhet **funksioni i kostos** ose **loss**)
3. Rregullon hamendjen pak në drejtimin që zvogëlon gabimin — kjo teknikë quhet **gradient descent** (imagjino një top që rrokulliset nëpër mal — natyrisht gjen pikën më të ulët)
4. Përsërit hapin 2–3 mijëra herë deri sa hamendjet ndalojnë se përmirësohen

**Cilësimi kryesor — C (Rregullimi/Regularizimi):**
C është si një rrotull ashpërsie:
- **C = 0.01** → Shumë strikt: "Mbaje shumë të thjeshtë." Por modeli mund të jetë TEPËR i thjeshtë. Kjo quhet **underfitting** (nën-përshtatje, si këpucë tepër të vogla).
- **C = 100** → Shumë i lirshëm: "Përshtatu çdo pike trajnimi perfektisht." Por modeli mund ta mësojë *përmendsh* datasetin në vend se të nxjerrë modele reale. Kjo quhet **overfitting** (tej-përshtatje, si këpucë të derdhuara vetëm për këmbën e majt — nuk përshtaten për të djathtën).
- **C = 10** → Pikërisht mirë. GridSearchCV e gjeti automatikisht.

**Rezultati: 98.09% saktësi** — vetëm 7 ditë nga 366 u parashikuan gabim!

---

### 2. Random Forest — "200 Ekspertët që Votojnë"

**Çka bën me fjalë të thjeshta:**
Imagjino të pyesësh 200 njerëz të ndryshëm të bëjnë diagramë po/jo për parashikimin e intensitetit. POR secili njeri sheh vetëm një pjesë rastësore të të dhënave. Pastaj, për të klasifikuar një ditë të re, të 200 votojnë, dhe shumica fiton.

**Pse është randomizimi i mirë?**
Nëse të 200 njerëzit shohin të njëjtat të dhëna, do të bëjnë të njëjtat gabime. Duke i dhënë secilit copa të ndryshme rastësore, gabimet e tyre janë të ndryshme dhe anullohen kur votojnë bashkë. Kjo quhet **ensemble learning** (mësim i bashkimit).

**Secili "njeri" është një Pemë Vendimmarrëse (Decision Tree):**
```
"A është energjia e rinovueshme > 50%?"
   ├── PO → "A është ditë pune?"
   │         ├── PO → Parashiko LOW ✅
   │         └── JO → Parashiko MEDIUM ✅
   └── JO → "A është hendeku i karbonit > 100?"
             ├── PO → Parashiko HIGH ✅
             └── JO → Parashiko MEDIUM ✅
```

**Cilësimet e gjetura:**
- Numri i pemëve: **200**
- Thellësia maksimale e secilës pemë: **20**

**Rezultati: 100% saktësi** — çdo parashikim ishte i saktë!

---

### 3. Gradient Boosting — "Zinxhiri i Korrigjimit të Gabimeve"

**Çka bën me fjalë të thjeshta:**
Random Forest ndërton pemët **njëkohësisht** (pavarësisht njëra-tjetrës). Gradient Boosting i ndërton **njëra pas tjetrës**, si stafetë:

1. Pema 1 bën parashikime — gabon disa
2. Pema 2 trajnohet vetëm për t'i rregulluar gabimet e Pemës 1
3. Pema 3 trajnohet për gabimet e mbetura pas Pemëve 1+2
4. ...vazhdon për 100 pemë

Secila pemë merr një **hap të vogël** drejt përgjigjes perfekte. **Learning rate** kontrollon sa i madh është secili hap.

**Cilësimet e gjetura:**
- 100 pemë me thellësi shumë të cekët (vetëm 3 nivele!)
- Learning rate 0.05 (hapa shumë të vegjël)
- Fuqia vjen nga bashkimi i shumë korrigjuesve të thjeshtë, jo nga një pemë komplekse

**Rezultati: 100% saktësi**

---

### 4. SVM me Kernel Linear — "Ndarësi me Hendekun më të Gjerë"

**Çka bën me fjalë të thjeshta:**
SVM = Support Vector Machine. Imagjino pikat si pika në letër. SVM gjen vijën që i ndan grupet me **hendekun (gap) më të gjerë të mundshëm** mes tyre.

```
🟢🟢🟢         |         🔴🔴🔴
  🟢🟢      ← gap →      🔴🔴
🟢🟢🟢         |         🔴🔴🔴
```

Pikat më afër vijës quhen **support vectors** — ato e "mbajnë" kufirin.

**Rezultati: 97.54% saktësi**

---

### 5. SVM me Kernel RBF — "Ndryshues i Formës"

**Çka bën me fjalë të thjeshta:**
Po sikur grupet nuk mund të ndahen me vijë të drejtë? Imagjino pika të kuqe të formojnë rreth brenda pikave kaltëra. Asnjë vijë e drejtë nuk funksionon!

**Truku i kernel-it** e zgjidh: RBF kernel matematikisht "ngrit" letrën e sheshtë 2D në hapësirë 3D ku rrethi bëhet kodër — dhe tani mund ta ndash me prerje horizontale!

**gamma** kontrollon sa "shumë" ndikon secila pikë trajnimi:
- Gamma e vogël = ndikim i gjerë = kufij të butë
- Gamma e madhe = ndikim lokal = kufij të valëzuar (rrezik overfitting)

**Rezultati: 96.17% saktësi** — në fakt MË KEQ se SVM Linear!

**Pse?** Pas shkallëzimit në Fazën 1, kufijtë mes klasave ishin pothuajse linearë. RBF krijoi kufij tepër kompleksë pa nevojë dhe performoi më keq. Mësimi: **më komplekse nuk do të thotë gjithmonë më mirë.**

---

### 6. Rrjeta Neurale (MLP) — "Mini-Truri"

**Çka bën me fjalë të thjeshta:**
Rrjeta neurale është si fabrikë me shumë kate. Lëndët e para (35 veçoritë tona) hyjnë në katin përdhes, kalojnë nëpër stacione përpunimi (neuronet) në secilin kat, dhe produkti i gatshëm (parashikimi) del në katin e fundit.

```
Kati Përdhes (Input):    35 veçori hyjnë
     ↓
Kati 1:    128 mini-kalkulatorë përpunojnë
     ↓
Kati 2:     64 mini-kalkulatorë përpunojnë më tutje
     ↓
Kati i Fundit (Output):  3 dyer: "low", "medium", "high"
                         Të dhënat dalin nga dera me
                         probabilitetin më të lartë
```

**Si punon secili neuron:**
1. Merr numra nga kati poshtë
2. Shumëzon secilin me një **peshë (weight)**
3. I mbledh bashkë
4. E kalon përmes një **funksioni aktivizimi** (ReLU: nëse pozitiv, mbaje; nëse negativ, bëje 0)
5. E dërgon lart

**Si mëson — Backpropagation:**
1. Të dhënat rrjedhin përpara nëpër rrjet (**forward pass**)
2. Në dalje, llogarisim gabimin — quhet **funksioni i kostos** (cost function)
3. Gjurmojmë mbrapa duke pyetur: "cila peshë kontriboi më shumë në gabim?" — kjo quhet **backpropagation**
4. Rregullojmë secilën peshë pak për ta zvogëluar gabimin — kjo quhet **gradient descent**
5. Përsërisim për çdo shembull trajnimi, shumë herë (secili kalim i plotë quhet **epokë**)

**Grafiku i Loss Curve** tregon gabimin duke u zvogëluar gjatë epokave — dëshmi se gradient descent funksionon. Fillon lartë (rrjeti nuk dinë asgjë) dhe bie deri sa barazohet (**konvergjencë** — mësoi sa mund të mësojë).

**Rezultati: 97.54% saktësi**

---

## 🔵 2 Algoritmet Unsupervised (Pa përgjigje)

### K-Means — "Gjej 3 Grupe sipas Afërsisë"

**Çka bën:**
1. Vendos rastësisht 3 pika qendrore
2. Cakto secilën pikë tek qendra më e afërt → formohen 3 grupe
3. Lëviz secilën qendër në mes të grupit
4. Përsërit deri sa qendrat ndalojnë

**Metoda Elbow:** Provojmë k=2,3,4,...,8 dhe shikojmë ku "përkulet" grafiku i cilësisë.

**Silhouette Score:** mat cilësinë e klasterëve nga -1 (e tmerrshme) deri +1 (perfekte).

**Rezultati ynë: 0.24** — i dobët. Grupet mezi formohen.

### Agglomerative — "Bashko nga Poshtë Lart"

Fillon me çdo pikë si grup i vet. Bashkon dy grupet më të afërta. Përsërit deri sa mbeten 3.

**Rezultati ynë: 0.24** — poashtu i dobët.

### Pse dështuan algoritmet unsupervised?
Pikat e "low", "medium" dhe "high" janë **të përziera bashkë**. Ndryshimi mes tyre dallohet vetëm kur i mëson modelit drejtpërdrejt se çka nënkuptojnë etiketat. Kjo **dëshmon që supervised learning nevojitet** për problemin tonë.

---

## 📊 PCA — Bërja e 35 Kolonave të Dukshme në 2D

Të dhënat kanë 35 kolona. Nuk mund të vizatosh graf 35-dimensional. **PCA** i ngjesh 35 dimensionet në 2 duke ruajtur informacionin më të rëndësishëm.

Bëmë 3 grafiqe krah-për-krah: klasterët e K-Means vs klasterët e Agglomerative vs etiketat e vërteta. Kjo tregon vizualisht sa të ndryshme janë grupet e unsupervised nga përgjigjet reale.

---

## 📈 Grafikët e Teorisë së Mësimit

### Learning Curves (Kurbat e Mësimit)
Tregon performancën kur rriten të dhënat e trajnimit:
- **Kaltër (trajnimi):** Duhet të jetë i lartë
- **Portokalli (validimi):** Duhet të jetë afër kaltrit

Kaltër i lartë por portokalli i ulët → **overfitting** (mësoi përmendsh)
Të dyja të ulta → **underfitting** (model tepër i thjeshtë)
Të dyja të larta dhe afër → **model i mirë** ✅

### Efekti i Regularizimit
Tregon si C ndikon Logistic Regression:
- Majtas (C i vogël) = tepër strikt → underfitting
- Djathtas (C i madh) = tepër i lirshëm → rrezik overfitting
- Mesi = pikë optimale

### Kurba e Kostos MLP
Tregon gabimin e rrjetës neurale duke u zvogëluar me kalimin e kohës ndërsa gradient descent funksionon.

---

## 🏆 Çka Arritëm

1. **Saktësi jashtëzakonisht e lartë:** Modelet më të mira (Random Forest, Gradient Boosting) arritën 100% saktësi. Edhe modeli më i thjeshtë (Logistic Regression) arriti 98%.

2. **Dëshmuam se supervised learning nevojitet:** Duke provuar edhe algoritmet unsupervised dhe duke treguar dështimin e tyre (silhouette score 0.24), dëshmuam që nivelet e karbonit nuk mund të zbulohen pa etiketa.

3. **Demonstruam të gjitha konceptet e ML-së nga syllabusi:** Gradient descent, cost functions, regularization, overfitting/underfitting, backpropagation, kernels, PCA, ensemble methods, neural networks.

4. **Ndërtuam pipeline të plotë dhe të riproduktushme:** Një komandë e vetme `python3 phase2_pipeline.py` e bën gjithçka dhe gjeneron 18 skedarë output.

5. **Krahasuam 8 algoritme sistematikisht:** 6 supervised + 2 unsupervised, të gjitha me hyperparameter tuning dhe cross-validation.

---

## 🎯 Pse i Bëmë Gjërat Kështu

| Vendimi | Pse |
|---------|-----|
| 3 klasa (low/medium/high) | Më realiste. 2 klasa do të ishin tepër të lehta. |
| GridSearchCV | Për të gjetur parametrat optimal automatikisht |
| 3-fold cross-validation | Për t'u siguruar që rezultatet janë stabile |
| Supervised DHE Unsupervised | Syllabusi kërkon të dyja. Gjithashtu dëshmon se unsupervised nuk mjafton. |
| 2 kernele SVM | Për të demonstruar konceptin e "kernels" nga syllabusi |
| Rrjeta Neurale | Për të demonstruar backpropagation dhe gradient descent |
| Learning curves | Për të demonstruar overfitting/underfitting |
| SMOTE në Fazën 1 | Pa balancim, modelet do të "mashtroheshin" duke parashikuar gjithmonë klasën më të madhe |
| StandardScaler | SVM dhe Neural Networks performojnë tmerrësisht me të dhëna pa shkallëzim |

---

## 🔮 Hapat e Ardhshëm (Parashikim i Fazës 3)

Faza 3 ka afat 17 Maj. Titullohet "Analiza dhe evaluimi (ri-trajnimi)":

1. **XGBoost** — Version edhe më i fuqishëm i Gradient Boosting
2. **Deep Learning me TensorFlow/Keras** — Rrjeta neurale më të avancuara
3. **SHAP Analysis** — Teknikë për të sqaruar PSE secili model bëri secilin parashikim
4. **Regresion** — Parashikimi i numrit aktual (psh. 345.2 gCO₂/kWh) jo vetëm klasës
5. **Cross-validation më i thellë** — StratifiedKFold me 5-10 folds
6. **Detektimi i anomalive** — Gjetja e ditëve vërtet të pazakonta
7. **Krahasimi me Fazën 2** — Tregojmë si përmirësimet ndikuan rezultatet
8. **Konkluzionet finale** — Kush përfiton nga kjo punë, çka mund të përmirësohet

---

## ❓ Pyetjet e Profesorit — Përgatitja për Mbrojtje

### Pyetje të Përgjithshme

**P: "Çka është Machine Learning?"**
> "Machine Learning është kur i japim kompjuterit të dhëna dhe e lëmë të gjejë modele vetë, në vend se ne të shkruajmë rregulla eksplicite. Në supervised learning, i japim shembuj me përgjigje korrekte që të mësojë lidhjen. Në unsupervised learning, i japim vetëm të dhëna pa përgjigje dhe ai provon të gjejë grupime natyrale."

**P: "Për çka bëhet fjalë në projektin tuaj?"**
> "Parashikojmë nivelin e intensitetit të karbonit të rrjeteve energjetike — nëse do të jetë i ulët, mesatar, ose i lartë — bazuar në veçoritë e prodhimit energjetik si përqindja e rinovueshme, energjia pa karbon, dhe modelet kohore. Përdorim të dhëna orare nga 2021 deri 2025 të agreguara në nivel ditor."

**P: "Çka është target variabla?"**
> "Targeti ynë është `target_quantile_class` me tri klasa: low, medium, dhe high. E krijuam duke ndarë vlerat e vazhdueshme të intensitetit të karbonit në tri grupe të barabarta me quantile binning."

### Pyetje për Algoritmet

**P: "Pse i zgjodhët këto algoritme specifike?"**
> "Deshëm të mbulojmë familjet kryesore nga syllabusi: modele lineare (Logistic Regression), metoda ensemble me pemë (Random Forest, Gradient Boosting), metoda kernel (SVM me dy kernele), dhe rrjeta neurale (MLP). Plus metoda unsupervised (K-Means, Agglomerative) për të treguar dallimin mes qasjeve supervised dhe unsupervised."

**P: "Si funksionon Random Forest?"**
> "Krijon shumë pemë vendimmarrëse, secila e trajnuar në një nënshtresë rastësore të të dhënave dhe veçorive. Secila pemë bën parashikimin e vet, dhe përgjigja finale vendoset me votim të shumicës. Randomizimi siguron që pemët bëjnë gabime të ndryshme, që anullohen kur votojnë."

**P: "Si funksionon Gradient Boosting?"**
> "Ndërton pemë në mënyrë sekuenciale. Secila pemë e re trajnohet për t'i rregulluar gabimet e të gjitha pemëve paraprake. Përdor gradient descent për ta minimizuar gabimin hap pas hapi. Learning rate kontrollon sa kontribuon secila pemë."

**P: "Cili është dallimi mes Random Forest dhe Gradient Boosting?"**
> "Random Forest ndërton pemë pavarësisht dhe paralelisht — secila sheh të dhëna rastësore. Gradient Boosting ndërton pemë sekuencialisht — secila fokusohet specifikisht në rregullimin e gabimeve. Të dyja janë metoda ensemble por me strategji të ndryshme."

**P: "Si mëson Rrjeta Neurale?"**
> "Përmes backpropagation dhe gradient descent. Të dhënat rrjedhin përpara nëpër shtresa neuronesh. Në dalje, llogarisim gabimin. Pastaj gjurmojmë mbrapa nëpër rrjet për të parë sa kontribuoi secila peshë në gabim. Rregullojmë secilën peshë për ta zvogëluar gabimin. Përsërisim për çdo shembull trajnimi, shumë herë."

**P: "Çka është kernel në SVM?"**
> "Kernel është funksion matematik që transformon të dhënat në hapësirë me dimensione më të larta. Në hapësirën origjinale, të dhënat mund të mos ndahen me vijë të drejtë. Por në hapësirën me dimensione më të larta, bëhen të ndashme. RBF kernel përdor funksionin Gausian për këtë transformim."

**P: "Çka është PCA?"**
> "PCA zvogëlon numrin e dimensioneve duke ruajtur sa më shumë informacion. Gjen drejtimet ku të dhënat variojnë më shumë dhe i projekton gjithçka në ato drejtime. E përdorëm për ta vizualizuar datasetin tonë 35-dimensional në 2D."

### Pyetje Teorike

**P: "Çka është overfitting? Si e detektoni?"**
> "Overfitting ndodh kur modeli e mëson përmendsh datasetin e trajnimit në vend se të nxjerrë modele të përgjithshme. Performon shkëlqyeshëm në trajnim por dobët në të dhëna të reja. E detektojmë me learning curves — nëse training score është shumë më i lartë se validation score, ka overfitting."

**P: "Çka është underfitting?"**
> "Underfitting ndodh kur modeli është tepër i thjeshtë për t'i kapur modelet. Të dyja score-t, training dhe test, do të jenë të ulta. Zgjidhja: model më kompleks, më shumë veçori, ose zvogëlim i regularizimit."

**P: "Çka është regularizimi?"**
> "Regularizimi parandalon overfitting duke shtuar penalitet për kompleksitet. Në LR dhe SVM, parametri C e kontrollon. C i vogël = regularizim i fortë (model më i thjeshtë), C i madh = regularizim i dobët (model më kompleks). Grafiku ynë i regularizimit e tregon vizualisht."

**P: "Çka është gradient descent?"**
> "Algoritem optimizimi. Imagjino je në mal me mjegull të dendur dhe duhet të arrishë luginën. E ndinë tatëpjetën nën këmbë dhe gjithmonë shkon nëpër rrugën më te pjerrët tatëpjetë. Secili hap është i vogël, por përfundimisht arrin fundin. Në ML, 'mali' është sipërfaqja e gabimit dhe 'lugina' është pika ku modeli bën më pak gabime."

**P: "Çka është funksioni i kostos?"**
> "Mat sa gaboi modeli. Merr parashikimet dhe përgjigjet reale dhe prodhon një numër — gabimin. Qëllimi i trajnimit është minimizimi i këtij numri. Për klasifikim përdorim cross-entropy loss."

**P: "Çka është cross-validation?"**
> "Në vend se të evaluojmë modelin në një ndarje të vetme train/test, e ndajmë datasetin e trajnimit në 3 pjesë. Trajnojmë në 2 dhe testojmë në të tretën. Rrotullojmë 3 herë. Kjo na jep 3 rezultate që i mesatarizojmë, duke e bërë evaluimin më të besueshëm."

**P: "Çka është confusion matrix?"**
> "Tabelë që tregon saktësisht cilat klasa i ngatërroi modeli me cilat. Diagonalja tregon parashikime korrekte, gjithçka jashtë diagonales tregon gabime. Na tregon JO vetëm sa gabime, por ÇFARË lloj gabimesh."

**P: "Çka është F1-Score dhe pse përdoret?"**
> "F1 balancon Precision-in (kur modeli parashikon një klasë, sa shpesh ka të drejtë?) me Recall-in (nga të gjitha rastet reale të një klase, sa gjeti modeli?). Përdorim F1 macro, që mesatarizon F1 nëpër të gjitha klasat njëlloj."

### Pyetje për Rezultatet

**P: "Pse Random Forest arriti 100%?"**
> "Përgatitja në Fazën 1 ishte shumë e tërësishme — krijuam veçori informative si life_cycle_gap dhe renewable_share_within_cfe, aplikuam shkallëzim të duhur, dhe balancuam klasat. Modelet e bazuara në pemë mund t'i mësonin lehtë modelet e qarta."

**P: "A nuk është 100% saktësi e dyshimtë? A mund të jetë overfitting?"**
> "E verifikuam me cross-validation 3-fold gjatë GridSearchCV dhe CV score ishte 0.994 — shumë afër test score-it. Test seti ishte komplet i ndëveçantë (kurrë nuk u pa gjatë trajnimit). Nëse modeli do të ishte overfitted, test score do të ishte shumë më i ulët se CV score."

**P: "Pse SVM RBF perfomoi më keq se SVM Linear?"**
> "Pas preprocessing-ut (StandardScaler), kufijtë mes klasave u bënë pothuajse linearë. RBF kernel krijoi kufij tepër kompleksë pa nevojë, që dëmtoi performancën. Kjo dëshmon se më shumë kompleksitet nuk është gjithmonë më mirë."

**P: "Pse algoritmet unsupervised patën silhouette scores të ulta?"**
> "Sepse nivelet e karbonit nuk formojnë klasterë natyrorë. Ditë me profile energjetike të ngjashme mund të kenë intensitete shumë të ndryshme varësisht nga miksi energjetik. Pa etiketa eksplicite, algoritmet e klasterimit nuk mund t'i dallojnë."

**P: "Çka tregoi grafiku i regularizimit?"**
> "Me C shumë të vogël (regularizim i fortë), modeli nën-performon — underfitting. Me rritjen e C, performanca rritet deri sa stabilizohet. Kjo tregon vizualisht bias-variance tradeoff."

**P: "Çka treguan learning curves?"**
> "Me rritjen e madhësisë së training setit, validation score rritet dhe i afrohet training score-it. Kjo tregon se modeli generalizon mirë. Hendeku zvogëlohet, duke sugjeruar se nuk ka overfitting."

**P: "Cila është vlera praktike e këtij projekti?"**
> "Operatorët e rrjeteve energjetike mund ta përdorin për ta parashikuar nivelin e ndotjes paraprakisht. Nëse modeli parashikon HIGH intensive për nesër, mund të kalojnë në burime të rinovueshme proaktivisht. Kjo ndihmon në reduktimin e emisioneve dhe planifikimin e tranzicionit energjetik."
