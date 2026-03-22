# Step 2: Core Data Cleaning

**Qëllimi:** Trajtimi i filtrave të ndotura dhe imputimi i vlerave (values) boshe.

**Veprimet kryesore algoritmike:**
- Zakonisht, të dhënat elektrike nga rajoni ynë (si p.sh matës të KEK-ut pa mbrojtje serverike) vuajnë theksueshëmnga ndotje ditore, kështu që moduli dedektiv `Constraints` u thirr algoritmikisht për të pastruar.
- Sidoqoftë thellësisht falë ndërvepruesit inteligjent Electricity Maps API, ky dataset vjen vizualisht the numerikisht i mbikëqyrur paraprakisht (mbeshtetur në metoden shkencore origjinale pre-kalkuluese: *ESTIMATED_TIME_SLICER_AVERAGE* te cilen po e thekson ndër-kolona shtesë).  
- Nuk pati domosdoshmëri për hedhje / detyrim artificial vlerash thelbësore sepse Null values ekzaktësisht mungonin. Skripti kreu 0 Imputime dhe 0 Fshirje të papritura të rasteve (cases).
- Rrjedha mbeti matematikisht e njëjta sa edhe në vëllimin e saj të origjinës nga hapi përcjellës: 1826 rreshta dhe 16 kolona.

**Rezultati kryesor:** Verifikim plotësisht i sigurt matematikor. Modeli mund të nis klasifikimet strategjikë ditorë i papenguar.
