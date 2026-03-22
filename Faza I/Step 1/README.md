# Step 1: Initial Data Overview

**Qëllimi:** Ngarkimi i të dhënave orare dhe bashkimi (merging) i tyre në një dataframe të plotë.  

**Veprimet kryesore:**
- 5 skedarë vjetorë orarë (`hourly-2021.csv`, `hourly-2022.csv`, `hourly-2023.csv`, `hourly-2024.csv`, `hourly-2025.csv`) u lexuan dhe shpresuan se do te bashkoheshin duke dhënë një matricë gjigande analitike origjinale e barabartë me 43,824 rreshta orarë e 12 kolona origjinale.
- Për shkak të mundësisë trajnuese ne aplikuam grupim dhe agregim thelbësor (grupim) ditor duke shënuar mesataren matematikore të ditës. Kjo redukton matjen drastikisht në rreshta ditorë: zbatueshmëri e thellë prej 1,826 rreshta x 16 kolona paralelisht të mbikëqyrura ditorisht.
- U krye një "schema audit" specifik nëpërmjet programit mbikqyrës për t'u garantuar numeriksht për llojet e data-types (obj, int, float) the numrin kardinal, ku vetëm 6 prej 16 kolona tregohen "High Cardinality".

**Rezultati mbikëqyrës ditor:** Pasqyrime thelbësore depozitohen vazhdimisht edhe thellë në terminal pas verifikimit logjik ekzekutiv, duke vulosur integritetin fillestar të fajllit ditor.
