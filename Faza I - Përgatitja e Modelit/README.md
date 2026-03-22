# Faza I: Përgatitja dhe Inxhinieria e të Dhënave (Data Preparation & Engineering)

Ky projekt realizon fazën e parë të procesit analitik për ndotjen e karbonit në sistemin elektroenergjetik të Kosovës (2021-2025). Gjatë kësaj faze, ne kemi transformuar të dhënat e papërpunuara orare në një dataset të strukturuar, të pastruar dhe gati për trajnimin e modeleve të Machine Learning.

---

## 🛠️ Përmbledhje e Procesit Modular

Pipeline-i është ndërtuar në **8 hapa ekzekutues**, ku secili folder përmban skriptin e tij (`step_X.py`) dhe dokumentacionin përkatës.

### 1. [Hapi 1 - Ngarkimi dhe Bashkimi](./Hapi%201%20-%20Ngarkimi%20dhe%20Bashkimi)
- **Qëllimi:** Agregimi i 5 viteve të dhënash orare në ditor.

### 2. [Hapi 2 - Pastrimi i të dhënave](./Hapi%202%20-%20Pastrimi%20i%20te%20dhenave)
- **Qëllimi:** Sigurimi i integritetit matematikor dhe imputimi.

### 3. [Hapi 3 - Diskretizimi dhe Inxhinieria](./Hapi%203%20-%20Diskretizimi)
- **Qëllimi:** Heqja e kolonave redundante (`Country`, `Zone`), fshirja e kolonave `Date` dhe zevendesimi me `day, month, year` si kolonat kryesore.

### 4. [Hapi 4 - Detektimi i Outliers](./Hapi%204%20-%20Detektimi%20i%20Outliers)
- **Qëllimi:** Auditimi dhe identifikimi i anomalive (Flagging).
- **Veprimet:** Aplikimi i 3 metodave për të parë saktësinë e të dhënave pa i fshirë ato ende.

### 5. [Hapi 5 - Balancimi dhe Mostrimi](./Hapi%205%20-%20Balancimi%20dhe%20Mostrimi)
- **Qëllimi:** Mbrojtja nga anshmëria dhe **Fshirja e Outliers**.
- **Veprimet:** Fshin 276 anomali dhe ndan të dhënat në Train/Test.

### 6. [Hapi 6 - Agregimi](./Hapi%206%20-%20Agregimi)
- **Qëllimi:** Analiza mujore e trendeve.

### 7. [Hapi 7 - Finalizimi i Datasetit](./Hapi%207%20-%20Finalizimi%20i%20Datasetit)
- **Qëllimi:** Gjenerimi i datasetit të plotë përfundimtar pa anomali.

### 8. [Hapi 8 - Raporti Final](./Hapi%208%20-%20Raporti%20Final)
- **Qëllimi:** Dokumentimi dhe vizualizmi përmes grafikëve dhe PDF.

---

## 📈 Konkluzionet e Fazës I
- **Cilësia:** Janë larguar ~15% e të dhënave që ishin anomali (Outliers).
- **Gatishmëria:** Faza II (Modelimi) ka tashmë inpute të balancuara dhe të pastruara.
