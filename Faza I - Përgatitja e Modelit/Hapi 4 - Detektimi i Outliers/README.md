# Hapi 4: Detektimi i Outliers (Multi-Method Outlier Detection)

### **Qëllimi**
Ky hap mbron modelin nga vlerat ekstreme ose të pasakta (anomalitë) në sistemin elektroenergjetik të Kosovës, duke përdorur një qasje me tre metoda paralele.

---

### **Metodat e Detektimit:**
1.  **Z-Score:** Mat distancën e secilës vlerë nga mesatarja në njësi të devijimit standard.
2.  **IQR (Interquartile Range):** Metodë robuste që identifikon vlerat jashtë "kutive" të shpërndarjes.
3.  **Isolation Forest:** Një algoritëm Machine Learning (bazuar në pemët e vendimit) që izolon vlerat anomale.

### **Rregulla e Konsensusit (Consensus Rule):**
Për të mos fshirë të dhëna të rëndësishme pa nevojë, një rresht shënohet si **Outlier** vetëm nëse të paktën **dy nga tre** metodat e mësipërme pajtohen.

> [!NOTE]
> Ky hap (Hapi 4) shërben për **identifikimin dhe auditimin** e anomalive (flagging), ndërsa **Hapi 5** dhe **Hapi 7** i fshijnë ato automatikisht për të pasur të dhëna "të pastra" për trajnimin e modeleve.

---

### **Struktura e Skedarëve**
- **Input:** Dataseti i plotë `../Hapi 3 - Diskretizimi/dataset_with_target.csv`.
- **Output-et e gjeneruara:**
    - `outlier_flags_dataset.csv`: Dataseti origjinal me kolona shtesë që tregojnë statusin e anomalive.

---

### **Si të ekzekutohet?**
```bash
python step_4.py
```
