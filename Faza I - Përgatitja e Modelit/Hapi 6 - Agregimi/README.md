# Hapi 6: Agregimi i të Dhënave (Data Aggregation)

### **Qëllimi**
Ky hap realizon një analizë makro përmes agregimit të datasetit në nivel **Mujor**, duke ndihmuar në identifikimin e trendeve sezonale dhe reduktimin e zhurmës që vjen nga lëkundjet ditore.

---

### **Veprimet Agregative Analitike Mujore:**
- Në këtë stad, të dhënat pasurohen me llogaritje mujore të intensitetit mesatar dhe total të karbonit.
- Kjo ndihmon në kuptimin e trendeve vjetore (p.sh. pse dimri ka intensitet më të lartë karboni në Kosovë).

---

### **Struktura e Skedarëve**
- **Input:** Dataseti me targetin `../Hapi 3 - Diskretizimi/dataset_with_target.csv`.
- **Output-et e gjeneruara:**
    - `aggregated_dataset.csv`: Dataseti i agreguar ku secili rresht përfaqëson një muaj.

---

### **Si të ekzekutohet?**
```bash
python step_6.py
```
