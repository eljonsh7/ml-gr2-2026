# Hapi 3: Inxhinieria e Veçorive dhe Definimi i Targetit

### **Qëllimi**
Në këtë hap, ne transformojmë një problem regresioni në **Klasifikim Multiklasë** dhe pasurojmë datasetin me variabla të reja (features) që ndihmojnë modelin të kuptojë kontekstin kohor dhe ambiental.

---

### **Veprimet kryesore:**
1.  **Redundant Column Removal:** Heqim kolonat `Country`, `Zone name`, dhe `Zone id` pasi ato janë konstante për të gjithë datasetin dhe nuk ofrojmë vlerë për modelin.
2.  **Temporal Features:** Nxjerrim informacione nga kolona kohore dhe fshijmë kolonën `Datetime` origjinale:
    - `day`, `month`, `year`: Vendosen si tri kolonat e para të datasetit.
    - `hour`, `day_of_week`, `is_weekend`: Karakteristika shtesë për kapjen e sezonalitetit.
3.  **Derived Features (Gap Analysis):**
    - Llogarisim hendekun midis intensitetit direkt dhe ciklit të jetës së karbonit.
    - Llogarisim peshën e energjisë së rinovueshme brenda energjisë pa-karbon.
4.  **Target Creation (Diskretizimi):**
    - Kolona target `Carbon intensity...` ndahet në 3 kategori (Low, Medium, High).
    - Kjo garanton një shpërndarje simetrike: secila klasë ka saktësisht ~33.3% të të dhënave.

---

### **Struktura e Skedarëve**
- **Input:** Dataseti i pastruar `../Hapi 2/cleaned_dataset.csv`.
- **Output-et e gjeneruara:**
    - `dataset_with_target.csv`: Dataseti final i pasuruar (30+ kolona).
    - `class_distribution.csv`: Statistikat e shpërndarjes së klasave.

---

### **Si të ekzekutohet?**
```bash
python step_3.py
```

