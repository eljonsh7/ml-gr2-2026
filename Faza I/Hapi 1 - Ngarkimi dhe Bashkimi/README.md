# Hapi 1: Ngarkimi dhe Agregimi Fillestar i të Dhënave

### **Qëllimi**
Ky hap shërben për grumbullimin e të dhënave të papërpunuara orare nga burime të ndryshme vjetore dhe bashkimin e tyre në një strukturë të vetme analitike. Githashtu, kryhet agregimi ditor për të kaluar nga analiza mikroskopike (orare) në atë makroskopike (ditore).

---

### **Veprimet kryesore:**
1.  **Leximi i Skedarëve:** Ngarkohen 5 skedarë vjetorë orarë (`hourly-2021.csv` deri `hourly-2025.csv`).
2.  **Bashkimi (Merging):** Krijohet një matricë origjinale me ~43,824 rreshta dhe 12 kolona.
3.  **Agregimi Ditor:** Për të reduktuar zhurmën dhe për të lehtësuar trajnimin e modeleve, të dhënat agregohen në nivel ditor duke llogaritur mesataren matematikore për çdo ditë. Kjo redukton datasetin në **1,826 rreshta**.
4.  **Schema Audit:** Kryhet një kontroll automatik i tipave të të dhënave (int, float, object) dhe identifikohen kolonat me "High Cardinality" (kardinalitet të lartë).

---

### **Struktura e Skedarëve**
- **Input:** Skedarët origjinalë `../hourly-20*.csv` ose `../initial_dataset.csv`.
- **Output-et e gjeneruara:**
    - `merged_hourly_dataset.csv`: Të dhënat e bashkuara orare (para agregimit).
    - `merged_daily_dataset.csv`: Dataseti ditor (baza për vazhdimin e projektit).
    - `schema_audit.csv`: Raporti teknik i tipave të kolonave.

---

### **Si të ekzekutohet?**
Për të rifreskuar të dhënat e këtij hapi, ekzekutoni:
```bash
python step_1.py
```

