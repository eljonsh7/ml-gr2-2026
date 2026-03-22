# Hapi 7: Dataseti Final i Inxhinieruar (Final Engineered Dataset)

### **Qëllimi**
Ky hap shërben si pikë përmbyllëse për inxhinierinë e veçorive, duke prodhuar datasetin e plotë dhe të pastër i cili përmban të gjitha variablat e nxjerra në hapat e mëparshëm.

---

### **Përmbajtja e Datasetit:**
Dataseti i gjeneruar këtu është "thesari" i pastër i Fazës I dhe përfshin:
1.  **Variablat Kohore:** `hour`, `is_weekend`, `month`, etj.
2.  **Variablat e Hendekut (Gap):** Diferencat midis burimeve të ndryshme të intensitetit.
3.  **Targetin e Klasifikuar:** Klasat `low`, `medium`, `high`.
4.  **Pastrimi i Outliers:** Ky dataset është filtruar plotësisht nga anomalitë (outliers) për të garantuar cilësi maksimale.

---

### **Struktura e Skedarëve**
- **Input:** Dataseti me targetin `../Hapi 3 - Diskretizimi/dataset_with_target.csv`.
- **Output-et e gjeneruara:**
    - `feature_engineered_dataset.csv`: Dataseti i plotë ditor gati për analizë statistikore ose vizualizim të jashtëm.

---

### **Si të ekzekutohet?**
```bash
python step_7.py
```
