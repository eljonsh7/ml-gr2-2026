# Step 4: Sampling & Balancing

**Qëllimi:** Mbrojtës absolut ndaj defekteve mbingarkuese vizualizuese the modeluese "Overfitting" përmes ndarjes teknologjike trajnuese the zbatimit gjenerativ opsional ekuilibrues.

**Veprimet kryesore zbatuese:**
- Sipas rezultateve jashtëzakonisht te mira shkencore të theksuara nga pika paraprake *Step 3*, gjetëm që shkalla jonë targetuese klase `low / medium / high` u lind e balancuar jashtë mase në nivel numerik (~33% propocionist).
- Me gjithë infrastrukturën e instaluar nga ne gjetkë thellë në platformë si për SMOTE ("Synthetic Minority Over-sampling Technique") dhe logjiken me shkëndijë komplekse ADASYN; ato morën paralajmërimin "Skipped (already balanced)". Nuk mund shtonim rekorde fiktiv të gënjeshtër energjetik kur kemi balans ideal! Përfitimi empirik nga këtu i mbron analizat Thella.
- Më pas vjen *Sistemi Ndarës Stratik*! Duke përdorur funksionin klasik `train_test_split(stratify=y, test_size=0.2)`, nxorrem modelin tonë 20% për Mbajtje Te Fshehtë Ekzekutive ("Holdout Model") dhe mbetja prej 1460 formuan Setin zyrtar ekskluzivist vlerësues modelor thella per trajnime kognitive.

**Rezultati zbulues i vërtetë:** Modeluar mbikëqyrje perfekte prej të gjenetizuarave numerika tërësisht trajnuese 1460 rows, mbajtur vizual të pastër.
