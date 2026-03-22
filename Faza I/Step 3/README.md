# Step 3: Class Definition & Imbalance Metrics

**Qëllimi:** Inovimi strategjik the krijimi automatizues i problemit "Multiclass Classification Problem" (klasifikimi ndër me shumë vlera) ndaj atij regresiv, për të përmbushur fjalën artistike kërkuar nga detyra master.

**Veprimet kryesore:**
- E nxorrem target (objektivin thelbësor vlerësues) e fshehur që është gjithandej thekse vizuale në gjetjet ambientale: `Carbon intensity gCO₂eq/kWh (direct)`. Ofertat e karbonit!
- Sidoqoftë karboni jep vetem numra precize numerikisht, andaj e diskretizuam klasifikisht vlerën ekzakte per të perçuar thellë tri vellime (Classes kategorialish thelbësore):
    1. Vëllimi *low* (ndotje më e paket e gjetur ose ditë vikendi prodhime rrymash pozitive).
    2. Vëllimi *medium*.
    3. Vëllimi *high* (ndotje kritike ekstreme e lartë përbërë një target mbikqyrus të përhapur rregullativ në prodhimet ndotesish).
- Përpunimi dhe definimi e nxjerr shpërndarjen e krijuar si super simetrike! Secila fushë përkufizohet si `low (609/33.35%)`, `high (609/33.35%)`, `medium (608/33.30%)`.
- Logjika garanton fuqishëm që asnjëra pakice ose miniority thellesisht kritike të mos paraqitet. Kjo zvogëlon gjasat defektivoz-strukturore artificiale.

**Rezultati kryesor:** Ndarja në 3 pole të ekuilibruara balancimi bën punën te detektohet në fazën e 2të i balancor trajnues ekzakt mjaft i qarte ekuilifikativ.
