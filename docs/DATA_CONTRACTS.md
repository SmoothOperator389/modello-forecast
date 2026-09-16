# Contratti dati — allineati alla riunione 11 ago 2026 + schema team

## Fonti (da riunione)

| Blocco | Fonte | Note |
|--------|--------|------|
| Vendite | **Valido** (DB, idealmente copia) | Fonte primaria |
| Strutturali PDV (mq, transit…) | **File interni team** | Non sempre in Valido |
| Calendario / meteo | File + API (evoluzione) | Non in scope riunione, utili dopo |

**Governance:** strumento di **supporto interno** al team (non autonomia a Produzione). Output = **domanda attesa** per guidare riassortimento / movimentazione. Controllo esclusivo del team.

---

## 1) Negozio (anagrafica / master PDV)

Campi concordati in riunione:

| Campo | Tipo tipico | Uso nel modello |
|-------|-------------|-----------------|
| `country` | testo / codice | embedding, stagionalità per paese |
| `commerciale_riferimento` | testo | embedding / cluster umano |
| `mq_espositivi` | numerico | potenziale vendita |
| `mq_stock` | numerico | capacità magazzino negozio |
| `mq_totali` | numerico | = espositivi + stock (derivato ok) |
| `numero_fatturazioni` | numerico | intensità traffico / attività |
| `transit_time` | numerico (giorni) | layer operativo riassortimento |

**CSV manuale proposto:** `data/manual/pdv_master.csv`

```csv
pdv_id,country,commerciale_riferimento,mq_espositivi,mq_stock,mq_totali,numero_fatturazioni,transit_time,data_apertura,pdv_strategico,city
MIL_01,IT,Rossi,120,40,160,320,5,2024-03-01,1,Milan
ROM_NEW,IT,Bianchi,90,30,120,40,5,2026-07-20,0,Rome
```

> `pdv_id`, `data_apertura`, `city` non erano nella lista riunione ma servono per cold start PDV e meteo. Da concordare col team se aggiungerli.

---

## 2) Vendite / movimenti (fact Valido) — **data model ufficiale**

### File canonico (set 2026)

| Voce | Valore |
|------|--------|
| Path tipico | `/Users/mirkobergantino/Downloads/export24kk.csv` (puntatore in `data/raw/export24kk.path`) |
| Formato | **nessuna intestazione**; separatore **`;`**; fine riga CRLF |
| Righe | ~**24,2M** |
| Periodo | **2024-01-01 → 2026-08-18** |
| PDV | ~**165** (`IDAnagrafica`) |
| SKU | ~**27k** (`IDArticolo`) |
| Paesi (`iso3`) | AND, ARE, AUT, BEL, CHE, CHN, DEU, ESP, FRA, GBR, IRL, ITA, KOR, LBN, LUX, NLD, NOR, SGP, SWE, THA, USA |

**Schema fisso (10 campi, ordine fisso):**

| # | Campo export | → modello | Note |
|---|--------------|-----------|------|
| 1 | `IDAnagrafica` | `pdv_id` | es. `SCOAN`, `OSIT2`, `T84__` |
| 2 | `iso3` | `country` | es. `ITA`, `NLD` |
| 3 | `data` | `data` | `YYYY-MM-DD` |
| 4 | `IDArticolo` | `sku_id` | es. `0901UL6036076` |
| 5 | `tipodocumento` | filtro | vedi tabella sotto |
| 6 | `DescrizioneSuStampa` | desc. doc | solo lettura / QA |
| 7 | `Qta` | qty movimento | vendita 100 di solito negativa |
| 8 | `PrezzoPubblico` | `prezzo_pubblico` / listino | 0 su carichi |
| 9 | `PrezzoPubblicoReale` | `pagato_effettivo` | può avere molti decimali float |
| 10 | `giacenza` | `giacenza_qty` | livello dopo movimento |

Questo export **è** il data model transazionale. Non contiene anagrafica prodotto (modello/tessuto/colore/…) né master negozio (mq, city, transit): restano file team separati.

Esempio riga:

```text
SCOAN;ITA;2025-05-15;0901UL6036076;100;Vendita al pubblico;-1;79.0;79.0;2
```

### Regola `tipodocumento`

| Codice | Descrizione (export) | Domanda (target) | Giacenza |
|--------|----------------------|------------------|----------|
| **`100`** | Vendita al pubblico (~14,7M) | **Sì** — unica fonte; qty = `abs(Qta)` | Scarico |
| **`47`** | Fattura / Invoice (~5,4M) | No | Carico |
| **`51`** | Fattura / Invoice (Women's clothing) (~0,8M) | No | **Trattare come carico** (variante 47) salvo diversa indicazione team |
| **`20`** | Azzeramento magazzino (~0,6M) | No | Reset stock |
| Altri (`14` inventario, `102` reso, `40` spostamento, DDT…) | Movimenti | No | Non nel target vendite; stock: preferire colonna `giacenza` / casi noti |

**Domanda / training:** solo `tipodocumento == 100`.

**Giacenza operativa:** almeno **100 + 47 + 51 + 20**, oppure `MAX(giacenza)` giornaliero per PDV×SKU sulle righe di movimento.

- `100`: `Qta` negativa → scarico vendita (~0,13% con prezzo 0 — da monitorare)
- `47` / `51`: carico (prezzo tipicamente 0)
- `20`: azzeramento (segno variabile)

Campi concordati (anagrafica prodotto — **fuori** da questo CSV, da join separato):

| Campo | Tipo tipico | Uso nel modello |
|-------|-------------|-----------------|
| `referenza` | modello-tessuto-colore (o 3 campi) | identità SKU / embedding attributi |
| `campagna` | testo | stagione / collezione |
| `classe` | testo | gerarchia prodotto |
| `sclasse` | testo | gerarchia |
| `ssclasse` | testo | gerarchia |
| `tipologia` | testo | continuativo / stagionale / … |
| `qnt_venduta` | numerico | target / lag / run rate |
| `qnt_giacenza` | numerico | feature + operativo (**dato rumoroso**) |
| `prezzo_pubblico` | numerico | listino |
| `pagato_effettivo` | numerico | prezzo reale / sconto implicito |
| `giorno` | data o giorno-periodo | rolling, calendario |

**Schema fact consigliato (normalizzato):**

| data | country | pdv_id | sku_id | modello | tessuto | colore | campagna | classe | sclasse | ssclasse | tipologia | qnt_venduta | qnt_giacenza | prezzo_pubblico | pagato_effettivo |
|------|---------|--------|--------|---------|---------|--------|----------|--------|---------|----------|-----------|-------------|--------------|-----------------|------------------|

`referenza` può restare colonna unica *oppure* essere spezzata in modello/tessuto/colore (meglio per embedding).

### Query tipo Valido (nomi tabella da allineare)

```sql
SELECT
    v.giorno                    AS data,
    n.country                   AS country,
    n.codice_pdv                AS pdv_id,
    v.referenza                 AS referenza,
    v.campagna,
    v.classe,
    v.sclasse,
    v.ssclasse,
    v.tipologia,
    SUM(v.qnt_venduta)          AS qnt_venduta,
    MAX(v.qnt_giacenza)         AS qnt_giacenza,   -- verificare logica stock
    AVG(v.prezzo_pubblico)      AS prezzo_pubblico,
    AVG(v.pagato_effettivo)     AS pagato_effettivo
FROM /* vendite Valido */ v
JOIN /* negozi */ n ON /* chiavi reali */
WHERE v.giorno BETWEEN :start AND :end
GROUP BY 1,2,3,4,5,6,7,8,9;
```

---

## 3) Criticità dati (da riunione — da non dimenticare)

- **Giacenza** poco affidabile dove non c’è spunta all’arrivo.
- **Transit time** complesso → usarlo con prudenza nel layer operativo.
- Anche **+1%** vs processo attuale è progresso utile.
- Output modello = **domanda attesa**, non “vendita certa”.

---

## 4) Mapping → feature modello (NN)

| Feature modello | Deriva da |
|-----------------|-----------|
| Embedding country, commerciale | Negozio |
| mq_*, numero_fatturazioni, transit | Negozio |
| Embedding modello/tessuto/colore/campagna/classe… | Vendite / anagrafica |
| run_rate_14d, has_history, giorni_da_lancio | aggregazioni su `qnt_venduta` + data lancio |
| prezzo / sconto implicito | `prezzo_pubblico`, `pagato_effettivo` → `sconto_pct`, `is_in_sconto`, `sconto_impatto_stimato` |
| y (target rolling) | somma `qnt_venduta` orizzonte successivo |
| qty suggerita (post-hoc) | domanda attesa + `qnt_giacenza` + `transit_time` |

---

## 5) Checklist accesso (Mirko / Elisa)

1. Copia DB o vista read-only **Valido**
2. Conferma nomi tabelle/colonne = elenco sopra
3. File PDV team con i 7 campi negozio (+ `pdv_id`)
4. Periodo storico minimo per pilot (idealmente 2–3+ anni se possibile; almeno 12–18 mesi)
5. Country in scope prima simulazione (Valentina: altri paesi entro venerdì)

---

## Allineamento terminologico riunione

| Termine riunione | Come lo usiamo |
|------------------|----------------|
| Domanda attesa | Output forecast (guida riassortimento) |
| Rolling | Train/inferenza sempre da reali, non da forecast passati |
| Curve dolci / coeff dinamici | Pesi NN + score continuo (no soglie rigide) |
| Supporto interno | Review umana obbligatoria, no autonomia Produzione |
