# Historical Newspaper Person Pointing

Repozitár obsahuje kód a pripravené dátové súbory k diplomovej práci zameranej na lokalizáciu fotografií osôb v digitalizovaných historických novinách. Úloha je postavená nad multimodálnym modelom Molmo. Model dostane obraz celej novinovej strany a textový dopyt, promt + meno osoby, a jeho výstupom je bod smerujúci na stred tváre hľadanej osoby alebo odpoveď `There are none.`, ak sa daná osoba na strane nenachádza.

## Inštalácia

Odporúčané je použiť samostatné virtuálne prostredie. Príklady nižšie predpokladajú spustenie z koreňového priečinka repozitára.

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Najprv je potrebná inštalácia `PyTorch` podľa cieľového prostredia. V experimentoch bola použitá zostava:

```text
torch==2.11.0
torchvision==0.26.0
```

Potom je potrebé nainštalovať závislosti projektu:

```bash
pip install -r requirements.txt
```

Dôležitá kompatibilitná poznámka: Molmo je citlivé na verziu knižnice `transformers`.  Je nutné použiť túto konkrétnu verziu:

```text
transformers==4.50.3
```

## Štruktúra repozitára

```text
HistoricalNewspaperPersonPointing/
├── README.md                          # prehľad projektu, inštalácia a štruktúra repozitára
├── requirements.txt                   # hlavné Python závislosti bez PyTorch/CUDA balíkov
├── datasets/
│   ├── digiknihovna_data/             # lokálne miesto pre obrazy celých novinových strán
│   └── peoplegators/                  # PeopleGator anotácie a pripravené Molmo JSONL súbory
├── experiments/
│   ├── evaluated_adapters/            # výsledky vyhodnotenia dotrénovaných adaptérov
│   ├── evaluated_pure_molmo/          # výsledky vyhodnotenia pôvodného modelu Molmo
│   └── trained_adapters/              # výstupy trénovania LoRA adaptérov a checkpointy
├── src/
│   ├── README.md                      # technický opis skriptov a príklady ich spustenia
│   ├── datasets/                      # príprava, vyvažovanie a samplovanie dátových sád
│   ├── eval/                          # vyhodnotenie pôvodného modelu a adaptérov
│   ├── statistics/                    # štatistiky, kontrolné výpisy a kvalitatívne obrázky
│   └── train/                         # trénovanie LoRA adaptérov pre Molmo
├── text/                              # text práce
└── video/                             # podklady k prezentačnému videu
```