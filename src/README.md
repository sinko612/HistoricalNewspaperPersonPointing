# Zdrojové kódy

Adresár `src/` obsahuje skripty na prípravu dát, evaluáciu modelov, výpočet štatistík a tréning LoRA adaptérov pre model Molmo. Každý skript sa spúšťa z adresára, v ktorom je uložený.

## `src/datasets/`

### `balance_trainig_datasets.py`

Vyváži existujúci inštrukčný JSONL dataset podľa typu úlohy a typu odpovede. Príkaz nižšie vyvažuje úlohy `is_person_present` a `point_person_by_name`, zachová ostatné úlohy a odstráni duplicitné riadky.

```bash
cd src/datasets
python balance_trainig_datasets.py \
  --input ../../datasets/peoplegators/molmo_peoplegator_train.jsonl \
  --output ../../datasets/peoplegators/molmo_peoplegator_train_balanced.jsonl \
  --seed 42 \
  --dedupe \
  --mode downsample_majority \
  --balance-tasks is_person_present,point_person_by_name \
  --keep-other-tasks
```

### `build_peoplegator_name_tasks_dataset.py`

Vytvorí inštrukčný JSONL dataset iba pre menné úlohy `point_person_by_name` a `is_person_present`. Skript negeneruje úlohu `point_all_person_photos`.

```bash
cd src/datasets
python build_peoplegator_name_tasks_dataset.py \
  --peoplegator-jsonl ../../datasets/peoplegators/people_gator__corresponding_faces__2026-02-11.dev.jsonl \
  --project-root ../../datasets/digiknihovna_data \
  --out-train ../../datasets/peoplegators/molmo_peoplegator_name_train.jsonl \
  --out-val ../../datasets/peoplegators/molmo_peoplegator_name_val.jsonl \
  --tasks point_person_by_name,is_person_present \
  --negative-names-per-page-point 1 \
  --negative-names-per-page-present 1 \
  --val-ratio 0.1 \
  --seed 42 \
  --include-metadata
```

### `create_train_dataset.py`

Vytvorí hlavné inštrukčné JSONL súbory pre Molmo z PeopleGator anotácií. Dev anotácie slúžia ako zdroj pozitívnych menných väzieb, test anotácie a detekcie sa používajú ako pomocný kontext pre kontrolu identít, negatívny sampling a úlohu `point_all_person_photos`.

```bash
cd src/datasets
python create_train_dataset.py \
  --peoplegator-jsonl ../../datasets/peoplegators/people_gator__corresponding_faces__2026-02-11.dev.jsonl \
  --peoplegator-test-jsonl ../../datasets/peoplegators/people_gator__corresponding_faces__2026-02-11.test.jsonl \
  --all-detections-jsonl ../../datasets/peoplegators/people_gator__detections.jsonl \
  --project-root ../../datasets/digiknihovna_data \
  --out-train ../../datasets/peoplegators/molmo_peoplegator_train.jsonl \
  --out-val ../../datasets/peoplegators/molmo_peoplegator_val.jsonl \
  --out-stats ../../datasets/peoplegators/molmo_peoplegator_stats.json \
  --tasks point_person_by_name,is_person_present,point_all_person_photos \
  --negative-names-per-positive-point 3 \
  --negative-names-per-positive-present 3 \
  --val-ratio 0.1 \
  --seed 42 \
  --include-metadata \
  --all-person-split-policy val_if_any_val_identity \
  --max-point-all-detections 8
```

### `make_peoplegator_sampled.py`

Vytvorí samplovanú variantu existujúceho Molmo JSONL datasetu. Skript umožňuje odstrániť duplicitné riadky, vyradiť negatívne mená bez pozitívneho výskytu v splite, prípadne vynechať úlohu `point_all_person_photos`.

```bash
cd src/datasets
python make_peoplegator_sampled.py \
  --input ../../datasets/peoplegators/molmo_peoplegator_dev_train.jsonl \
  --output ../../datasets/peoplegators/molmo_peoplegator_dev_train_3sampled.jsonl \
  --seed 42 \
  --dedupe \
  --drop-neg-name-without-positive \
  --positive-ratio 2.0
```

### Notebooky

`export_data_peoplegator.ipynb`, `get_data.ipynb` a `get_data_with_confidence.ipynb` sú pomocné notebooky pre export a predspracovanie dát.

## `src/eval/`

### `evaluate_molmo_adapters.py`

Vyhodnotí dotrénovaný LoRA adaptér nad PeopleGator anotáciami. Parameter `--adapter-dir` ukazuje na adresár adaptéra alebo na konkrétny checkpoint.

```bash
cd src/eval
python evaluate_molmo_adapters.py \
  --model-id allenai/Molmo-7B-D-0924 \
  --adapter-dir ../../experiments/trained_adapters/molmo_adapter_run1/checkpoint-1250 \
  --annotations-jsonl ../../datasets/peoplegators/people_gator__corresponding_faces__2026-02-11.test.jsonl \
  --detections-jsonl ../../datasets/peoplegators/people_gator__detections.jsonl \
  --data-root ../../datasets/digiknihovna_data \
  --out-dir ../../experiments/evaluated_adapters/molmo_adapter_run1_checkpoint_1250 \
  --eval-mode all \
  --negative-names-per-page 1 \
  --max-pages 0 \
  --resize-long-side 512 \
  --max-crops 2 \
  --sequence-length 768 \
  --max-new-tokens 160 \
  --device-map auto \
  --save-failures \
  --max-failures 5
```

### `evaluate_molmo_pure.py`

Vyhodnotí pôvodný model `allenai/Molmo-7B-D-0924` bez LoRA adaptéra. Výstupy evaluácie sa ukladajú do adresára zadaného cez `--out-dir`.

```bash
cd src/eval
python evaluate_molmo_pure.py \
  --model-id allenai/Molmo-7B-D-0924 \
  --annotations-jsonl ../../datasets/peoplegators/people_gator__corresponding_faces__2026-02-11.test.jsonl \
  --detections-jsonl ../../datasets/peoplegators/people_gator__detections.jsonl \
  --data-root ../../datasets/digiknihovna_data \
  --out-dir ../../experiments/evaluated_pure_molmo/test_all \
  --eval-mode all \
  --negative-names-per-page 1 \
  --negative-random-seed 42 \
  --max-pages 0 \
  --resize-long-side 512 \
  --max-crops 2 \
  --sequence-length 768 \
  --device-map auto \
  --save-failures \
  --max-failures 5
```

## `src/statistics/`

### `check_molmo_dataset_stats.py`

Vypíše štatistiky tréningového a validačného Molmo JSONL datasetu vrátane rozpisu úloh a pomeru kladných a záporných vzoriek.

```bash
cd src/statistics
python check_molmo_dataset_stats.py \
  --train ../../datasets/peoplegators/molmo_peoplegator_train.jsonl \
  --val ../../datasets/peoplegators/molmo_peoplegator_val.jsonl
```

### `count_sampled_datasets.py`

Vypočíta porovnávaciu tabuľku pre základný a samplované tréningové datasety.

```bash
cd src/statistics
python count_sampled_datasets.py \
  ../../datasets/peoplegators/molmo_peoplegator_dev_train.jsonl \
  ../../datasets/peoplegators/molmo_peoplegator_dev_train_2sampled.jsonl \
  ../../datasets/peoplegators/molmo_peoplegator_dev_train_3sampled.jsonl
```

### `make_a12_figures.py`

Vytvorí kvalitatívne obrázky pre vybraný evaluačný beh A12. Skript vie čítať priamu cestu k `results.csv` alebo ZIP archív s evaluačnými výstupmi.

```bash
cd src/statistics
python make_a12_figures.py \
  --results-csv ../../experiments/evaluated_adapters/molmo_lora_run_checkpoint_1250/results.csv \
  --detections-jsonl ../../datasets/peoplegators/people_gator__detections.jsonl \
  --faces-jsonl ../../datasets/peoplegators/people_gator__corresponding_faces__2026-02-11.test.jsonl \
  --output-dir ../../experiments/a12_qualitative \
  --work-dir ../../experiments/a12_qualitative_tmp \
  --image-root ../../datasets/digiknihovna_data \
  --max-width 1800
```

### `verify_peoplegator_dev_test_stats.py`

Overí počty záznamov a prekryvy medzi PeopleGator dev a test anotáciami.

```bash
cd src/statistics
python verify_peoplegator_dev_test_stats.py \
  --dev ../../datasets/peoplegators/people_gator__corresponding_faces__2026-02-11.dev.jsonl \
  --test ../../datasets/peoplegators/people_gator__corresponding_faces__2026-02-11.test.jsonl
```

### `visualize_peoplegator_train_samples.py`

Vykreslí tréningové vzorky s cieľovými bodmi a detekčnými rámčekmi do výstupného adresára.

```bash
cd src/statistics
python visualize_peoplegator_train_samples.py \
  --train-jsonl ../../datasets/peoplegators/molmo_peoplegator_train.jsonl \
  --detections-jsonl ../../datasets/peoplegators/people_gator__detections.jsonl \
  --project-root ../.. \
  --out-dir ../../experiments/train_sample_visualizations \
  --samples-per-task 5 \
  --seed 42 \
  --include-none-responses \
  --draw-all-detections \
  --allow-missing-images
```

## `src/train/`

### `train_molmo_lora.py`

Dotrénuje LoRA adaptér nad modelom `allenai/Molmo-7B-D-0924`. Skript podporuje 4-bitové QLoRA načítanie, výber cieľových LoRA modulov, zmrazenie vizuálneho enkódera, manuálne rozdelenie blokov modelu medzi GPU a voliteľné logovanie tokenizačných štatistík.

```bash
cd src/train
mkdir -p ../../experiments/trained_adapters/molmo_adapter_run1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u train_molmo_lora.py \
  --model-id allenai/Molmo-7B-D-0924 \
  --train-jsonl ../../datasets/peoplegators/molmo_peoplegator_dev_train_3sampled.jsonl \
  --val-jsonl ../../datasets/peoplegators/molmo_peoplegator_dev_val.jsonl \
  --images-root ../.. \
  --output-dir ../../experiments/trained_adapters/molmo_adapter_run1 \
  --learning-rate 3e-5 \
  --weight-decay 0.0 \
  --num-train-epochs 2 \
  --warmup-ratio 0.05 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --logging-steps 5 \
  --save-steps 250 \
  --eval-steps 250 \
  --max-grad-norm 1.0 \
  --seed 42 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-mode text \
  --freeze-vision-encoder \
  --print-trainable-names \
  --debug-token-first-n 10 \
  --debug-token-log-steps 500 \
  --debug-token-log-file ../../experiments/trained_adapters/molmo_adapter_run1/token_debug.jsonl \
  --resize-long-side 1024 \
  --max-crops 8 \
  --sequence-length 1536 \
  --device-map block
```
