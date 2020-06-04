# NER
A simple NER tagger.
## Running
You can run the whole process very easily. Take the conll2003 corpus for example,

### Step 1: Clean the pretraind embeddings.
```bash
./scripts/conll2003.sh clean
```

### Step 2: Convert to bieos tagging scheme.
```bash
./scripts/conll2003.sh bieos
```

### Step 3: Prepare data.
```bash
./scripts/conll2003.sh preprocess
```

### Step 4: Use elmo.
```bash
./scripts/conll2003.sh elmo
```
### Step 5: Train (with elmo).
```bash
./scripts/conll2003.sh train(train_elmo)
```

## Performance Comparision

-|Dataset|F1
:-:|:-:|:-:
Original|conll2003|91.62%
This Implementation (BiLSTM-CRF)|conll2003|91.19%
