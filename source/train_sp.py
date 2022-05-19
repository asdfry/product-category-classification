import pandas as pd
import sentencepiece as spm

# Read csv
prefix = ""
df = pd.read_csv(f"../data/splited_data/{prefix}train.csv")
print("> train size:", len(df))
print("> exist nan:", df.isna().values.any())

# Make train.txt
text_path = f"../data/text_for_sp/{prefix}train.txt"
f = open(text_path, "w")
f.close()
name_lst = df["name"].to_list()
del [df]
with open(text_path, "a") as f:
    for name in name_lst:
        f.write(name + "\n")
print("> create", text_path)

# Train
vocab_size = 32000 # vocab 사이즈
character_coverage = 1 # 몇 %의 데이터를 커버할것인가 (train 때 OOV를 없애려면 1)
model_path = f"../model/sentence_piece/{prefix}spm_{character_coverage}_{vocab_size}" # 저장될 tokenizer 모델에 붙는 이름
model_type ='bpe' # Choose from unigram (default), bpe, char, or word

# bos_id, eos_id, unk_id, pad_id
spm.SentencePieceTrainer.train(
    input=text_path,
    model_prefix=model_path,
    model_type=model_type,
    vocab_size=vocab_size,
    character_coverage=character_coverage,
    eos_id=3,
    bos_id=2,
    unk_id=1,
    pad_id=0
    )

# Test
vocab = spm.SentencePieceProcessor()
vocab.load(model_path + ".model")
print("> eos:", vocab.eos_id(), vocab.IdToPiece(vocab.eos_id()))
print("> bos:", vocab.bos_id(), vocab.IdToPiece(vocab.bos_id()))
print("> unk:", vocab.unk_id(), vocab.IdToPiece(vocab.unk_id()))
print("> pad:", vocab.pad_id(), vocab.IdToPiece(vocab.pad_id()))
print("> vocab size:", len(vocab))

line = "갤럭시S20 스탠딩 풀커버 슬림 가죽케이스 P012"
ids = vocab.EncodeAsIds(line)
pieces = [vocab.IdToPiece(id) for id in ids]
print(line)
print(ids)
print(pieces)
