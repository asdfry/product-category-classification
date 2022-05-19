import json
import time

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.params import Form
from pynori.korean_analyzer import KoreanAnalyzer

app = FastAPI()
cnt = 1


class Net(nn.Module):
    def __init__(self, vocab_size, label_size, input_size, hidden_size, num_layers=1, dropout=0, bidirectional=False):
        super(Net, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size, padding_idx=0)
        self.cnn_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size // 2, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size // 2, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_4 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_5 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size // 2, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_6 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size // 2, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.lstm_layer = nn.LSTM(
            input_size // 2,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, label_size)  # 출력은 라벨 크기만큼 가져야함

    def forward(self, x):
        # Embedding
        # 크기 변화: (배치 크기, 시퀀스 길이) => (배치 크기, 시퀀스 길이, 임베딩 차원)
        output = self.embedding_layer(x)

        # 크기 변화: (배치 크기, 시퀀스 길이, 임베딩 차원) => (배치 크기, 임베딩 차원, 시퀀스 길이)
        output = output.transpose(1, 2)

        # CNN + concat (CV = convolution, MP = max pooling, CC = concat)
        # 크기 변화: (배치 크기, 임베딩 차원, 시퀀스 길이) =CV=> (배치 크기, 임베딩 차원, 시퀀스 길이 + 0 ~ 1) =MP=> (배치 크기, 임베딩 차원, 1) =CC=> (배치 크기, 임베딩 차원, 6)
        output = torch.cat(
            [
                self.cnn_layer_1(output),
                self.cnn_layer_2(output),
                self.cnn_layer_3(output),
                self.cnn_layer_4(output),
                self.cnn_layer_5(output),
                self.cnn_layer_6(output),
            ],
            dim=-1,
        )

        # 크기 변화: (배치 크기, 임베딩 차원, 6) => (배치 크기, 6, 임베딩 차원)
        output = output.transpose(1, 2)

        # LSTM
        # 크기 변화: (배치 크기, 6, 임베딩 차원) => output (배치 크기, 6, 은닉층 크기), hidden (층의 개수, 배치 크기, 은닉층 크기)
        output, hidden = self.lstm_layer(output)

        # Linear (마지막 time step의 hidden state 값만 사용)
        # 크기 변화: (배치 크기, 6, 은닉층 크기) => (배치 크기, 6, 라벨 크기)
        output = self.linear(output[:, -1, :])

        return output


def load_analyzer():
    start_time = time.perf_counter()
    print(f"> loading analyzer...", end=" ")
    nori = KoreanAnalyzer(
        decompound_mode="NONE",
        infl_decompound_mode="NONE",
        discard_punctuation=True,
        output_unknown_unigrams=False,
        pos_filter=False,
        synonym_filter=False,
    )
    print(f"[time: {(time.perf_counter() - start_time):.3f} sec]")

    return nori


def load_vocab():
    vocab_name = f"data/spm_{character_coverage}_{vocab_size}"
    print(f"> loading vocab... [{vocab_name}.model]")
    vocab = spm.SentencePieceProcessor()
    vocab.Load(f"{vocab_name}.model")

    return vocab


def load_label_lst():
    with open(f"data/label_list.txt", "r") as f:
        lines = f.readlines()
        label_lst = [int(line.strip()) for line in lines]

    return label_lst


def load_model(vocab_length, label_lst_length):
    # 하이퍼 파라미터
    input_size = 1024  # 임베딩 시킬 차원의 크기 및 RNN 층 입력 차원의 크기
    hidden_size = 512  # LSTM 은닉층 크기
    num_layers = 4  # LSTM 층 개수
    dropout = 0.3
    bidirectional = True  # bidirectional 사용 플래그
    packing = False  # packing 사용 플래그
    epochs = 30

    # 모델 이름 결정 (사전, 라벨 리스트, 모델을 불러오는데 사용)
    model_name = f"i-{input_size}_h-{hidden_size}_d-{dropout}_n-{num_layers}"
    model_name += "_bi" if bidirectional else ""

    # 모델 불러오기
    model_name = f"{model_name}_e-{epochs}"
    print(f"> loading model... [{model_name}.model]")
    model = Net(vocab_length, label_lst_length, input_size, hidden_size, num_layers, dropout, bidirectional)
    model.load_state_dict(torch.load(f"data/{model_name}.model", map_location="cpu"))
    model.eval()
    print(model)

    return model


def find_category_name(code):
    target_idx = df[df["code"] == code].index.values[0]
    large = None
    middle = None
    # small = None
    # 타겟 인덱스부터 역순으로 읽으면서 알맞은 depth 1 ~ 2을 추출
    for idx in range(target_idx, -1, -1):
        if large and middle:
            break
        series = df.loc[idx]
        d1 = series["depth_1"]
        d2 = series["depth_2"]
        # d3 = series["depth_3"]
        if not type(d1) is float:
            large = d1
        elif not type(d2) is float and middle is None:
            middle = d2
        # elif not type(d3) is float and small is None:
        #     small = d3

    return f"{large} > {middle} > {df.loc[target_idx]['depth_3']}"


def find_d4_list(code):
    target_idx = df[df["code"] == code].index.values[0]
    d4_lst = []
    # 타겟 인덱스 다음부터 읽으면서 하위 depth 4 추출
    for idx in range(target_idx + 1, len(df) - 1):
        series = df.loc[idx]
        if not type(series["depth_4"]) is float:
            d4_lst.append(series["depth_4"])
        else:
            break

    return ", ".join(d4_lst)


@app.post("/recommend")
def recommend(goods_name: str = Form(...), topk_num: int = Form(...)):
    global cnt
    start_time = time.perf_counter()
    print(f"\n> > > INFERENCE [top-{topk_num}] < < <")
    print(f"> goods name: {goods_name}, topk_num: {topk_num}")
    ta = " ".join(nori.do_analysis(goods_name)["termAtt"])
    ids = torch.IntTensor(vocab.EncodeAsIds(ta))
    if ids.shape[0] < 1:
        raise HTTPException(status_code=404, detail="tensor size is zero")
    ids = ids.reshape(1, len(ids))

    # 모델에 입력
    test_y = model(ids)
    test_y = nn.functional.softmax(test_y, dim=-1)

    # 상위 10개 값(확률) 리스트
    inference_value_lst = test_y.topk(topk_num, dim=-1).values[0].tolist()
    inference_value_lst = [round(i * 100, 4) for i in inference_value_lst]

    # 상위 10개 인덱스 리스트
    inference_idx_lst = test_y.topk(topk_num, dim=-1).indices[0].tolist()
    inference_id_lst = [label_lst[i] for i in inference_idx_lst]

    # 상위 10개 카테고리(d1 > d2 > d3) 리스트
    inference_name_lst = [find_category_name(id) for id in inference_id_lst]

    # 상위 10개 카테고리(d4) 리스트
    inference_d4_lst = [find_d4_list(id) for id in inference_id_lst]

    result_dict = {
        "goods_name": goods_name,
        "category_id": inference_id_lst,
        "depth_3": inference_name_lst,
        "depth_4": inference_d4_lst,
        "probability": inference_value_lst,
        "spend_time": round(time.perf_counter() - start_time, 3),
    }

    with open(f"sample_results/sample_result_{cnt}.json", "w") as f:
        cnt += 1
        json.dump(result_dict, f, ensure_ascii=False, indent="\t")

    return result_dict


if __name__ == "__main__":
    vocab_size = 29000
    character_coverage = 0.995

    nori = load_analyzer()
    vocab = load_vocab()
    label_lst = load_label_lst()
    model = load_model(len(vocab), len(label_lst))
    df = pd.read_excel("data/category.xlsx")

    uvicorn.run(app, host="0.0.0.0", port=8502)
