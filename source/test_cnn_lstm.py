import argparse
import json

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from tqdm import tqdm


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_num", help="gpu number for use [Default=0]", default=0)
    parser.add_argument(
        "-i", "--input_size", help="LSTM input size (embedding dimension size) [Required]", required=True
    )
    parser.add_argument("-hs", "--hidden_size", help="LSTM hidden size [Required]", required=True)
    parser.add_argument("-n", "--num_layers", help="number of layers [Default=1]", default=1)
    parser.add_argument("-d", "--dropout", help="dropout [Default=0]", default=0)
    parser.add_argument(
        "-b", "--bidirectional", help="bidirectional flag [Default=F]", default=False, action="store_true"
    )
    parser.add_argument("-e", "--epochs", help="epochs [Required]", required=True)
    args = parser.parse_args()

    # 현재 Setup 되어있는 device 확인
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"> available devices: {torch.cuda.device_count()}")
    print(f"> current cuda device: {torch.cuda.current_device()}")
    print(f"> device name: {torch.cuda.get_device_name(device)}")

    # 하이퍼 파라미터
    input_size = int(args.input_size)  # 임베딩 시킬 차원의 크기 및 RNN 층 입력 차원의 크기
    hidden_size = int(args.hidden_size)  # LSTM 은닉층 크기
    num_layers = int(args.num_layers)  # LSTM 층 개수
    dropout = int(args.dropout) if float(args.dropout) == 0 else float(args.dropout)
    bidirectional = args.bidirectional  # bidirectional 사용 플래그
    epochs = int(args.epochs)

    # 모델 이름 결정 (사전, 라벨 리스트, 모델을 불러오는데 사용)
    model_name = f"i-{input_size}_h-{hidden_size}_d-{dropout}_n-{num_layers}"
    model_name += "_bi" if bidirectional else ""

    # 테스트하기 전 설정 #
    prefix = "d3_"
    vocab_size = 29000
    character_coverage = 0.995
    # 테스트하기 전 설정 #

    # 사전 불러오기
    vocab_name = (
        f"spm_{character_coverage}_{vocab_size}"
        if prefix == "d3_"
        else f"{prefix}spm_{character_coverage}_{vocab_size}"
    )
    print(f"> loading vocab... [{vocab_name}.model]")
    vocab = spm.SentencePieceProcessor()
    vocab.Load(f"../model/sentence_piece/{vocab_name}.model")

    # 라벨 리스트 불러오기
    train_name = prefix + "train"
    print(f"> loading label list... [{train_name}.txt]")
    with open(f"../model/label_list/{train_name}.txt", "r") as f:
        lines = f.readlines()
        label_lst = [int(line.strip()) for line in lines]
    print(f"> vocab size: {len(vocab)}, label size: {len(label_lst)}")

    # 모델 불러오기
    model_name = f"{model_name}_e-{epochs}"
    print(f"> loading model... [{train_name}/{model_name}.model]")
    model = Net(len(vocab), len(label_lst), input_size, hidden_size, num_layers, dropout, bidirectional).cuda()

    # 불러오는 네트워크가 GPU tensor를 포함하는 경우 default로 0번 gpu에 로드시킴
    # 이를 방지하고자 map_location 옵션 사용, device = "cuda:gpu_num"
    model.load_state_dict(torch.load(f"../model/cnn-lstm/{train_name}/{model_name}.model", map_location=device))
    # model.load_state_dict(torch.load(f"../model/lstm/{model_name}.model"))
    model.eval()
    print(model)

    # 필요한 파일 불러와 데이터프레임으로 변환
    test_nmae = train_name.replace("train", "test")
    print(f"> loading test data... [{test_nmae}.csv]")
    df_category = pd.read_excel("../data/category.xlsx")
    df_test = pd.read_csv(f"../data/splited_data/{test_nmae}.csv")
    name_lst = df_test["name"].to_list()
    category_lst = df_test["category_id"].to_list()
    del [df_test]

    print("> testing model...")
    total_correct = [0 for _ in range(4)]
    topn_lst = [1, 3, 5, 10]
    wrong_dict = {}
    done_flag = True

    with torch.no_grad():
        for idx, corpus in enumerate(tqdm(name_lst)):
            # sentence piece를 이용하여 corpus -> ids (IntTensor)
            # GPU 사용시 torch.cuda.IntTensor
            ids = torch.cuda.IntTensor(vocab.EncodeAsIds(corpus))

            # 입력을 2차원(배치 크기, 시퀀스 길이)으로 변환
            ids = ids.view(1, len(ids))

            # 출력
            test_y = model(ids)
            if test_y is None:
                done_flag = False
                break

            # 마지막 차원을 대상으로 argmax 후 label_lst에서 해당하는 인덱스의 값(category id) 추출
            # inference_id = label_lst[test_y.argmax(dim=-1).item()]

            # 마지막 차원을 대상으로 topk 및 리스트로 변환 후 label_lst에서 해당하는 인덱스의 값(category id) 추출
            inference_id_lst = test_y.topk(10, dim=-1).indices[0].tolist()
            inference_id_lst = [label_lst[i] for i in inference_id_lst]

            # 추론한 카테고리 id를 이용하여 카테고리 이름 추출
            # row = df_category[df_category["code"] == inference_id]
            # inference_name = row["depth_4"].item()

            # 정답 카테고리 id
            correct_id = category_lst[idx]
            # row = df_category[df_category["code"] == correct_id]

            # 정답 상품명
            # correct_name = row["depth_4"].item()

            for idx, topn in enumerate(topn_lst):
                if correct_id in inference_id_lst[:topn]:
                    total_correct[idx] += 1
                # top-10 차례에 오답 기록
                elif idx == 3:
                    if correct_id in wrong_dict:
                        wrong_dict[correct_id] += 1
                    else:
                        wrong_dict[correct_id] = 1

    if done_flag:
        with open("../result.txt", "a") as f:
            f.write(f"\n{model_name}\n")
            for idx, topn in enumerate(topn_lst):
                correct = total_correct[idx]
                sentence = f"> top-{topn}: {correct} / {len(name_lst)} ({(correct / len(name_lst) * 100):.3f} %)"
                print(sentence)
                f.write(f"{sentence}\n")

        json_path = f"../wrong_json/{train_name}/{model_name}.json"
        with open(json_path, "w") as f:
            json.dump(wrong_dict, f, indent="\t")
            print(f"> wrong json saved at {json_path}")
