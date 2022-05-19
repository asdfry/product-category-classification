import os
import sys
import time

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"


# setup the distributed backend for managing the distributed training
torch.distributed.init_process_group("nccl")
rank = torch.distributed.get_rank()

if rank == 0:
    for i in range(torch.cuda.device_count()):
        print(f"> GPU {i}:", torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i))

device = torch.device("cuda", rank)

# # # 중 요 # # #
torch.cuda.set_device(device)

# print("device:", device)

if not torch.cuda.is_available():
    print("> cuda is not available")
    sys.exit(1)


class CustomeDataset(Dataset):
    def __init__(self, csv_path, is_train):
        df = pd.read_csv(csv_path)
        name_lst = df["name"].to_list()
        category_lst = df["category_id"].to_list()

        if is_train:
            self.label_lst = df["category_id"].unique().tolist()

        del [df]

        self.pair_lst = list(zip(name_lst, category_lst))

    def __len__(self):
        return len(self.pair_lst)

    def __getitem__(self, idx):
        text = self.pair_lst[idx][0]
        ids = vocab.EncodeAsIds(text)
        text = torch.LongTensor(ids)

        label = self.pair_lst[idx][1]
        idx = train_data.label_lst.index(label)
        label = torch.LongTensor([idx])

        return text, label


def custom_collate(batch):
    train_tensor_lst = []
    label_tensor_lst = []

    for text, label in batch:
        train_tensor_lst.append(text)
        label_tensor_lst.append(label)

    return pad_sequence(train_tensor_lst, batch_first=True).cuda(), torch.stack(label_tensor_lst).cuda()


def create_data_set_loader(csv_path, is_train):
    tv = "train" if is_train else "valid"
    if rank == 0:
        print(f"> creating {tv} data set & data loader... [{csv_path}]")

    data_set = CustomeDataset(csv_path, is_train)

    # shuffle 옵션은 sampler에게 줘야함, dataloader 에게 주면 에러
    dist_sampler = DistributedSampler(data_set, shuffle=True)

    # CNN 들어가기 전 reshape 과정에서 에러가 발생하므로 drop_last를 켜줌
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=custom_collate,
        pin_memory=False,
        num_workers=0,
        sampler=dist_sampler,
        drop_last=True,
    )

    # data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    if rank == 0:
        print(f"> {tv} data set size:", len(data_set))

    return data_set, data_loader


class Net(nn.Module):
    def __init__(self, vocab_size, label_size, embedding_dim, hidden_size, num_layers=1, dropout=0, bidirectional=False):
        super(Net, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.cnn_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim // 2, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim // 2, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_4 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_5 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim // 2, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_layer_6 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim // 2, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.lstm_layer = nn.LSTM(
            embedding_dim // 2,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, label_size)  # 입출력은 라벨 크기만큼 가져야함

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
        # 크기 변화: (배치 크기, 은닉층 크기) => (배치 크기, 라벨 크기)
        output = self.linear(output[:, -1, :])

        return output


def train(checkpoint=0):
    global total_time

    # tensor board writer 생성
    if rank == 0:
        writer_train = SummaryWriter(f"runs/{train_name}/{model_name}")
        writer_valid = SummaryWriter(f"runs/{valid_name}/{model_name}")

    # --------------------- 학습 --------------------- #
    if rank == 0:
        print(f"> epochs: {epochs}, checkpoint: {checkpoint}")
        print("> training model...")

    for epoch in range(0, epochs + 1):
        last_time = time.perf_counter()
        # 학습
        model.train()
        total_loss_train = 0
        for X, Y in iter(train_loader):
            # 기울기 초기화
            optimizer.zero_grad()
            # 순방향 전파
            output = model(X)
            # 손실값 계산 (cross entropy는 소프트맥스 함수 포함이며 실제 입력값은 원-핫 인코딩이 필요 없음)
            loss = F.cross_entropy(output, Y.squeeze(1))
            # 손실값 덧셈
            total_loss_train += loss.item()

            if not epoch == 0:
                # 역방향 전파
                loss.backward()
                # 매개변수 업데이트
                optimizer.step()

        # 검증
        model.eval()
        total_loss_valid = 0
        with torch.no_grad():
            for X, Y in iter(valid_loader):
                # 순방향 전파
                output = model(X)
                if output is None:
                    print("> Error occurred while training")
                    return
                # 손실값 계산 (cross entropy는 소프트맥스 함수 포함이며 실제 입력값은 원-핫 인코딩이 필요 없음)
                loss = F.cross_entropy(output, Y.squeeze(1))
                # 손실값 덧셈
                total_loss_valid += loss.item()

        loss_train = total_loss_train / len(train_loader.dataset)
        loss_valid = total_loss_valid / len(valid_loader.dataset)

        if rank == 0:
            writer_train.add_scalar("loss/train", loss_train, epoch)
            writer_valid.add_scalar("loss/valid", loss_valid, epoch)
            writer_train.add_scalar("loss/total", loss_train, epoch)
            writer_valid.add_scalar("loss/total", loss_valid, epoch)

        time_per_epoch = int(time.perf_counter() - last_time)
        total_time += time_per_epoch
        if rank == 0:
            print(
                f"[{epoch} / {epochs} epoch] train loss: {loss_train:.7f}, valid loss: {loss_valid:.7f}, time: {time_per_epoch//60}m {time_per_epoch%60}s"
            )

        if rank == 0 and checkpoint != 0 and epoch != 0 and epoch % checkpoint == 0:
            done(epoch)

    if rank == 0:
        writer_train.flush()
        writer_valid.flush()
        writer_train.close()
        writer_valid.close()

    if rank == 0 and checkpoint == 0:
        done(epoch)


def done(epoch):
    # 소요 시간 출력
    hour = total_time // 3600
    min = total_time % 3600 // 60
    sec = total_time % 3600 % 60
    print(f"> train done! total time: {hour}h {min}m {sec}s")

    # 라벨 리스트 저장
    save_path = f"../model/label_list/{train_name}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for label in train_data.label_lst:
            f.write(f"{label}\n")
    print(f"> label list saved at {save_path}")

    # 모델 저장
    model_name_epoch = f"{model_name}_e-{epoch}"
    save_path = f"../model/cnn-lstm/{train_name}/{model_name_epoch}.model"
    torch.save(model.module.state_dict(), save_path)
    print(f"> model saved at {save_path}")


if __name__ == "__main__":

    # 랜덤 시드 설정
    torch.manual_seed(777)

    batch_size = 128
    if rank == 0:
        print("> batch size:", batch_size)

    # 학습하기 전 설정 #
    prefix = "d3_"
    vocab_size = 29000
    character_coverage = 0.995
    # 학습하기 전 설정 #

    # 사전 불러오기
    vocab_name = (
        f"spm_{character_coverage}_{vocab_size}"
        if prefix == "d3_"
        else f"{prefix}spm_{character_coverage}_{vocab_size}"
    )
    if rank == 0:
        print(f"> loading vocab... [{vocab_name}.model]")
    vocab = spm.SentencePieceProcessor()
    vocab.load(f"../model/sentence_piece/{vocab_name}.model")

    # 훈련 및 검증 데이터 생성
    train_name = prefix + "train"
    valid_name = train_name.replace("train", "valid")
    train_data, train_loader = create_data_set_loader(f"../data/splited_data/{train_name}.csv", True)
    valid_data, valid_loader = create_data_set_loader(f"../data/splited_data/{valid_name}.csv", False)

    # 하이퍼 파라미터
    input_size = 1024  # 임베딩 시킬 차원의 크기 및 CNN 입력 차원의 크기
    hidden_size = 512  # LSTM 은닉층 크기
    num_layers = 4  # LSTM 층 개수
    dropout = 0.3
    bidirectional = True  # bidirectional 사용 플래그
    epochs = 36
    checkpoint = 3

    # 모델 생성
    if rank == 0:
        print("> creating model...")

    model = Net(
        len(vocab),  # 사전 크기는 임베딩에 사용되며 스폐셜 토큰도 포함됨
        len(train_data.label_lst),  # 라벨 크기는 최종 출력에 사용됨
        input_size,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
    ).cuda()

    # to.(device)를 통해 gpu에 복사하고 ddp 처리를 해야함
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # 옵티마이저 정의
    optimizer = optim.Adam(params=model.parameters())

    # 학습시 갱신되는 모델 파라미터 출력
    if rank == 0:
        print(model)

    # 모델 이름 결정 (텐서보드, 라벨 리스트 저장, 모델 저장에 사용)
    model_name = f"i-{input_size}_h-{hidden_size}_d-{dropout}_n-{num_layers}"
    model_name += "_bi" if bidirectional else ""

    # 학습
    total_time = 0
    train(checkpoint)
