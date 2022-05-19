import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Lock
from pynori.korean_analyzer import KoreanAnalyzer
from tqdm import tqdm


tag_lst = [
    "NNG",
    "NNP",
    "NNB",
    "NNBC",
    "NR",
    "NP",
    "VV",
    "VA",
    "VX",
    "VCP",
    "VCN",
    "MM",
    "MAG",
    "MAJ",
    "IC",
    "JKS",
    "JKC",
    "JKG",
    "JKO",
    "JKB",
    "JKV",
    "JKQ",
    "JX",
    "JC",
    "EP",
    "EF",
    "EC",
    "ETN",
    "ETM",
    "XPN",
    "XSN",
    "XSV",
    "XSA",
    "XR",
    "SF",
    "SE",
    "SSO",
    "SSC",
    "SC",
    "SY",
    "SL",
    "SH",
    "SN",
]


def analysis_nori(pair_lst):
    for pair in pair_lst:
        ta = " ".join(nori.do_analysis(pair[0])["termAtt"])
        lock.acquire()
        name_lst_nori.append(ta)
        category_lst_nori.append(pair[1])
        lock.release()


if __name__ == "__main__":
    # ====================== pynori ======================
    # decompound_mode / infl_decompound_mode - 복합명사 / 굴절어 처리 방식 결정
    ## 'MIXED': 원형과 서브단어 모두 출력
    ## 'DISCARD': 서브단어만 출력
    ## 'NONE': 원형만 출력
    # discard_punctuation - 구두점 제거 여부
    # output_unknown_unigrams - 언논 단어를 음절 단위로 쪼갬 여부
    # pos_filter - POS 필터 실행 여부
    # stop_tags - 필터링되는 POS 태그 리스트 (pos_filter=True일 때만 활성)
    # synonym_filter - 동의어 필터 실행 여부
    # mode_synonym - 동의어 처리 모드 (NORM or EXTENSION) (synonym_filter=True일 때만 활성)
    # ==================================================================

    # NNG: 일반 명사, NNP: 고유 명사, SL: 외국어, SN: 숫자
    non_stop_tags = ["NNG", "NNP", "SL"]

    for tag in non_stop_tags:
        tag_lst.remove(tag)

    print("> creating nori analyzer...")

    nori = KoreanAnalyzer(
        decompound_mode="NONE",  # DISCARD or MIXED or NONE
        infl_decompound_mode="NONE",  # DISCARD or MIXED or NONE
        discard_punctuation=True,
        output_unknown_unigrams=False,
        pos_filter=False,
        stop_tags= tag_lst,  # (post filter가 True일 경우 활성)
        synonym_filter=False,
        # mode_synonym=,  # NORM or EXTENSION (synonym filter가 True일 경우 활성)
    )

    print("> reading csv to dataframe...")
    csv_name = "goods"
    df = pd.read_csv(f"../data/{csv_name}.csv")
    name_lst = df["name"].to_list()
    category_lst = df["category_id"].to_list()
    del [df]

    pair_lst = list(zip(name_lst, category_lst))
    name_lst_nori = Manager().list()
    category_lst_nori = Manager().list()
    begin_time = time.time()

    lock = Lock()
    with ProcessPoolExecutor(max_workers=32) as executor:
        step = 200000
        begin_idx = 0
        end_idx = step
        p_num = 1
        while True:
            if end_idx > len(pair_lst):
                future = executor.submit(analysis_nori, pair_lst[begin_idx:])
                print(f"> [p{p_num}] start analysis df[{begin_idx}:{len(pair_lst)}]")
                break
            else:
                future = executor.submit(analysis_nori, pair_lst[begin_idx:end_idx])
                print(f"> [p{p_num}] start analysis df[{begin_idx}:{end_idx}]")
                p_num += 1
            begin_idx += step
            end_idx += step

    end_time = int(time.time() - begin_time)
    sec = end_time % 60
    min = end_time // 60
    hour = min // 60
    min = min - 60 * hour
    print(f"> tokenize done! total time: {hour}h {min}m {sec}s")

    print("> creating new dataframe...")
    df = pd.DataFrame({"name": list(name_lst_nori), "category_id": list(category_lst_nori)})
    print("> dataframe size:", len(df))

    print("> drop nan...")
    df.dropna(axis=0, inplace=True)
    print("> dataframe size:", len(df))

    print("> drop duplicates...")
    df.drop_duplicates("name", inplace=True)
    print("> dataframe size:", len(df))

    save_path = f"../data/{csv_name}_nori.csv"
    df.to_csv(save_path, index=False)
    print("> save at", save_path)
