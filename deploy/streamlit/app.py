import datetime
import requests
import streamlit as st


def main():
    print(st.session_state)

    st.header("한메소프트 상품 카테고리 추천")

    # 라디오 버튼을 수평으로 표시
    st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)
    st.radio("", ("예시", "직접 입력"), key="radio")

    if st.session_state.radio == "예시":
        goods_name = st.selectbox("- 상품명을 선택해주세요.", goods_ex, key="input")

    elif st.session_state.radio == "직접 입력":
        goods_name = st.text_input("- 상품명을 입력해주세요.", key="input")
    st.text("")

    topk_num = st.slider(label="- 원하는 추천 개수를 선택하세요.", min_value=1, max_value=15, value=5, key="slider")
    st.text("")

    st.text(f"- 아래 버튼을 누르면 위 상품명에 알맞은 {topk_num}개의 카테고리가 출력됩니다.")

    if st.button("카테고리 추천", key="button"):
        if not goods_name:
            st.error(f"상품명이 비었습니다.")
        else:
            print(f"> request to {url}")

            with st.spinner("분석 중 . . ."):
                result = requests.post(url, data={"goods_name": goods_name, "topk_num": topk_num})
                print(f"> response from {url}")

            if result.status_code == 200:
                result_json = result.json()

                st.success("추천 성공")
                st.text(f"> 상품명: {goods_name}")
                st.text(f"> 소요 시간: {result_json['spend_time']} sec")
                del result_json["spend_time"]
                st.table(result_json)

            else:
                st.error(f"추천 실패 (분석이 불가능한 상품명)")


if __name__ == "__main__":
    st.set_page_config(page_title="Category Recommender", page_icon="rainbow", layout="wide")
    print(f"\nRe Run {datetime.datetime.utcnow() + datetime.timedelta(hours=9)} KST")
    url = "http://fastapi:8502/recommend"
    goods_ex = (
        "클링턴 중 스푼 커트러리 양식기 수저 숟가락 디저트",
        "예스뷰티 전문가용 헤어드라이어 YB-801",
        "간편 속눈썹-미인 DG 타입 붙이는속눈썹 아이매이크업",
        "예초기 날 나일론 줄 뭉치 이중4각날 2.6 50M",
        "조카 손주 손녀 투푹수아라 공룡 모형 생일 선물 완구",
        "안드로이드 케이블 Micro USB 1.5m Silver",
        "하드우드 750 프리미엄 입식테이블/철제테이블",
        "공구팜(09farm)LED 메이크업 손거울 FARM-8LEDP 핑크",
        "발레딘 마호가니원목 2단 엔틱서랍장 협탁",
        "자석거치대 철판 원형홀더x5p_블랙/핸드폰"
    )
    main()
