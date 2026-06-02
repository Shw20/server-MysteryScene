"""MysteryScene 규칙 기반 입력 키워드 사전.

팀원이 수정하기 쉽도록 사용자 표현 사전과 실제 라우팅용 키워드를
컨트롤러에서 분리해 둔다. RAW_INTENT_KEYWORDS는 기획용 원본 사전에 가깝고,
아래 *_TERMS 상수들은 현재 컨트롤러가 안전하게 사용하는 정규화 키워드다.

이 파일을 수정할 때의 기준:
- RAW_INTENT_KEYWORDS: 기획 문서에 가까운 전체 표현 목록이다.
- *_TERMS: 실제 서버 라우팅에 쓰는 안전한 키워드 목록이다.
- 너무 넓은 단어를 *_TERMS에 넣으면 엉뚱한 단서가 열릴 수 있으니 주의한다.
"""


def normalize_keywords(values: list[str] | set[str]) -> set[str]:
    """컨트롤러의 부분 문자열 매칭 규칙과 같은 방식으로 키워드를 정규화한다."""

    return {
        "".join(ch for ch in str(value).casefold() if ch.isalnum())
        for value in values
        if str(value).strip()
    }


RAW_INTENT_KEYWORDS = {
    # 아래 원본 사전은 팀원/기획자가 "이 표현도 같은 행동"이라고 이해하기 위한 표다.
    # 컨트롤러가 전부 그대로 쓰지는 않고, 실제 라우팅은 아래 *_TERMS를 사용한다.
    "inspect_memo": [
        "메모",
        "쪽지",
        "종이",
        "눌린 글씨",
        "글씨",
        "필기",
        "낙서",
        "메모지",
        "문구",
        "단어",
        "힌트",
        "종이 조사",
        "메모 조사",
        "메모 확인",
        "메모 보기",
        "메모 읽기",
        "종이 본다",
        "종이 확인",
        "글씨 확인",
        "눌린 자국",
    ],
    "inspect_beer": [
        "맥주",
        "흑맥주",
        "기네스",
        "캔",
        "맥주캔",
        "캔 조사",
        "캔 확인",
        "맥주 조사",
        "흑맥주 조사",
        "브랜드",
        "로고",
        "맥주 브랜드",
        "Guinness",
        "술",
        "흘린 자국",
        "맥주 자국",
    ],
    "ask_relation_guinness": [
        "기네스북",
        "기네스 기록",
        "세계기록",
        "세계 기록",
        "기록",
        "기록책",
        "largest",
        "가장 큰",
        "기네스 세계기록",
        "세계에서 가장 큰",
        "세계 최대",
    ],
    "inspect_largest_book": [
        "책",
        "큰 책",
        "가장 큰 책",
        "세계에서 가장 큰 책",
        "largest book",
        "book",
        "도서",
        "거대한 책",
    ],
    "inspect_largest_mailbox": [
        "우체통",
        "큰 우체통",
        "가장 큰 우체통",
        "세계에서 가장 큰 우체통",
        "mailbox",
        "postbox",
        "편지함",
        "우편함",
    ],
    "inspect_largest_sundial": [
        "해시계",
        "큰 해시계",
        "가장 큰 해시계",
        "세계에서 가장 큰 해시계",
        "sundial",
    ],
    "analyze_common_pattern": [
        "공통점",
        "연결점",
        "연관",
        "관련",
        "의미",
        "패턴",
        "규칙",
        "왜",
        "공통",
        "이어진다",
        "연결된다",
        "다 같은",
        "다 비슷한",
        "흐름",
        "힌트",
        "무슨 뜻",
    ],
    "inspect_trojan_horse": [
        "트로이",
        "트로이의 목마",
        "목마",
        "trojan horse",
        "horse",
        "거대한 목마",
        "큰 목마",
        "가장 큰 목마",
        "세계에서 가장 큰 목마",
    ],
    "guess_location": [
        "어디",
        "위치",
        "장소",
        "여행지",
        "목적지",
        "행선지",
        "도시",
        "지역",
        "나라",
        "어디로 갔지",
        "어디 있는거야",
    ],
    "submit_answer": [
        "여주",
        "경기도 여주",
        "정답",
        "답",
        "마지막 장소",
        "최종 장소",
        "최종 목적지",
        "여기가 답",
    ],
    "request_hint": [
        "힌트",
        "도움",
        "모르겠다",
        "어렵다",
        "막혔다",
        "다음",
        "어떻게",
        "뭐 해야 돼",
        "진행 안 돼",
    ],
    "force_answer": [
        "정답 알려줘",
        "답 알려줘",
        "스킵",
        "그냥 말해",
        "모르겠어 답",
        "끝내기",
        "바로 정답",
    ],
    "invalid_action": [
        "범인",
        "경찰",
        "죽였다",
        "집 간다",
        "게임 종료",
        "탈출",
        "싸운다",
        "도망간다",
    ],
}

INTENT_TO_TARGET = {
    # 원본 intent 이름을 LLM/컨트롤러에서 쓰는 추상 target 이름으로 바꾸는 표다.
    # 최종 단서 ID(E01 등)로 바꾸는 작업은 investigation_controller.py에서 한다.
    "inspect_memo": "memo",
    "inspect_beer": "beer",
    "ask_relation_guinness": "record_system",
    "inspect_largest_book": "book_record",
    "inspect_largest_mailbox": "postbox_record",
    "inspect_largest_sundial": "sundial_record",
    "analyze_common_pattern": "pattern",
    "inspect_trojan_horse": "final_location",
    "guess_location": "final_location",
    "submit_answer": "final_location",
}

# 실제 라우팅은 포괄 키워드를 그대로 쓰지 않는다.
# 예: "기네스"는 캔 브랜드이면서 기록 체계라서 beer보다 record 쪽에서 우선 처리한다.
# 아래 상수들은 모두 normalize_keywords를 거쳐 공백/대소문자 차이를 제거한 뒤 매칭된다.
ACTION_VERB_TERMS = normalize_keywords(
    {
        "간다",
        "뒤져",
        "뒤진",
        "둘러",
        "본다",
        "보다",
        "살펴",
        "열어",
        "이동",
        "조사",
        "찾아",
        "찾는",
        "탐색",
        "확인",
        "검색",
        "정리",
        "대조",
    }
)
STUDY_ACTION_TERMS = normalize_keywords({"서재", "현장", "주변"})
CHAIR_ACTION_TERMS = normalize_keywords({"안락의자", "의자"})

# E01: 테이블 위 메모 단서로 라우팅하는 표현들.
MEMO_ACTION_TERMS = normalize_keywords(
    {
        "글자",
        "눌러쓴",
        "눌러",
        "눌린 글씨",
        "눌린 자국",
        "메모",
        "메모지",
        "쪽지",
        "사이드테이블",
        "용지",
        "자국",
        "종이",
        "테이블",
    }
)

# E02: 쓰레기통의 흑맥주 캔 단서로 라우팅하는 표현들.
BEER_ACTION_TERMS = normalize_keywords(
    {
        "맥주",
        "맥주캔",
        "쓰레기통",
        "캔",
        "휴지통",
        "흑맥주",
        "로고",
        "맥주 브랜드",
        "흘린 자국",
        "맥주 자국",
    }
)

# E03: 기네스/세계기록 연결 단서로 라우팅하는 표현들.
RECORD_ACTION_TERMS = normalize_keywords(
    {
        "guinness",
        "기네스",
        "기네스북",
        "기네스 기록",
        "기네스 세계기록",
        "브랜드",
        "세계기록",
        "세계 기록",
        "기록",
        "기록책",
        "largest",
        "세계에서 가장 큰",
        "세계 최대",
    }
)

# E04/E05/E06: 메모 속 세 물건의 "가장 큰 X" 기록 조사 표현들.
BOOK_ACTION_TERMS = normalize_keywords(
    {"가장 큰 책", "세계에서 가장 큰 책", "큰 책", "largest book", "book", "거대한 책", "책"}
)
POSTBOX_ACTION_TERMS = normalize_keywords(
    {
        "가장 큰 우체통",
        "세계에서 가장 큰 우체통",
        "큰 우체통",
        "mailbox",
        "postbox",
        "편지함",
        "우편함",
        "우체통",
    }
)
SUNDIAL_ACTION_TERMS = normalize_keywords(
    {"가장 큰 해시계", "세계에서 가장 큰 해시계", "큰 해시계", "sundial", "해시계"}
)

# E07: 책/우체통/해시계의 공통 규칙을 정리하는 표현들.
PATTERN_ACTION_TERMS = normalize_keywords(
    {
        "공통",
        "공통점",
        "연결점",
        "다 같은",
        "다 비슷한",
        "규칙",
        "나머지",
        "패턴",
        "흐름",
        "대조",
    }
)

# E08 또는 최종 목적지 계열 표현들.
# 단, 트로이 목마 기록 조사 표현은 E07과 겹치므로 TROJAN_PATTERN_TERMS로 한 번 더 보정한다.
FINAL_ACTION_TERMS = normalize_keywords(
    {
        "마지막 여행지",
        "목마",
        "최종 목적지",
        "트로이",
        "트로이 목마",
        "트로이의 목마",
        "trojan horse",
        "horse",
        "거대한 목마",
        "큰 목마",
        "가장 큰 목마",
        "세계에서 가장 큰 목마",
        "여주",
        "경기도 여주",
        "국내",
        "국내 장소",
        "국내 트로이",
        "국내 트로이의 목마",
        "어디",
        "위치",
        "장소",
        "여행지",
        "목적지",
        "행선지",
    }
)

# 해시계 이후 "트로이 목마" 입력을 최종 목적지(E08)가 아니라
# 먼저 패턴 단서(E07)로 보내기 위한 전용 키워드다.
TROJAN_PATTERN_TERMS = normalize_keywords(
    {
        "트로이",
        "트로이 목마",
        "트로이의 목마",
        "trojan horse",
        "horse",
        "거대한 목마",
        "큰 목마",
        "가장 큰 목마",
        "가장 큰 트로이 목마",
        "가장 큰 트로이의 목마",
        "세계에서 가장 큰 목마",
        "세계에서 가장 큰 트로이 목마",
        "세계에서 가장 큰 트로이의 목마",
    }
)

# "그건 뭐랑 관련 있어?"처럼 방금 본 단서에 붙는 후속 질문 판별용 키워드다.
FOLLOWUP_REFERENCE_TERMS = normalize_keywords(
    {"그건", "그거", "그단서", "그럼그건", "방금", "앞서", "이건", "이거", "이단서"}
)
FOLLOWUP_RELATION_TERMS = normalize_keywords(
    {"관계", "관련", "뭐랑", "무엇과", "연결", "연관", "이어"}
)

# 아래 키워드들은 정답 진행과 무관한 자유 행동을 분위기 응답으로 처리할 때 사용한다.
DRINK_ACTION_TERMS = normalize_keywords(
    {"들이켜", "마셔", "마신", "마신다", "맛봐", "맛본", "먹어", "먹는다"}
)
SIT_ACTION_TERMS = normalize_keywords({"걸터", "눕", "앉"})
CALL_ACTION_TERMS = normalize_keywords({"전화", "연락", "문자", "카톡", "부른", "불러"})
LEAVE_ACTION_TERMS = normalize_keywords({"나가", "나간", "나간다", "도망", "떠나", "밖으로"})
DAMAGE_ACTION_TERMS = normalize_keywords(
    {"깨", "던져", "망가", "부순", "부숴", "불", "찢", "태워", "파괴", "훼손"}
)
ODD_OBJECT_ACTION_TERMS = normalize_keywords({"창문", "커튼", "램프", "서랍", "바닥", "문고리"})
GENERIC_LOCKED_TERMS = normalize_keywords(
    {"final", "finallocation", "solution", "결론", "단서", "여행지", "정답", "최종"}
)
