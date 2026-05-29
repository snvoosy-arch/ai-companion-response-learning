from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"
PREFIX = "black_meaning_cross_domain_schema_boundary_v1_20260430"


ROWS: list[tuple[str, str, str, str]] = [
    ("XB160001", "relationship", "서운한 일이 생기면 바로 말하는 타입이야, 아니면 혼자 정리하고 나중에 말하는 타입이야?", "self_style"),
    ("XB160002", "relationship", "좋아하는 사람이 생기면 티가 나는 편이야, 아니면 끝까지 숨기는 편이야?", "self_style"),
    ("XB160003", "relationship", "다투고 나면 먼저 손 내미는 쪽이야, 상대가 풀릴 때까지 기다리는 쪽이야?", "self_style"),
    ("XB160004", "relationship", "연락이 늦어지면 바로 불안해지는 편이야, 꽤 담담하게 넘기는 편이야?", "self_style"),
    ("XB160005", "relationship", "관계에서 선을 정할 때 말로 분명히 하는 타입이야, 분위기로 맞추는 타입이야?", "self_style"),
    ("XB160006", "relationship", "상대가 힘들다고 하면 해결책부터 찾는 편이야, 일단 옆에서 들어주는 편이야?", "self_style"),
    ("XB160007", "relationship", "호감이 생기면 자주 연락하는 타입이야, 일부러 템포를 천천히 두는 타입이야?", "self_style"),
    ("XB160008", "relationship", "친한 사람한테 장난을 많이 치는 편이야, 말투가 더 조심스러워지는 편이야?", "self_style"),
    ("XB160009", "relationship", "사과할 때 길게 설명하는 편이야, 짧고 정확하게 말하는 편이야?", "self_style"),
    ("XB160010", "relationship", "상대가 애매하게 굴면 바로 확인하는 편이야, 조금 더 지켜보는 편이야?", "self_style"),
    ("XB160011", "relationship", "질투가 나면 얼굴에 티 나는 편이야, 속으로만 눌러두는 편이야?", "self_style"),
    ("XB160012", "relationship", "친해질수록 말이 많아지는 타입이야, 오히려 조용히 편해지는 타입이야?", "self_style"),
    ("XB160013", "relationship", "상대의 작은 변화도 빨리 알아차리는 편이야, 말해줘야 아는 편이야?", "self_style"),
    ("XB160014", "relationship", "좋은 감정은 바로 표현하는 편이야, 확신이 생길 때까지 아끼는 편이야?", "self_style"),
    ("XB160015", "relationship", "갈등이 생기면 그 자리에서 풀어야 하는 타입이야, 시간을 두고 다시 보는 타입이야?", "self_style"),
    ("XB160016", "relationship", "상대가 농담으로 넘겨도 진심을 읽으려는 편이야, 그대로 가볍게 받는 편이야?", "self_style"),
    ("XB160017", "relationship", "관계가 어색해지면 먼저 분위기를 풀어보는 편이야, 조용히 거리를 두는 편이야?", "self_style"),
    ("XB160018", "relationship", "기념일 같은 건 챙기는 타입이야, 일상에서 꾸준히 챙기는 쪽이 더 맞아?", "self_style"),
    ("XB160019", "relationship", "좋아하는 사람 앞에서는 말이 빨라지는 편이야, 오히려 말수가 줄어드는 편이야?", "self_style"),
    ("XB160020", "relationship", "상대에게 기대는 걸 편하게 하는 편이야, 혼자 감당하려는 편이야?", "self_style"),
    ("XB160021", "relationship", "연락은 자주 하는 관계가 좋아, 각자 시간 존중하는 관계가 좋아?", "preference_disclosure"),
    ("XB160022", "relationship", "데이트는 집에서 편하게 쉬는 쪽이 좋아, 밖에서 돌아다니는 쪽이 좋아?", "preference_disclosure"),
    ("XB160023", "relationship", "감정 표현은 직접적인 사람이 좋아, 은근히 챙겨주는 사람이 좋아?", "preference_disclosure"),
    ("XB160024", "relationship", "갈등이 생기면 바로 대화하는 방식이 좋아, 조금 식히고 말하는 방식이 좋아?", "preference_disclosure"),
    ("XB160025", "relationship", "관계에서는 설렘이 더 중요해, 편안함이 더 중요해?", "preference_disclosure"),
    ("XB160026", "relationship", "상대가 계획적인 편이 좋아, 즉흥적인 편이 좋아?", "preference_disclosure"),
    ("XB160027", "relationship", "친구 같은 연애가 좋아, 서로 확실히 챙기는 연애가 좋아?", "preference_disclosure"),
    ("XB160028", "relationship", "말 많은 사람이 좋아, 조용하지만 안정적인 사람이 좋아?", "preference_disclosure"),
    ("XB160029", "relationship", "기념일을 크게 챙기는 게 좋아, 소소하게 넘어가는 게 좋아?", "preference_disclosure"),
    ("XB160030", "relationship", "상대가 솔직하게 다 말해주는 게 좋아, 필요한 만큼만 말하는 게 좋아?", "preference_disclosure"),
    ("XB160031", "relationship", "공개적으로 애정 표현하는 쪽이 좋아, 둘만 있을 때 표현하는 쪽이 좋아?", "preference_disclosure"),
    ("XB160032", "relationship", "싸운 뒤에는 먼저 안아주는 사람이 좋아, 차분히 설명하는 사람이 좋아?", "preference_disclosure"),
    ("XB160033", "relationship", "연애 초반에는 천천히 알아가는 게 좋아, 확실히 빠르게 가까워지는 게 좋아?", "preference_disclosure"),
    ("XB160034", "relationship", "상대와 취미가 비슷한 게 좋아, 서로 다른 세계를 갖는 게 좋아?", "preference_disclosure"),
    ("XB160035", "relationship", "장거리라도 깊은 관계가 좋아, 가까이 자주 보는 관계가 좋아?", "preference_disclosure"),
    ("XB160036", "relationship", "연락할 때 짧고 자주가 좋아, 길게 한 번이 좋아?", "preference_disclosure"),
    ("XB160037", "relationship", "상대가 다정한 말투인 게 좋아, 담백하고 솔직한 말투가 좋아?", "preference_disclosure"),
    ("XB160038", "relationship", "데이트 코스는 미리 정해두는 게 좋아, 만나서 정하는 게 좋아?", "preference_disclosure"),
    ("XB160039", "relationship", "친구들에게 소개하는 관계가 좋아, 둘만 조용히 아는 관계가 좋아?", "preference_disclosure"),
    ("XB160040", "relationship", "좋아하는 마음은 말로 듣는 게 좋아, 행동으로 느끼는 게 좋아?", "preference_disclosure"),
    ("XB160041", "relationship", "평소에 친한 사람 안부를 먼저 묻는 편이야?", "habit_preference"),
    ("XB160042", "relationship", "메시지 답장은 바로바로 하는 편이야?", "habit_preference"),
    ("XB160043", "relationship", "상대가 말한 작은 취향을 기억해두는 편이야?", "habit_preference"),
    ("XB160044", "relationship", "싸운 뒤에도 먼저 일상 이야기를 꺼내는 편이야?", "habit_preference"),
    ("XB160045", "relationship", "친구나 연인한테 고마운 일 있으면 바로 표현하는 편이야?", "habit_preference"),
    ("XB160046", "relationship", "약속 잡을 때 상대 일정부터 먼저 확인하는 편이야?", "habit_preference"),
    ("XB160047", "relationship", "상대가 힘들어 보이면 먼저 괜찮냐고 묻는 편이야?", "habit_preference"),
    ("XB160048", "relationship", "기분이 상해도 일단 웃어넘기는 편이야?", "habit_preference"),
    ("XB160049", "relationship", "대화가 끊기면 먼저 새 주제를 꺼내는 편이야?", "habit_preference"),
    ("XB160050", "relationship", "친한 사람 생일이나 중요한 날을 잘 챙기는 편이야?", "habit_preference"),
    ("XB160051", "relationship", "서운한 게 쌓이면 메모처럼 정리해두는 편이야?", "habit_preference"),
    ("XB160052", "relationship", "상대 말투가 달라지면 이유를 자주 생각하는 편이야?", "habit_preference"),
    ("XB160053", "relationship", "좋은 일이 생기면 가까운 사람한테 먼저 말하는 편이야?", "habit_preference"),
    ("XB160054", "relationship", "관계가 멀어진 것 같으면 먼저 연락해보는 편이야?", "habit_preference"),
    ("XB160055", "relationship", "상대가 좋아한다던 장소나 음식을 기억하는 편이야?", "habit_preference"),
    ("XB160056", "relationship", "친해져도 기본 예의는 일부러 더 지키는 편이야?", "habit_preference"),
    ("XB160057", "relationship", "상대가 바쁘다고 하면 연락을 줄여주는 편이야?", "habit_preference"),
    ("XB160058", "relationship", "감정이 올라올 때 바로 말하기보다 하루 정도 두는 편이야?", "habit_preference"),
    ("XB160059", "relationship", "친한 사람한테 농담 섞어서 걱정 표현하는 편이야?", "habit_preference"),
    ("XB160060", "relationship", "관계에서 문제가 생기면 같은 이야기를 여러 번 되짚는 편이야?", "habit_preference"),
    ("XB160061", "relationship", "상대 답장이 늦어서 서운할 때 바로 말하는 게 나아, 조금 기다리는 게 나아?", "soft_decision_advice"),
    ("XB160062", "relationship", "친구가 약속을 자주 늦으면 바로 지적하는 게 좋을까, 한 번 더 지켜보는 게 좋을까?", "soft_decision_advice"),
    ("XB160063", "relationship", "연락 빈도 때문에 다투면 기준을 정하는 게 나아, 그때그때 맞추는 게 나아?", "soft_decision_advice"),
    ("XB160064", "relationship", "상대가 힘들다는데 나도 지쳐 있으면 어떻게 반응하는 게 좋을까?", "soft_decision_advice"),
    ("XB160065", "relationship", "호감이 애매할 때는 먼저 물어보는 게 나아, 자연스럽게 더 보는 게 나아?", "soft_decision_advice"),
    ("XB160066", "relationship", "친구 사이가 갑자기 어색해지면 먼저 연락해볼까, 시간을 두는 게 좋을까?", "soft_decision_advice"),
    ("XB160067", "relationship", "서운한 말을 들었을 때 바로 반박하는 게 나아, 감정 정리하고 말하는 게 나아?", "soft_decision_advice"),
    ("XB160068", "relationship", "상대가 선을 넘는 농담을 하면 웃어넘길까, 바로 선을 말하는 게 좋을까?", "soft_decision_advice"),
    ("XB160069", "relationship", "관계가 식은 것 같을 때 대화를 먼저 꺼내는 게 나아, 행동을 더 지켜보는 게 나아?", "soft_decision_advice"),
    ("XB160070", "relationship", "좋아하는 마음을 숨기는 게 힘들면 고백하는 게 나아, 더 확신을 기다리는 게 나아?", "soft_decision_advice"),
    ("XB160071", "relationship", "상대가 바쁜 시기에는 연락을 줄이는 게 배려일까, 오히려 챙겨주는 게 좋을까?", "soft_decision_advice"),
    ("XB160072", "relationship", "오해가 생겼을 때 길게 설명하는 게 좋을까, 핵심만 짧게 말하는 게 좋을까?", "soft_decision_advice"),
    ("XB160073", "relationship", "친구가 내 비밀을 가볍게 말했으면 바로 따지는 게 나아, 조용히 거리를 두는 게 나아?", "soft_decision_advice"),
    ("XB160074", "relationship", "상대가 내 기분을 몰라줄 때 기대를 낮추는 게 나아, 정확히 말하는 게 나아?", "soft_decision_advice"),
    ("XB160075", "relationship", "관계에서 반복되는 문제가 있으면 한 번 더 기회를 주는 게 좋을까, 선을 긋는 게 좋을까?", "soft_decision_advice"),
    ("XB160076", "relationship", "질투가 날 때 솔직히 말하는 게 나아, 혼자 정리하는 게 나아?", "soft_decision_advice"),
    ("XB160077", "relationship", "사과를 받았는데 마음이 덜 풀렸으면 바로 말하는 게 좋을까, 조금 시간을 두는 게 좋을까?", "soft_decision_advice"),
    ("XB160078", "relationship", "친구가 힘든 얘기를 반복하면 계속 들어주는 게 나아, 조심스럽게 도움을 권하는 게 나아?", "soft_decision_advice"),
    ("XB160079", "relationship", "상대와 생활 리듬이 너무 다르면 맞춰보는 게 나아, 처음부터 현실적으로 보는 게 나아?", "soft_decision_advice"),
    ("XB160080", "relationship", "내가 서운한 건지 예민한 건지 헷갈릴 때는 어떻게 판단하는 게 좋아?", "soft_decision_advice"),
    ("XB160081", "work_school", "일이나 공부 시작할 때 바로 집중하는 타입이야, 준비 시간이 오래 걸리는 타입이야?", "self_style"),
    ("XB160082", "work_school", "마감이 가까워져야 힘이 나는 타입이야, 미리 끝내야 마음 편한 타입이야?", "self_style"),
    ("XB160083", "work_school", "회의나 조별과제에서 먼저 말 꺼내는 편이야, 듣다가 필요한 때 말하는 편이야?", "self_style"),
    ("XB160084", "work_school", "모르는 게 있으면 바로 질문하는 타입이야, 혼자 찾아보고 묻는 타입이야?", "self_style"),
    ("XB160085", "work_school", "업무나 과제가 많아지면 목록부터 만드는 편이야, 눈앞의 것부터 처리하는 편이야?", "self_style"),
    ("XB160086", "work_school", "실수했을 때 바로 공유하는 편이야, 해결책 찾고 나서 말하는 편이야?", "self_style"),
    ("XB160087", "work_school", "새로운 일을 맡으면 부담보다 흥미가 먼저 오는 타입이야?", "self_style"),
    ("XB160088", "work_school", "팀에서 분위기 맞추는 쪽이야, 기준을 세우는 쪽이야?", "self_style"),
    ("XB160089", "work_school", "집중 안 될 때 자리부터 정리하는 편이야, 일단 손부터 움직이는 편이야?", "self_style"),
    ("XB160090", "work_school", "칭찬받으면 더 힘나는 타입이야, 부담이 커지는 타입이야?", "self_style"),
    ("XB160091", "work_school", "업무 피드백은 바로 듣는 게 편한 편이야, 정리된 글로 받는 게 편한 편이야?", "self_style"),
    ("XB160092", "work_school", "어려운 일을 보면 피하고 싶어지는 편이야, 오히려 붙잡고 파는 편이야?", "self_style"),
    ("XB160093", "work_school", "발표할 때 긴장해도 밀고 가는 편이야, 시작 전에 많이 굳는 편이야?", "self_style"),
    ("XB160094", "work_school", "계획이 틀어지면 빨리 새 계획 세우는 타입이야, 잠깐 멈추는 타입이야?", "self_style"),
    ("XB160095", "work_school", "동료나 친구가 막히면 먼저 도와주는 편이야, 요청이 올 때까지 기다리는 편이야?", "self_style"),
    ("XB160096", "work_school", "반복 업무는 안정적으로 느끼는 편이야, 금방 지루해지는 편이야?", "self_style"),
    ("XB160097", "work_school", "일이 많을 때 말수가 줄어드는 편이야, 오히려 말하면서 푸는 편이야?", "self_style"),
    ("XB160098", "work_school", "성과가 안 보이면 쉽게 흔들리는 편이야, 과정만 맞으면 버티는 편이야?", "self_style"),
    ("XB160099", "work_school", "공부할 때 혼자 파는 타입이야, 누가 옆에 있어야 잘 되는 타입이야?", "self_style"),
    ("XB160100", "work_school", "바쁜 날에는 감정 표현이 줄어드는 편이야, 평소처럼 유지하려는 편이야?", "self_style"),
    ("XB160101", "work_school", "일할 때 조용한 환경이 좋아, 적당히 소리 있는 환경이 좋아?", "preference_disclosure"),
    ("XB160102", "work_school", "공부는 아침에 하는 게 좋아, 밤에 하는 게 좋아?", "preference_disclosure"),
    ("XB160103", "work_school", "팀플은 역할이 분명한 게 좋아, 자유롭게 나누는 게 좋아?", "preference_disclosure"),
    ("XB160104", "work_school", "피드백은 직설적인 게 좋아, 부드럽게 돌려 말하는 게 좋아?", "preference_disclosure"),
    ("XB160105", "work_school", "업무는 빠르게 쳐내는 게 좋아, 완성도를 높이는 게 좋아?", "preference_disclosure"),
    ("XB160106", "work_school", "회의는 짧고 바로 끝나는 게 좋아, 충분히 이야기하는 게 좋아?", "preference_disclosure"),
    ("XB160107", "work_school", "공부할 때 필기하면서 하는 게 좋아, 그냥 읽고 이해하는 게 좋아?", "preference_disclosure"),
    ("XB160108", "work_school", "업무 메신저는 빠른 답장이 좋아, 필요한 때만 답하는 게 좋아?", "preference_disclosure"),
    ("XB160109", "work_school", "출근이나 등교는 여유 있게 도착하는 게 좋아, 딱 맞춰 가는 게 좋아?", "preference_disclosure"),
    ("XB160110", "work_school", "쉬는 시간에는 혼자 있는 게 좋아, 사람들과 잡담하는 게 좋아?", "preference_disclosure"),
    ("XB160111", "work_school", "과제는 혼자 하는 게 좋아, 같이 나눠서 하는 게 좋아?", "preference_disclosure"),
    ("XB160112", "work_school", "일정 관리는 앱으로 하는 게 좋아, 종이에 적는 게 좋아?", "preference_disclosure"),
    ("XB160113", "work_school", "업무는 익숙한 루틴이 좋아, 새로운 프로젝트가 좋아?", "preference_disclosure"),
    ("XB160114", "work_school", "공부 장소는 도서관이 좋아, 카페가 좋아?", "preference_disclosure"),
    ("XB160115", "work_school", "상사는 세세히 봐주는 사람이 좋아, 믿고 맡기는 사람이 좋아?", "preference_disclosure"),
    ("XB160116", "work_school", "마감 전에는 혼자 집중하는 게 좋아, 같이 점검하는 게 좋아?", "preference_disclosure"),
    ("XB160117", "work_school", "문제 풀이는 쉬운 것부터 하는 게 좋아, 어려운 것부터 잡는 게 좋아?", "preference_disclosure"),
    ("XB160118", "work_school", "회사나 학교에서는 넓은 인간관계가 좋아, 깊은 몇 명이 좋아?", "preference_disclosure"),
    ("XB160119", "work_school", "일할 때 음악이 있는 게 좋아, 완전 조용한 게 좋아?", "preference_disclosure"),
    ("XB160120", "work_school", "쉬는 날에는 밀린 일을 처리하는 게 좋아, 완전히 끊고 쉬는 게 좋아?", "preference_disclosure"),
    ("XB160121", "work_school", "평소에 할 일을 목록으로 적어두는 편이야?", "habit_preference"),
    ("XB160122", "work_school", "마감 날짜를 자주 확인하는 편이야?", "habit_preference"),
    ("XB160123", "work_school", "업무나 공부 시작 전에 책상 정리부터 하는 편이야?", "habit_preference"),
    ("XB160124", "work_school", "집중하려고 휴대폰 알림을 꺼두는 편이야?", "habit_preference"),
    ("XB160125", "work_school", "점심 먹고 나면 커피를 찾는 편이야?", "habit_preference"),
    ("XB160126", "work_school", "회의 내용이나 수업 내용을 바로 메모하는 편이야?", "habit_preference"),
    ("XB160127", "work_school", "일정이 바뀌면 캘린더를 바로 수정하는 편이야?", "habit_preference"),
    ("XB160128", "work_school", "아침에 도착하면 먼저 메일이나 공지부터 확인하는 편이야?", "habit_preference"),
    ("XB160129", "work_school", "공부하다 막히면 검색부터 하는 편이야?", "habit_preference"),
    ("XB160130", "work_school", "쉬는 시간에도 할 일을 생각하는 편이야?", "habit_preference"),
    ("XB160131", "work_school", "작업 파일 이름을 정리해서 저장하는 편이야?", "habit_preference"),
    ("XB160132", "work_school", "업무나 과제 끝나면 한 번 더 검토하는 편이야?", "habit_preference"),
    ("XB160133", "work_school", "집중이 끊기면 짧게 산책하거나 물 마시는 편이야?", "habit_preference"),
    ("XB160134", "work_school", "주말에도 다음 주 할 일을 미리 떠올리는 편이야?", "habit_preference"),
    ("XB160135", "work_school", "중요한 일은 알람을 여러 개 맞춰두는 편이야?", "habit_preference"),
    ("XB160136", "work_school", "모르는 내용은 따로 모아서 나중에 다시 보는 편이야?", "habit_preference"),
    ("XB160137", "work_school", "일이나 공부가 끝나도 머릿속에서 계속 되짚는 편이야?", "habit_preference"),
    ("XB160138", "work_school", "새로운 도구나 앱이 나오면 업무에 써보는 편이야?", "habit_preference"),
    ("XB160139", "work_school", "피곤해도 정해둔 공부량은 채우려는 편이야?", "habit_preference"),
    ("XB160140", "work_school", "업무 중간에 진행 상황을 자주 공유하는 편이야?", "habit_preference"),
    ("XB160141", "work_school", "마감이 가까운데 완성도가 애매하면 제출하는 게 나아, 조금 더 붙잡는 게 나아?", "soft_decision_advice"),
    ("XB160142", "work_school", "상사가 주말에 연락하면 바로 답하는 게 좋을까, 월요일까지 기다려도 될까?", "soft_decision_advice"),
    ("XB160143", "work_school", "과제가 너무 많을 때 쉬운 것부터 할까, 중요한 것부터 할까?", "soft_decision_advice"),
    ("XB160144", "work_school", "팀원이 일을 늦게 주면 바로 말하는 게 나아, 한 번 더 기다리는 게 나아?", "soft_decision_advice"),
    ("XB160145", "work_school", "공부가 안 되는 날은 억지로 앉아 있는 게 나아, 짧게 쉬고 다시 하는 게 나아?", "soft_decision_advice"),
    ("XB160146", "work_school", "회의에서 내 의견이 묻히면 다시 말하는 게 좋을까, 나중에 따로 정리해서 보내는 게 좋을까?", "soft_decision_advice"),
    ("XB160147", "work_school", "모르는 업무를 받았을 때 바로 물어보는 게 나아, 먼저 찾아보고 질문하는 게 나아?", "soft_decision_advice"),
    ("XB160148", "work_school", "발표가 불안하면 원고를 다 외우는 게 나아, 키워드만 잡는 게 나아?", "soft_decision_advice"),
    ("XB160149", "work_school", "퇴근 후 운동을 갈까 말까 고민될 때는 몸 상태를 먼저 봐야 할까, 계획을 지켜야 할까?", "soft_decision_advice"),
    ("XB160150", "work_school", "새 프로젝트가 부담되면 맡아보는 게 좋을까, 지금 일부터 안정시키는 게 좋을까?", "soft_decision_advice"),
    ("XB160151", "work_school", "야근이 반복되면 바로 말하는 게 나아, 상황이 지나갈 때까지 버티는 게 나아?", "soft_decision_advice"),
    ("XB160152", "work_school", "시험 전날 잠을 줄여서 더 볼까, 컨디션 챙기고 자는 게 나을까?", "soft_decision_advice"),
    ("XB160153", "work_school", "동료가 실수했을 때 바로 알려주는 게 좋을까, 분위기 보고 말하는 게 좋을까?", "soft_decision_advice"),
    ("XB160154", "work_school", "집중이 안 되면 장소를 바꾸는 게 나아, 하던 자리에서 버티는 게 나아?", "soft_decision_advice"),
    ("XB160155", "work_school", "일이 너무 많으면 도움을 요청하는 게 나아, 우선순위를 줄여보는 게 나아?", "soft_decision_advice"),
    ("XB160156", "work_school", "조별과제에서 아무도 안 나서면 내가 맡는 게 나아, 역할 분담부터 요구하는 게 나아?", "soft_decision_advice"),
    ("XB160157", "work_school", "업무 피드백이 모호하면 다시 물어보는 게 나아, 일단 내 방식대로 진행하는 게 나아?", "soft_decision_advice"),
    ("XB160158", "work_school", "수업이나 회의가 지루해도 끝까지 집중하려고 해야 할까, 핵심만 잡아도 될까?", "soft_decision_advice"),
    ("XB160159", "work_school", "해야 할 일이 많아서 막막하면 계획부터 세우는 게 나아, 하나라도 바로 시작하는 게 나아?", "soft_decision_advice"),
    ("XB160160", "work_school", "성과가 잘 안 나올 때 방향을 바꾸는 게 좋을까, 조금 더 밀어붙이는 게 좋을까?", "soft_decision_advice"),
]


def _row(*, item_id: str, domain: str, text: str, schema: str) -> dict[str, Any]:
    cues = [f"domain_{domain}", schema]
    targets = {
        "coarse_intent": "smalltalk_opinion",
        "domain": domain,
        "schema": schema,
        "speech_act": "ask",
        "pragmatic_cues": cues,
        "slots": {},
        "slot_spans": [],
    }
    return {
        "id": item_id,
        "text": text,
        "coarse_intent": targets["coarse_intent"],
        "domain": domain,
        "schema": schema,
        "speech_act": targets["speech_act"],
        "pragmatic_cues": cues,
        "slots": {},
        "slot_spans": [],
        "targets": targets,
        "label_status": "gold_direct",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "manual_cross_domain_schema_boundary",
            "source_version": PREFIX,
            "category": f"{domain}_schema_boundary",
            "no_seed_expansion": True,
        },
    }


def build_rows() -> list[dict[str, Any]]:
    return [
        _row(item_id=item_id, domain=domain, text=text, schema=schema)
        for item_id, domain, text, schema in ROWS
    ]


def _probe_row(row: dict[str, Any]) -> dict[str, Any]:
    targets = row["targets"]
    return {
        "id": row["id"],
        "text": row["text"],
        "expect": {
            "coarse": targets["coarse_intent"],
            "domain": targets["domain"],
            "schema": targets["schema"],
            "speech_act": targets["speech_act"],
            "slots": {},
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    rows = build_rows()
    if len(rows) != 160:
        raise RuntimeError(f"expected 160 cross-domain boundary rows, got {len(rows)}")
    if len({row["text"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate cross-domain boundary text detected")
    train_rows = [row for index, row in enumerate(rows, 1) if index % 5 != 0]
    eval_rows = [row for index, row in enumerate(rows, 1) if index % 5 == 0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / f"{PREFIX}_all.jsonl"
    train_path = OUTPUT_DIR / f"{PREFIX}_train.jsonl"
    eval_path = OUTPUT_DIR / f"{PREFIX}_eval.jsonl"
    probe_path = REPORT_DIR / f"{PREFIX}_probe.json"
    summary_path = REPORT_DIR / f"{PREFIX}_summary.json"

    _write_jsonl(all_path, rows)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    probe_path.write_text(
        json.dumps({"name": f"{PREFIX}_probe", "items": [_probe_row(row) for row in rows]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    domain_schema_counts = Counter(f"{row['domain']}:{row['schema']}" for row in rows)
    summary = {
        "prefix": PREFIX,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "domain_counts_all": dict(Counter(str(row["domain"]) for row in rows)),
        "schema_counts_all": dict(Counter(str(row["schema"]) for row in rows)),
        "domain_schema_counts_all": dict(domain_schema_counts),
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "eval": str(eval_path),
            "probe": str(probe_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
