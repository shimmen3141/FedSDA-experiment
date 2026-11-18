"""1標本の処理とrunのtestで使う、手で作った系列（2クラス、2特徴）。"""

CONCEPT_RULES = {
    0: lambda x: int(x[0] >= 3.0),
    1: lambda x: int(x[0] < 3.0),
    2: lambda x: int(x[1] >= 2.0),
}


def make_stream(segments):
    """segments: (コンセプトID, 件数)の列。(特徴の辞書, クラス, コンセプト)の列を返す。"""
    stream = []
    index = 0
    for concept_id, count in segments:
        for _ in range(count):
            x = {0: float(index % 7), 1: float((index * 3) % 5)}
            stream.append((x, CONCEPT_RULES[concept_id](x), concept_id))
            index += 1
    return stream
