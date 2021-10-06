from transformer import __version__
# 상대 경로를 사용하는 법, 모듈 다루는 라이브러리로 접근하는 방법, .toml/.txt/.in등 다양함

def test_version():
    assert __version__ == '0.1.0'
