# 프로젝트 작성 가이드

## General Python Architecture

깃허브에서 확인할 수 있는 파이썬 프로젝트 구조를 살펴보면 다음과 같습니다.

```
```bash
example-project
├── LICENSE
├── README.md
├── docs
│   ├── examples
│   └── papers
│       └── cnn_papers.md
├── images
├── requirements.txt
├── rsc
├── src
│   ├── detection
│   │   └── __init__.py
│   └── transformer
│       └── __init__.py
└── tests
    ├── test_detection.py
    └── test_transformer.py
```

## Which env should we install?

venv, virtualenvwrapper, pyenv, pyenv-virtualenv, pyenv-virtualenvwrapper, poetry, autoenv 외에도 가상환경 내에서 버전 컨트롤을 하고 패키지를 관리하는 툴은 다양하게 있습니다.

기본 버전으로 접근하는 것도 하나의 방법이나, 제 OS에서는 pyenv+poetry로 관리하는 게 나을 것이란 칼럼을 본 뒤 해당 방법을 사용하고 있습니다. 다만, OS 업데이트 뒤에 발생하는 오류를 겪을 때마다 코랩 최고를 외치게 됩니다.

코랩에서도 버전 컨트롤을 하는 방법은 알고 있지만, 로컬에서 서버에 접근하는 방식으로 진행될 것 같아서 실습 코드는 코랩에서 작성한 뒤 .py파일로 pull-request를 주시면 보다 편할 것 같습니다.

논문 리뷰를 하면서, 각자가 편한 환경에서 버전 컨트롤을 연습할 방법을 정리하려고 했으나, 🐛는 항상 시간을 부족하게 만드네요ㅎㅎ

어떤 가상환경을 사용해야 하는가는 운영체제나 상황, 로컬 스펙에 따라 달라지는 부분이 크기 때문에 case-by-case라고 할 수 있습니다.

## pyenv install available

* python(org, dev)
* activepython
* anaconda
* graalpython
* ~~ironpython(org, dev)~~
* ~~jython(org, dev)~~
* ~~mambaforge(org, org+pypy)~~
* micropython(org, dev)
* miniconda
* miniforge(org, org-pypy3)
* pypy(c-jit-latest, stm, src)
* pyston
* stackless

dev와 release의 차이는?

pypy3가 같이 있는 release는 무슨 이유로 그러한가?

c-jit-latest가 작성된 pypy는 연산 효율이 더 좋은가?

등의 의문이 들었지만, 각 가상환경을 다 적용할 수 없기도 하기에 우선은 제가 사용하는 방식으로 확인 및 사용에 대한 예시를 보이고자 합니다.

## pyenv + pyenv-virtualenv + poetry

```
# pyenv install list 확인
$ pyenv install -l

# install
$ pyenv install {version-name}

# show installed version
$ pyenv versions

```

로 local과 global의 버전을 확인할 수 있습니다.

```
  system
* 3.7.7 (set by /Users/ashbee/.pyenv/version)
  3.7.7/envs/ai-challenge
  3.9.0
  3.9.0/envs/hackathon
  3.9.7
  3.9.7/envs/develop-domains
  ai-challenge
  develop-domains
  hackathon
```


```
# pyenv virtualenv {python-version} {environment_name}

# pyenv activate로 경로 연결되는지 확인하고 deactivate
# poetry 설치 후 경로 설정
$ poetry config virtualenvs.in-proeject true

# poetry 가상환경 연결과 호출은 운영체제별 차이가 있기에 생략합니다.
# poetry new {project-name}으로 디렉토리를 생성하거나, 프로젝트 폴더를 생성해줍니다.
$ poetry init
# 모두 enter를 하거나, yes를 택하는 등 상세 설정은 원하는 방식으로 하시면 됩니다.
```



```
# 가상환경 활성화
$ poetry shell

# 라이브러리 추가
$ poetry add {library-name}

# poetry.lock에 깃헙 등의 주소를 추가하거나, 자체적으로 구상한 module을 경로로 적어 update할 수 있습니다.
$ poetry update

# pip show list와 유사하나, 의존성 확인 가능함
$ poetry show --tree
```

tree로 확인 가능한 부분이 poetry 사용 결정에 큰 요인이었습니다.

```
asyncio 3.4.3 reference implementation of PEP 3156
bs4 0.0.1 Dummy package for Beautiful Soup
└── beautifulsoup4 *
    └── soupsieve >1.2 
discord.py 1.7.3 A Python wrapper for the Discord API
└── aiohttp >=3.6.0,<3.8.0
    ├── async-timeout >=3.0,<4.0 
    ├── attrs >=17.3.0 
    ├── chardet >=2.0,<5.0 
    ├── multidict >=4.5,<7.0 
    ├── typing-extensions >=3.6.5 
    └── yarl >=1.0,<2.0 
        ├── idna >=2.0 
        └── multidict >=4.0 (circular dependency aborted here)
```


```
# 가상환경 정보 확인
$ poetry env info
```

```$
Virtualenv
Python:         3.7.7
Implementation: CPython
Path:           $HOME/.pyenv/versions/3.7.7/envs/{environment-name}
Valid:          True

System
Platform: darwin
OS:       posix
Python:   $HOME/.pyenv/versions/3.7.7
```



다만, 패키지 관리툴은 poetry 말고도 많고, 굳이 필요가 없다고 생각하시면 기본 가상환경 설정을 하는 방법도 괜찮다고 생각합니다. 특히, OS, Programming Language, Virtual Environment에 따라 발생하는 에러가 다양하고 많기 때문에, github에 올라오는 issues tracking을 해야할 수도 있습니다. 따라서 도전 의식이 높은 편이거나 주말에 코딩하고 싶다면 이번 기회에 해보는 것도 좋은 경험이 될 것 같습니다.
