# colab을 vscode로 불러와서 진행할 때
apt-get update && apt-get upgrade
apt-get install build-essential cmake tree

mkdir cudnn
cd cudnn
tar -zxvf /content/drive/Shareddrives/DataAnalytics/cudnn-11.2-linux-x64-v8.1.1.33.tgz

cp cudnn/cuda/include/cudnn.h /usr/local/cuda/include
cp cudnn/cuda/lib64/libcudnn* /usr/local/cuda/lib64

cd ../
chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
apt-get install libcupti-dev
# install한 내역 반영
apt-get update
rm -rf cudnn

echo 'export PATH="$PATH:/usr/local/cuda/bin"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"' >> ~/.bashrc

# 일단 이렇게 해보고 너무 구조가 정리가 되지 않은 상태이거나, 오작동 시 최상위에 프로젝트 폴더 생성하는 것으로 바꾸기
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
# pip install --user poetry

echo 'export PATH="$HOME/.poetry/bin:$PATH"' >> ~/.bashrc
source $HOME/.poetry/env
poetry completions bash > /etc/bash_completion.d/poetry.bash-completion
source $HOME/.bashrc

# mkdir ai-challenge
# cd ai-challenge

# 모듈 전체 정리가 된다면???
# poetry new ai-challenge
# # ai-challenge
# # ├── README.rst
# # ├── ai_challenge
# # │   └── __init__.py
# # ├── pyproject.toml
# # └── tests
# #     ├── __init__.py
#     └── test_ai_challenge.py
cp /content/drive/Shareddrives/DataAnalytics/Dockerfile .
