# torch.vision 필요 소스  클론 후 배치
git clone https://github.com/pytorch/vision.git

# 하위 폴더를 생성한다.
mkdir PTV

# PTV에 vision 클론한 부분을 넣는다.
cp vision/references/detecion/* PTV/

# vision의 클론한 부분을 제거한다.
rm -rf vision

# 상위 폴더로 빠져 나온다.
cd ../