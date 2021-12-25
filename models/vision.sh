# torch.vision 필요 소스 클론 후 배치
git clone https://github.com/pytorch/vision.git
mkdir PTV
cp vision/references/detection/* PTV/
rm -rf vision
cd ../
