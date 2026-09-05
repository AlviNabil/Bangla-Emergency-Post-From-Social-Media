# Bangla emergency post classification (9 classes; data is pulled from the Hub)
python train.py --model=xlm-roberta-base --task=social-media --batch-size=16 --lr=0.00001 --save-path='logs/social-media/xlm-base'

# other transformer backbones reported in the paper
# python train.py --model=bert-base-multilingual-uncased --task=social-media --batch-size=16 --lr=0.00003 --save-path='logs/social-media/mbert'
# python train.py --model=sagorsarker/bangla-bert-base --task=social-media --batch-size=16 --lr=0.00003 --save-path='logs/social-media/banglabert'
