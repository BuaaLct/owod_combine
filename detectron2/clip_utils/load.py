import itertools
import clip
import torch

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant", "sofa", "tvmonitor"
]

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl"
]

UNK_CLASS = ["object"]
BG_CLASS = ["background"]
# 实例的类别的名称
VOC_COCO_CLASS_NAMES = tuple(
    itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS,BG_CLASS))

SUPER_CLASS_NAMES = tuple(
    itertools.chain(['outdoor','accessories','appliance','truck','sports','food','electronic','indoor','kitchen','furniture'])
)

class ClipProcess:
    def __init__(self, class_names=VOC_COCO_CLASS_NAMES, name="/lct/clip/OWOD-master/ViT-B-32.pt"):
        self.class_names = class_names
        self.name = name
        self.super_class_name = SUPER_CLASS_NAMES

        self.cuda_clip_model = None
        self.cuda_clip_preprocess = None
        self.cuda_text_features = None
        self.cuda_text_token = None

        self.cpu_clip_model = None
        self.cpu_clip_preprocess = None
        self.cpu_text_features = None
        self.cpu_text_token = None

        self._load()
        self.init()

        self.super_cls_list=[{0,1,3,5,6,13,18},{2,7,9,11,12,16},{14},{4,8,10,15,17,19},{20},{21}]

    def _load(self):
        self.cuda_clip_model, self.cuda_clip_preprocess = clip.load(self.name, device='cuda')
        self.cuda_clip_model, self.cuda_clip_preprocess = clip.load(self.name, device='cpu')

    def init(self):
        text_token = clip.tokenize(self.class_names)
        super_text_token = clip.tokenize(self.super_class_name)
        with torch.no_grad():
            self.cuda_text_features = self.cuda_clip_model.encode_text(text_token)
            self.cpu_text_features = self.cuda_clip_model.encode_text(text_token)
            self.cuda_super_text_token = torch.rand((10,512))
            self.cpu_super_text_token = self.cuda_super_text_token

    def get_super_features(self, device):
        if device == 'cuda':
            return self.cuda_super_text_token
        else:
            return self.cpu_super_text_token

    def get_text_features(self, device):
        if device == 'cuda':
            return self.cuda_text_features
        else:
            return self.cpu_text_features
