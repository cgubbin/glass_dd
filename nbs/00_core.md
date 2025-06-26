---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: python3
    language: python
    name: python3
---

# core

> This module aims to classify defects observed on glass surfaces.

```python
from fastai.vision.all import *

def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
def_device = 'cpu'

device = torch.device(def_device)


path = Path("/Users/kit/Resource/Data/defect/AGDD/data/image/")
path.ls()
```

```python
img_files = get_image_files(path)
def img2label(x): 
    name = Path(x.name).stem

    return x.parents[2]/'labels_rect'/Path(x).parent.name/f"{name}.txt"
img2label(img_files[0])
```

```python
im = PILImage.create(img_files[0])
print(f'image shape: {im.shape}')
im.to_thumb(250)
```
```python
import PIL
import torchvision.transforms as T

ident = {
        0: "contusion",
        1: "scratch",
        2: "crack",
        3: "spot",
}
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def get_x(x):
    image = PILImage.create(x)
    tensor = T.ToTensor()(image)
    tensor.to(device)
    return tensor

# Convert from (centre_x, centre_y, width, height) to
# (upper_left_y, upper_left_x, lower_right_y, lower_right_x)
def get_bounding_box(f):
    bb = np.genfromtxt(img2label(f))

    if bb.ndim == 0:
        return None
    elif bb.ndim == 1:
        bb = np.expand_dims(bb, axis=0)

    bb = bb[:, 1:]

    for ii in range(bb.shape[0]):
        bb[ii] = box_cxcywh_to_xyxy(tensor(bb[ii]))

    im_width, im_height = PIL.Image.open(f).size


    bb[:, 0] *= im_width
    bb[:, 2] *= im_width
    bb[:, 1] *= im_height
    bb[:, 3] *= im_height

    return tensor(bb.astype(int), device=device)

def get_label(f):
    label = np.genfromtxt(img2label(f))

    if label.ndim == 0:
        return None
    elif label.ndim == 1:
        return [ident[label[0]]]
    # return tensor([c1, c2])
    # Scale centers on [-1, +1]

    return [ident[each] for each in label[:, 0]]

get_y = [get_bounding_box, get_label]

for ii in range(2):
    print(ii)
    print(get_bounding_box(img_files[ii]))
    print(get_label(img_files[ii]))
```
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ii = 0

bb = get_bounding_box(img_files[ii])
im = PIL.Image.open(img_files[ii])

fig, ax = plt.subplots()

ax.imshow(im)

for each in bb:
    each = each.cpu()
    height = each[1] - each[3]
    width = each[2] - each[0]
    rect = patches.Rectangle((each[0], each[1] - height), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
```

```python
topside = DataBlock(
    blocks = (ImageBlock, BBoxBlock, BBoxLblBlock),
    n_inp=1,
    get_items = get_image_files,
    get_y = get_y,
    splitter = FuncSplitter(lambda s: Path(s).parent.name == 'val'),
    batch_tfms=[*aug_transforms(size=(240,320)),
                Normalize.from_stats(*imagenet_stats)]
)
#| default_exp core
```
```python
topside.summary(path, device=device)
```

```python
dls = topside.dataloaders(path, device=def_device)
print(default_device())
dls.show_batch(max_n=3, figsize=(15, 12))
```

```python
dls = topside.dataloaders(path, bs=16)
xb,yb,lb = dls.one_batch()
print(xb.shape,yb.shape,lb.shape)
```
```python
learn = detr_learner(dls)
```

```python
learn.lr_find()
```

```python
from wwf.vision.object_detection import *

encoder = create_body(resnet18(), pretrained=True) #, pretrained=True)
encoder.to(device)
```

```python
get_c(dls)
```
```python
arch = RetinaNet(encoder, get_c(dls), final_bias=-4)
arch.to(device)
arch.smoothers, arch.classifier
```
```python
ratios = [1/2,1,2]
scales = [1,2**(-1/3), 2**(-2/3)]
```

```python
crit = RetinaNetFocalLoss(scales=scales, ratios=ratios)
crit.to(device)
def _retinanet_split(m): return L(m.encoder,nn.Sequential(m.c5top6, m.p6top7, m.merges, m.smoothers, m.classifier, m.box_regressor)).map(params)
```
```python
learn = Learner(dls, arch, loss_func=crit)#, splitter=_retinanet_split)
learn.to(device)
print(learn)
```

```python
learn.lr_find()
```
```python
learn.freeze()
```
```python
learn.fit_one_cycle(10, slice(1e-5, 1e-4))
```

```python
learn.dls.device = 'mps:0'
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
```

```python
#| hide
from nbdev.showdoc import *
```

```python
#| export
def foo(): pass
```

```python
#| hide
import nbdev; nbdev.nbdev_export()
```
