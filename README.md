# 1 What's Gist feature?  Gist特征是什么?

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1) The feature description of a image scenes in macro meaning. 一种宏观意义的场景特征描述

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2) Only detect scenes ——"There are some people in the street".
Don't care about how many people and others.
 只识别“大街上有一些行人”这个场景，无需知道图像中在那些位置有多少人，或者有其他什么对象。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3) Gist Vector could represent the macro meaning of the image. 特征向量可以一定程度表征这种宏观场景特征

There are five spatial envelope name of GIST. GIST中有五种对空间包络的描述方法

|Spatial envelope name 空间包络名|Explanation 阐释|
|--|--|
|Degree of Naturalness 自然度|If the scene contains high levels of horizontal and vertical lines, this indicates that the scene has obvious artificial traces, usually natural scenes have textured areas and undulating outlines. Therefore, the edge has a natural tendency that the height is perpendicular to the horizontal, and the natural degree is high. 场景如果包含高度的水平和垂直线，这表明该场景有明显的人工痕迹，通常自然景象具有纹理区域和起伏的轮廓。所以，边缘具有高度垂直于水平倾向的自然度低，反之自然度高。|
|Degree of Openness 开放度|Open or Close? Close: forest, mountain, center of city. Open:coast, high speed road. 空间包络是否是封闭（或围绕）的。封闭的，例如：森林、山、城市中心。或者是广阔的，开放的，例如：海岸、高速公路。|
|Degree of Roughness 粗糙度|Mainly refers to the particle size of the main components. It depends on the size of the elements in each space, their possibility to construct more complex elements, and the structural relationship between the constructed elements, etc. Roughness is related to the fractal dimension of the scene, so it can be called complexity. 主要指主要构成成分的颗粒大小。这取决于每个空间中元素的尺寸，他们构建更加复杂的元素的可能性，以及构建的元素之间的结构关系等等。粗糙度与场景的分形维度有关，所以可以叫复杂度。|
|Degree of Expansion 膨胀度|The parallel lines converge, giving the depth characteristics of the spatial gradient. For example, buildings in plan view have a low degree of expansion. In contrast, very long streets have a high degree of expansion. 平行线收敛，给出了空间梯度的深度特点。例如平面视图中的建筑物，具有低膨胀度。相反，非常长的街道则具有高膨胀度。|
|Degree of Ruggedness 险峻度|That is offset from the horizontal. (For example, a mountain landscape on a flat horizontal ground and a steep ground). In the precipitous environment, the slanted outline is produced in the picture, and the horizon line is hidden. Most man-made environments establish flat ground. Therefore, the precarious environment is mostly natural. 即相对于水平线的偏移。（例如，平坦的水平地面上的山地景观与陡峭的地面）。险峻的环境下在图片中生产倾斜的轮廓，并隐藏了地平线线。大多数的人造环境建立了平坦地面。因此，险峻的环境大多是自然的。|

# 2 Implement of Gist--LMgist 

* [LMgist Matlab Code](http://people.csail.mit.edu/torralba/code/spatialenvelope/LMgist.m)

* LMgist Matlab Usage

```
% Read image 
img = imread('demo2.jpg');

% Set parameters of GIST
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing GIST
[gist, param] = LMgist(img, '', param);
```

# 3 Python Implement of LMgist

# 3.1 Extract Gist feature 抽取Gist特征

```
import cv2
from img_gist_feature.utils_gist import *

s_img_url = "./test/A.jpg"
gist_helper = GistUtils()

np_img = cv2.imread(s_img_url, -1)

print("default: rgb")
np_gist = gist_helper.get_gist_vec(np_img)
print("shape ", np_gist.shape)
print("noly show 10dim", np_gist[0,:10], "...")
print()

print("convert rgb image")
np_gist = gist_helper.get_gist_vec(np_img, mode="rgb")
print("shape ", np_gist.shape)
print("noly show 10dim", np_gist[0,:10], "...")
print()

print("convert gray image")
np_gist = gist_helper.get_gist_vec(np_img, mode="gray")
print("shape ", np_gist.shape)
print("noly show 10dim", np_gist[0,:10], "...")
print()

```
Result:

default: rgb
shape  (1, 1536)
noly show 10dim [0.02520592 0.05272802 0.05941689 0.05476999 0.13110509 0.13333975
 0.29072759 0.16522023 0.25032277 0.36850457] ...

convert rgb image
shape  (1, 1536)
noly show 10dim [0.02520592 0.05272802 0.05941689 0.05476999 0.13110509 0.13333975
 0.29072759 0.16522023 0.25032277 0.36850457] ...

convert gray image
shape  (1, 512)
noly show 10dim [0.10004389 0.20628179 0.17682694 0.16277722 0.10557428 0.14448622
 0.29214159 0.11260066 0.16488087 0.28381876] ...




# 3.2 Get cosine similarity of Gist feature

Please run python _test_get_cossim.py

<center>
<img src="./README/01.jpg"><br>
</center>

<center>
<img src="./README/02.jpg"><br>
</center>

# 4 Theory of LMgist

## 4.1 Main process of LMgist

* G1:Do preprocess of input image 

* G2:Do Prefilt of input image

* G3:Compute Gist vector of input image

## 4.2 G2: Do Prefilt of input image

### 4.2.1 Pad images to reduce boundary artifacts  (扩边+去伪影)

$${\bf{matlog}} = \log \left( {{\bf{mat}} + 1} \right)$$
$${\bf{matPad}} = {\mathop{\rm sympading}\nolimits} \left( {{\bf{matlog}},\left[ {5,5,5,5} \right]} \right)$$

<center>
<img src="./README/03.jpg"><br>
Figure.1 sympading
</center>

### 4.2.2 Filter  (构造滤波器)

<center>
<img src="./README/04.jpg"><br>
</center>

<center>
<img src="./README/05.jpg"><br>
</center>


$${\bf{matGf}} = {\mathop{\rm FFTSHITF}\nolimits} \left( {\exp \left( { - \frac{{{\bf{matF}}{{\bf{x}}^2} + {\bf{matF}}{{\bf{y}}^2}}}{{{{\left( {\frac{{fc}}{{\sqrt {\log \left( 2 \right)} }}} \right)}^2}}}} \right)} \right)$$


### 4.2.3 Whitening  (白化)

$${\bf{matRes}} = {\bf{matPad}} - {\mathop{\rm Real}\nolimits} \left( {{\mathop{\rm IFFT}\nolimits} \left( {{\mathop{\rm FFT}\nolimits} \left( {{\bf{matPad}}} \right){\bf{matGf}}} \right)} \right)$$


### 4.2.4 Local contrast normalization (局部对比度归一化）

$${\bf{matLocal}} = \sqrt {\left| {{\mathop{\rm IFFT}\nolimits} \left( {{\mathop{\rm FFT}\nolimits} \left( {{\bf{matRes}} \cdot {\bf{matRes}}} \right) \cdot {\bf{matGf}}} \right)} \right|} $$

$$ {\bf{matRes}} = \frac{{{\bf{matRes}}}}{{0.2 + {\bf{matLocal}}}} $$



### 4.2.5 Local contrast normalization (局部对比度归一化）

$${\bf{matPrefilt = matRes}}\left[ {5:64 + 5,5:64 + 5} \right]$$

## 4.3 G3: Compute Gist vector of input image

### 4.3.1 Pading  

$${\bf{matPad}} = {\mathop{\rm sympading}\nolimits} \left( {{\bf{matPrefilt}},\left[ {32,32,32,32} \right]} \right)$$

### 4.3.2 FFT

$${\bf{matLocal}} = {\mathop{\rm FFT}\nolimits} \left( {{\bf{matPad}}} \right)$$

### 4.3.3 Gabor Filtering

<center>
<img src="./README/06.jpg"><br>
</center>

<center>
<img src="./README/07.jpg"><br>
Figure2 Get Gist feature
</center>


# Reference

* https://www.cnblogs.com/justany/archive/2012/12/06/2804211.html
* https://blog.csdn.net/qq_16234613/article/details/78909839

