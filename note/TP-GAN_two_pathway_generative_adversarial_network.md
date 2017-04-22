# Beyond Face Rotaion: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis
## TP-GAN (Two-Pathway Generative Adversarial Network)

### タスク
人の横顔から正面の顔を生成するGAN(Fig.1)

<p align="center">
<img src="fig/TPGAN_fig1.jpg" width=500px>
</p>

### 構成
個別のパーツを生成するGeneratorと、輪郭を生成するGeneratorをパラで用意し、
それぞれから生成された画像を合成して正面の画像にする。

<p align="center">
<img src="fig/TPGAN_fig2_framework.jpg" width=800px>
</p>

### Lossの定義
それぞれのGeneratorはCNNのEncoderとDecoderより構成されており、
式1よりGeneratorのパラメータθを最適化する。

<p align="center">
<img align="center" src="fig/TPGAN_eq1.jpg" width=400>
</p>

* Grobal pathwayのエンコーダがボトルネックになっているらしく、個別のロスを定義している。αL_closs_entropy....


* L_synのロスは以下の5つのロスから構成される。

<p align="center">
<img src="fig/TPGAN_eq7.jpg" width=400>
</p>

1. Pixel上でのロス
Ground truthと生成された画像とのピクセルごとの差分

<p align="center">
<img src="fig/TPGAN_eq3.jpg" width=400>
</p>

2. 対象性ロス
生成された画像が左右対称かどうかをピクセルごとに計算

<p align="center">
<img src="fig/TPGAN_eq4.jpg" width=400>
</p>

3. Adversarialロス
GANのいつものロス（うまく騙せたかどうか）

<p align="center">
<img src="fig/TPGAN_eq5.jpg" width=400>
</p>

4. Identity Preservingロス
Light CNNの最後の2層を使い、
横顔を入力した結果と生成された画像を入力した結果との差。

<p align="center">
<img src="fig/TPGAN_eq6.jpg" width=400>
</p>

5. Total Variation regularization
よくCNNで使われるロス。画像を滑らかにする制約になる。

### 結果
各角度からの生成で認識性能がSOTA(Fig.4, Table 1)

<p align="center">
<img src="fig/TPGAN_fig4.jpg" width=800px>
<img src="fig/TPGAN_table1.jpg" width=500px>
</p>
