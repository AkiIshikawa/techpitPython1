# クラス定義やパッケージのインポート
from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

# 変数の初期化（クラスラベルの定義と画像サイズの指定）
classes = ["cat","dog","monkey"]
num_classes = len(classes)
image_size = 50

# 画像の読み込み
X = []
Y = []

for index, classlabel in enumerate(classes):
    # ディレクトリ名を生成
    photos_dir = "./" + classlabel

    # ファイル一覧を取得
    files = glob.glob(photos_dir + "/*.jpg")

    # 各ファイルをNumPyアレーに変換し、リストに追加
    for i, file in enumerate(files):
        # filesから1個づつ取り出しfileに入れ付番

        if i >= 200: break
        image = Image.open(file)  # Imageクラスのopen関数でファイルをオープン
        image = image.convert("RGB")  # 配色データをRGBの順に揃える
        image = image.resize((image_size, image_size))  # 画像サイズを揃える
        data = np.asarray(image)  # NumPyアレーに変換
        X.append(data)  # リストXの末尾に追加
        Y.append(index)  # リストYの末尾に追加

# リスト型変数X,YをNumPyアレーに変換
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./animal.npy", xy)