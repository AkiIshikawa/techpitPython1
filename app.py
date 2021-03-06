import os
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import keras,sys
import numpy as np
from PIL import Image

import os
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import keras,sys
import numpy as np
from PIL import Image



classes = ["cat","dog","monkey"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])


app = Flask(__name__) #Flaskサーバーインスタンスを生成
app.secret_key = "12345" #秘密鍵を設定（任意）
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER #アプリの設定に追加


def allowed_file(filename): #ファイル名をチェックする関数
    # ファイル名がOKなら、拡張子を小文字にしてパスを返す
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST']) #ルートURLの定義
def upload_file(): #ルートが呼ばれた時に呼ばれる関数
    if request.method == 'POST': #データがブラウザからPOSTで送信された場合
        if 'file' not in request.files: #POSTリクエストにファイルが無い場合
            flash('ファイルがありません') #エラーを吐く
            return redirect(request.url) #元のページに戻す
        file = request.files['file'] #ファイルをリクエストから取得
        if file.filename == '': #ファイルが空なら
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) #ファイルをアップロードフォルダーに保存
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) #ファイルパスを生成

            model = load_model('./animal_cnn_aug.h5')  # 学習済みモデルをファイルから読み込む

            image = Image.open(filepath)  # ファイルを開く
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))  # サイズを揃える
            data = np.asarray(image) / 255  # 正規化
            X = []  # Xを空のリストとして初期化
            X.append(data)  # XにNumPy配列にしたdataを追加
            X = np.array(X)  # NumPyアレーに型変換

            result = model.predict([X])[0]
            predicted = result.argmax()  # ハイスコアの添字を取得
            percentage = int(result[predicted] * 100)  # 推定確率をパーセント形式に

            return "ラベル： " + classes[predicted] + ", 確率：" + str(percentage) + " %"

    return '''
        <!doctype html>
        <html>
        <head>
        <meta charset="UTF-8">
        <title>ファイルをアップロードして判定しよう</title></head>
        <body>
        <h1>ファイルをアップロードして判定しよう！</h1>

        <form method = post enctype = multipart/form-data>
        <p><input type=file name=file>
        <input type=submit value=Upload>
        </form>
        </body>
        </html>
        '''


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))