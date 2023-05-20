import os
import requests
from flask import Flask, send_file, request, render_template, send_from_directory, current_app, redirect, url_for
from werkzeug.utils import secure_filename
import io
import glob
from inference import start

DOWNLOAD_FOLDER = './Scripts/LDR2HDR/images_LDR'
app = Flask(__name__)
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = './'

#file upload
@app.route('/uploader', methods = ['GET','POST'])
def upload_file():
    delete_file = glob.glob(f'./*/*/*/*.jpg')
    delete_file.extend(glob.glob(f'./*/*/*/*.exr'))
    delete_file.extend(glob.glob(f'./*/*/*/*.npy'))
    delete_file.extend(glob.glob(f'./*/*/*/*.hdr'))
    delete_file.extend(glob.glob(f'./*/*/*/*.json'))
    print(delete_file)
    for i in delete_file:
        os.remove(i) 
        pass
    print(f'finish clearing file:\n {delete_file} ')
    #改成用byte位元組來拚圖片就可以了
    f = request.files['file']
    img_buffer = io.BytesIO(f.read())
    img_data = img_buffer.getvalue()
    with open(DOWNLOAD_FOLDER+'/upload.jpg', 'wb') as f:
        f.write(img_data)
    start()
    return "success"

#file download        
@app.route('/download/<path:filename>', methods=['GET'])
def download_exr(filename):
    if request.method == 'GET':
        #uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'], filename)
        p = os.path.join(current_app.root_path, filename)
        path = os.path.isfile(p)
        print(p)
        if path:
            return send_from_directory(app.config['UPLOAD_FOLDER'] , path=filename, as_attachment=True)


# @app.route('/download/json/<path:filename>', methods=['GET'])
# def download_json(filename):
#     if request.method == 'GET':
#         #uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'], filename)
#         path = os.path.isfile(os.path.join(current_app.root_path, filename))
#         if path:
#             return send_from_directory(app.config['UPLOAD_FOLDER_json'], path=filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)