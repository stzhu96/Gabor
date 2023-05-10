import cv2
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import base64

app = dash.Dash()

app.layout = html.Div([
    html.H1("Gabor特征展示"),
    html.Div([
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                '拖放或 ',
                html.A('选择图片')
            ]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='output-image-upload'),
        html.Div(id='output-mean-stddev')
    ])
])

# 解码图片并展示
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = np.asarray(bytearray(decoded), dtype="uint8")
    img_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # 定义Gabor滤波器参数
    ksize = 31
    sigma = 4.0
    theta = 0.5 * np.pi
    Lambda = 10.0
    gamma = 0.5
    psi = 0.0
    ktype = cv2.CV_32F
    # 创建Gabor滤波器
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, Lambda, gamma, psi, ktype)
    # 对图像进行Gabor滤波并计算平均值和方差
    gabor_img = cv2.filter2D(img_bgr, cv2.CV_32F, kernel)
    mean, stddev = cv2.meanStdDev(gabor_img)
    mean_row = mean.ravel()
    stddev_row =stddev.ravel()

    # 将图像转换为base64编码，并展示在网页上
    ret, buffer = cv2.imencode('.jpg', img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    img_src = f"data:image/jpg;base64,{img_base64}"
    return html.Div([
        html.H2("原始图像"),
        html.Img(src=img_src,style={'width': '400px'}),
        html.H2("Gabor纹理特征均值和标准差"),
        html.P(f"Gabor纹理B\G\R通道均值: {mean_row[0]:.2f}, {mean_row[1]:.2f}, {mean_row[2]:.2f}"),
        html.P(f"Gabor纹理B\G\R通道标准差: {stddev_row[0]:.2f}, {stddev_row[1]:.2f}, {stddev_row[2]:.2f}")
    ])

# 回调函数，解码图片并展示
@app.callback(Output('output-image-upload', 'children'),
              Output('output-mean-stddev', 'children'),
              Input('upload-image', 'contents'))
def update_output(contents):
    if contents is not None:
        children = parse_contents(contents)
        return children, None
    else:
        return None, None

if __name__ == '__main__':
    app.run_server(debug=True,port=804)
