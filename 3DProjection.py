from PIL import Image as PImage
import plotly.graph_objects as go
import numpy as np
import skimage.io

d = skimage.io.imread('C:/Data/HeatMaps/nyu_data/data/nyu2_train/classroom_0003_out/1.png')
img = skimage.io.imread('C:/Data/HeatMaps/nyu_data/data/nyu2_train/classroom_0003_out/1.jpg')

def create_rgb_surface(rgb_img, depth_img, **kwargs):
    rgb_img = rgb_img.swapaxes(0, 1)[:, ::-1]
    depth_img = depth_img.swapaxes(0, 1)[:, ::-1]
    eight_bit_img = PImage.fromarray(rgb_img).convert('P', palette='ADAPTIVE', dither=None)
    idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    depth_map = depth_img.copy().astype('float')*5
    return go.Surface(
        z=depth_map,
        surfacecolor=np.array(eight_bit_img),
        cmin=0,
        cmax=255,
        colorscale=colorscale,
        showscale=False,
        **kwargs
    )

d = np.flipud(d)
img = np.flipud(img)


fig = go.Figure(
    data=[create_rgb_surface(img,
                             1 - d,
                             contours_z=dict(show=True, project_z=True, highlightcolor="limegreen"),
                             opacity=1.0,
                            ),
                            ],
    layout_title_text="3D Surface",
)
camera = dict(
    eye=dict(x=0.001, y=0, z=6)
)

fig.update_layout(scene_camera=camera)

fig.update_layout(
    scene = dict(
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        zaxis =dict(visible=False)
        )
    )
fig.show()
