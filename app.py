import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import tempfile
import time
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io
import PIL.ExifTags
import PIL.Image
import numpy as np
import os
import json

st.session_state.SCALE = 0.5
st.session_state.FOCAL_LENGTH = 3979.911
st.session_state.X_A = 1244.772
st.session_state.X_B = 1369.115
st.session_state.Y = 1019.507
st.session_state.DOFFS = st.session_state.X_B - st.session_state.X_A
st.session_state.CAMERA_DISTANCE = 193.001
st.session_state.block_size = 5

st.set_page_config(
    page_title="Stereo Vision to 3D Point Cloud",
    page_icon="🎥",
    layout="wide"
)

# Load image function
def load_image(image_path):
    img = Image.open(image_path)
    return np.array(img)

# Downsample image function
def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape
        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image

def generate_window(row, col, image, blockSize):
    window = (image[row:row + blockSize, col:col + blockSize])
    return window

def disparitymap(imgL, imgR, dispMap=[]):
    # Size of the search window 
    blockSize = st.session_state.block_size
    print(blockSize)
    h, w = imgL.shape
    dispMap = np.zeros((h, w))
    # maximum disparity to search for (Tuned by experimenting)
    max_disp = int(w // 2)
    dispVal = 0
    tic = time.time()

    progress_bar = st.progress(0) 
    progress_text = st.empty()

    total_steps = (h - blockSize + 1) // blockSize
    completed_steps = 0

    for row in range(0, h - blockSize + 1, blockSize):
        for col in range(0, w - blockSize + 1, blockSize):
            winR = generate_window(row, col, imgR, blockSize)
            sad = 9999
            dispVal = 0
            for colL in range(col + blockSize, min(w - blockSize, col + max_disp)):
                winL = generate_window(row, colL, imgL, blockSize)
                tempSad = int(abs(winR - winL).sum())
                if tempSad < sad:
                    sad = tempSad
                    dispVal = abs(colL - col)
            for i in range(row, row + blockSize):
                for j in range(col, col + blockSize):
                    dispMap[i, j] = dispVal

        completed_steps += 1
        progress_percentage = int((completed_steps / total_steps) * 100)
        progress_bar.progress(progress_percentage)
        progress_text.text(f"Processing row {row}... {progress_percentage}% complete")
        
        if (row % 50 == 0):
            print('Row number {} Percent complete {} %'.format(row, row * 100 / h))
   
    progress_bar.progress(100)
    progress_text.text("Disparity map computation complete!")
    toc = time.time()
    print('elapsed time... {} mins'.format((toc - tic) / 60))

    fig, ax = plt.subplots()
    ax.set_title('Disparity Map')
    ax.set_ylabel(f'Height {dispMap.shape[0]}')
    ax.set_xlabel(f'Width {dispMap.shape[1]}')
    ax.imshow(dispMap, cmap='gray')

    st.pyplot(fig)
    return dispMap

def depth_map(dispMap, orignal_pic):
    print("Calculating depth....")
    depth = np.zeros(dispMap.shape)
    coordinates = []
    h, w = dispMap.shape
    for r in range(0, h):
        for c in range(0, w):
            disparity = dispMap[r, c]
            Yoffset = ((h - r) * 2) - st.session_state.Y
            Xoffset = ((w - c) * 2) - st.session_state.X_A
            depth[r, c] = (st.session_state.CAMERA_DISTANCE * st.session_state.FOCAL_LENGTH) / (dispMap[r, c])
            ZZ = (st.session_state.CAMERA_DISTANCE * st.session_state.FOCAL_LENGTH) / (disparity + st.session_state.DOFFS)
            YY = (ZZ / st.session_state.FOCAL_LENGTH) * Yoffset
            XX = (ZZ / st.session_state.FOCAL_LENGTH) * Xoffset
            coordinates += [[XX, YY, ZZ, orignal_pic[r][c][2], orignal_pic[r][c][1], orignal_pic[r][c][0]]]
    
    fig, ax = plt.subplots()
    depthmap = ax.imshow(depth, cmap='jet_r')
    plt.colorbar(depthmap, ax=ax)
    ax.set_title("Depth Map")
    st.pyplot(fig)
    
    return coordinates

# Create .ply file from coordinates
def create_output(vertices, filename):
	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

# Convert matplotlib figure to image
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)

def Image_processing(image_location):
    img= color.rgb2gray(io.imread(image_location))
    img = rescale(img, 0.5, anti_aliasing=False)

    return img

# Function to parse constants from a JSON file
def parse_constants(file):
    constants = {}
    try:
    
        file_content = json.load(file)
        
        if isinstance(file_content, dict):
            constants = file_content
            if 'SCALE' in file_content and file_content['SCALE'] is not None:
                st.session_state.SCALE = file_content['SCALE']
                
            if 'FOCAL_LENGTH' in file_content and file_content['FOCAL_LENGTH'] is not None:
                st.session_state.FOCAL_LENGTH = file_content['FOCAL_LENGTH']
                
            if 'X_A' in file_content and file_content['X_A'] is not None:
                st.session_state.X_A = file_content['X_A']
                
            if 'X_B' in file_content and file_content['X_B'] is not None:
                st.session_state.X_B = file_content['X_B']
                
            if 'Y' in file_content and file_content['Y'] is not None:
                st.session_state.Y = file_content['Y']
                
            st.session_state.DOFFS = st.session_state.X_B - st.session_state.X_A if 'DOFFS' not in file_content else file_content['DOFFS']

            if 'CAMERA_DISTANCE' in file_content and file_content['CAMERA_DISTANCE'] is not None:
                st.session_state.CAMERA_DISTANCE = file_content['CAMERA_DISTANCE']

        else:
            st.error("Invalid JSON structure. Expected a dictionary of constants.")
        return constants

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return {}
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return {}

def get_user_inputs():
    st.sidebar.title("Stereo Vision Parameters")
    
    # Load constants from file, if provided
    constants_file = st.sidebar.file_uploader("Upload Constants File (JSON)", type=["json"])
    constants = {}

    if constants_file is not None:
        st.sidebar.success("File uploaded. Parsing constants...")
        constants = parse_constants(constants_file)

    # Ensure constants are a dictionary even if no file is uploaded
    if not constants:
        constants = {}

    st.sidebar.warning("Provide constants manually if not in the file.")

    st.session_state.block_size = constants.get(
        "BLOCK_SIZE",
        st.sidebar.slider("Block Size", min_value=3, max_value=25, value=st.session_state.get("block_size", 5), step=2)
    )

    st.session_state.SCALE = constants.get(
        "SCALE",
        st.sidebar.number_input("Scale", value=st.session_state.get("SCALE", 0.5), step=0.1)
    )

    st.session_state.FOCAL_LENGTH = constants.get(
        "FOCAL_LENGTH",
        st.sidebar.number_input("Focal Length", value=st.session_state.get("FOCAL_LENGTH", 3979.911), step=0.1)
    )

    st.session_state.X_A = constants.get(
        "X_A",
        st.sidebar.number_input("X_A", value=st.session_state.get("X_A", 1244.772), step=0.1)
    )

    st.session_state.X_B = constants.get(
        "X_B",
        st.sidebar.number_input("X_B", value=st.session_state.get("X_B", 1369.115), step=0.1)
    )

    st.session_state.Y = constants.get(
        "Y",
        st.sidebar.number_input("Y Offset", value=st.session_state.get("Y", 1019.507), step=0.1)
    )

    st.session_state.CAMERA_DISTANCE = constants.get(
        "CAMERA_DISTANCE",
        st.sidebar.number_input("Camera Distance", value=st.session_state.get("CAMERA_DISTANCE", 193.001), step=0.1)
    )

    constants["DOFFS"] = constants.get("DOFFS", float(st.session_state.get("DOFFS", st.session_state.X_B - st.session_state.X_A)))
    st.sidebar.number_input("DOFFS", value=constants["DOFFS"], step=0.1)


    # Populate constants dictionary from session_state
    constants = {
        "BLOCK_SIZE": st.session_state.block_size,
        "SCALE": st.session_state.SCALE,
        "FOCAL_LENGTH": st.session_state.FOCAL_LENGTH,
        "X_A": st.session_state.X_A,
        "X_B": st.session_state.X_B,
        "Y": st.session_state.Y,
        "CAMERA_DISTANCE": st.session_state.CAMERA_DISTANCE,
        "DOFFS": st.session_state.DOFFS,
    }

    return constants

st.markdown(
    """
    <style>
    div.stButton > button {
        display: block;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #4CAF50; /* Green background */
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px 24px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        color: #4CAF50; /* Green text */
        background-color: white; /* White background */
        border: 2px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom styled title with modern design and icons using HTML and CSS
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: 900;
        margin-bottom: 50px;
        text-align: center;
        color: #fff; /* White text for contrast */
        background: linear-gradient(45deg, #ff9a9e, #fad0c4, #fad0c4); /* Soft pink gradient */
        border-radius: 15px;
        padding: 20px 30px;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        letter-spacing: 2px;
    }
    .icon {
        font-size: 50px;
        margin-right: 15px;
    }
    .title:hover {
        background: linear-gradient(45deg, #76b852, #8DC26F); /* Gradient hover effect */
        cursor: pointer;
    }
    </style>
    <div class="title">
        <span class="icon">🎥</span> 
        Stereo Vision to 3D Point Cloud 
        <span class="icon">🖥️</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def main():
    
    # Fetch user inputs for constants
    constants = get_user_inputs()
    st.sidebar.markdown("### Current Constants:")
    st.sidebar.json(constants)

    st.session_state.SCALE = constants["SCALE"]
    st.session_state.FOCAL_LENGTH = constants["FOCAL_LENGTH"]
    st.session_state.X_A = constants["X_A"]
    st.session_state.X_B = constants["X_B"]
    st.session_state.Y = constants["Y"]
    st.session_state.CAMERA_DISTANCE = constants["CAMERA_DISTANCE"]
    st.session_state.DOFFS = constants["DOFFS"]

    # Upload images
    uploaded_file1 = st.file_uploader("Upload Left Image", type=["png", "jpg", "jpeg"])
    uploaded_file2 = st.file_uploader("Upload Right Image", type=["png", "jpg", "jpeg"])
    if st.button("Generate 3D Model"):
        if uploaded_file1 is not None and uploaded_file2 is not None:
            img1 = Image.open(uploaded_file1)
            img2 = Image.open(uploaded_file2)

            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file1.name)
        
            with open(path, "wb") as f:
                f.write(uploaded_file1.getvalue())
        
            st.success(f"File saved temporarily at: {path}")
            
            st.image(img1, caption="Left Image", use_container_width=True)
            st.image(img2, caption="Right Image", use_container_width=True)

            imgL = Image_processing(uploaded_file1)
            imgR = Image_processing(uploaded_file2)
            
            # Compute disparity map
            dispMap = disparitymap(imgL, imgR)

            plt.title('Disparity Map')
            plt.ylabel('Height {}'.format(dispMap.shape[0]))
            plt.xlabel('Width {}'.format(dispMap.shape[1]))
            plt.imshow(dispMap, cmap='gray')
            plt.show()
            
            # Display disparity map with colormap
            fig_disp = plt.figure()

            img = cv2.imread(path, 1)
            img = downsample_image(img, 1) 
            
            st.success(f"Generating Depth Map")
            # Compute depth map
            coordinates = depth_map(dispMap, img)

            # Provide the option to download the point cloud
            ply_file = 'output.ply'
            create_output(coordinates, ply_file)
            st.success(f"3D done")
            with open(ply_file, "rb") as f:
                st.download_button("Download .ply", f, file_name="output.ply")
        
if __name__ == "__main__":
    main()

