import os
import cv2
import numpy as np
import onnxruntime as ort

# ===============================
# 1️⃣ Load ONNX Model
# ===============================
def load_onnx_model(model_path):
    """Loads an ONNX model for U²-Net segmentation."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    return ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# ===============================
# 2️⃣ Image Preprocessing
# ===============================
def preprocess_image(image):
    """Prepares an image for U²-Net by resizing, normalizing, and adding batch dimension."""
    image = cv2.resize(image, (320, 320))  # Resize to model input size
    image = image.astype(np.float32) / 255.0  # Normalize pixel values (0 to 1)
    image = np.transpose(image, (2, 0, 1))  # Change format to (C, H, W)
    return np.expand_dims(image, axis=0)  # Add batch dimension

# ===============================
# 3️⃣ Post-processing with Edge Refinement
# ===============================
def refine_mask(mask, original_size):
    """Applies Gaussian blur, morphological operations, and adaptive feathering to smooth edges."""
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)  # Convert from float (0 to 1) to uint8 (0 to 255)

    # Apply Gaussian Blur to soften edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Morphological operations: Erosion (removes halo) then Dilation (restores size)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Normalize for better alpha channel blending
    return cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)

# ===============================
# 4️⃣ Segmentation Process
# ===============================
def segment_image(image_path, model_path, output_dir):
    """Runs U²-Net segmentation on an image, removes background, and saves results."""

    # Load ONNX Model
    session = load_onnx_model(model_path)

    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Error: Failed to load the image.")

    # Extract original size & filename
    original_size = (image.shape[1], image.shape[0])
    file_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract name without extension

    # Convert to RGBA if not already
    if image.shape[2] == 3:
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        image_rgba = image.copy()

    # Preprocess image for ONNX model
    input_tensor = preprocess_image(image)

    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})[0]

    # Refine segmentation mask for smoother edges
    segmentation_mask = refine_mask(output[0, 0, :, :], original_size)

    # Apply mask as alpha channel
    alpha_channel = segmentation_mask

    # Create cut-out object (foreground)
    selected_object = image_rgba.copy()
    selected_object[:, :, 3] = alpha_channel  # Apply refined mask

    # Create transparent background image
    background_rgba = image_rgba.copy()
    background_rgba[alpha_channel > 0] = (0, 0, 0, 0)  # Make subject area transparent

    # Create output directory with the original file name
    output_folder = os.path.join(output_dir, file_name)
    os.makedirs(output_folder, exist_ok=True)

    # Save results
    selected_filename = os.path.join(output_folder, f"{file_name}_selected.png")
    background_filename = os.path.join(output_folder, f"{file_name}_background.png")

    cv2.imwrite(selected_filename, selected_object)
    cv2.imwrite(background_filename, background_rgba)

    print(f"✅ Segmentation completed! Files saved in '{output_folder}':\n- {selected_filename}\n- {background_filename}")

# ===============================
# 5️⃣ Run Segmentation
# ===============================
if __name__ == "__main__":
    # File paths
    image_path = "images/test/2_people_together.jpg"  # Change to your image path
    model_path = "models/u2net.onnx"  # Update with your ONNX model path
    output_directory = "images/segmentation_result"  # Base output directory

    # Run the segmentation pipeline
    segment_image(image_path, model_path, output_directory)
