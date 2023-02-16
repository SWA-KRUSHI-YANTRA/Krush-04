# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Define the path to the frozen inference graph and the label map
PATH_TO_FROZEN_GRAPH = 'path/to/frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = 'path/to/label_map.pbtxt'

# Load the label map
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABEL_MAP, use_display_name=True)

# Load the frozen inference graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Set up the video stream
cap = cv2.VideoCapture(0)

while True:
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Get handles to input and output tensors
    ops = detection_graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={
                           image_tensor: image_np_expanded})

    # Visualize the results
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(output_dict['detection_boxes']),
        np.squeeze(output_dict['detection_classes']).astype(np.int32),
        np.squeeze(output_dict['detection_scores']),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Display the resulting image
    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
