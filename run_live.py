import cv2
import numpy as np

# ***** CFG ******
IMAGE_SIZE = (1280, 720)
MOVING_OBJ_THRESHOLD = 1  # percent of image
SCORE_THRESHOLD = (MOVING_OBJ_THRESHOLD / 100) * (IMAGE_SIZE[0] * IMAGE_SIZE[1])
# **** END CFG ****

images = []
for i in range(0, 52):
    image = cv2.imread(f"saved_images/image_0{i}.jpg" if i < 10 else f"saved_images/image_{i}.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
    images.append(image)

first_line = []
obj_index_array = []
bbox_coords = ((0, 0), (0, 0))
line_counter = 0


def get_obj_index_array(line):
    global bbox_coords
    global line_counter

    found_obj_index_array = []
    reading_obj = False
    index_start = 0
    index_end = 0
    score = 0
    for k in range(0, len(line)):
        if reading_obj:
            if line[k]:
                score += line[k]
                continue
            else:
                index_end = k - 1
                reading_obj = False
                if score >= SCORE_THRESHOLD:
                    bbox_coords = ((index_start, max(0, line_counter - int(score / (index_end - index_start) * 2))), (index_end, line_counter))
                    break
                found_obj_index_array.append((index_start, index_end, score))
        elif line[k]:
            score = line[k]
            index_start = k
            reading_obj = True
    return found_obj_index_array


def init_first_line(width):
    global first_line
    global line_counter
    global bbox_coords
    global obj_index_array

    bbox_coords = ((0, 0), (0, 0))
    obj_index_array = []
    line_counter = 0
    first_line = np.zeros(width, dtype=np.uint8)


def add_line(line):
    global first_line
    global line_counter
    global obj_index_array
    global bbox_coords

    if bbox_coords != ((0, 0), (0, 0)):
        return

    first_line = np.add(first_line, line)
    line_counter += 1

    for obj in obj_index_array:
        if (first_line[max(0, obj[0] - 1)] == 1 or first_line[min(obj[1] + 1, len(first_line) - 1)] == 1) or np.sum(first_line[obj[0]:(obj[1] + 1)]) != obj[2]:
            continue
        else:
            first_line[obj[0]:(obj[1] + 1)] = np.zeros((obj[1] - obj[0] + 1), dtype=np.uint8)

    obj_index_array = get_obj_index_array(first_line)

    # print(f"{[obj for obj in obj_index_array]} - {list(first_line)}")


while True:
    for k, image in enumerate(images):
        previous_image = images[k-1].copy() if k != 0 else None
        if previous_image is not None:
            diff = cv2.absdiff(previous_image, image)
            _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            init_first_line(width=previous_image.shape[1])
            for line in thresh:
                add_line(line/255)
            # cv2.imshow('thresh', thresh)
            display_image = image.copy()
            if bbox_coords != ((0, 0), (0, 0)):
                cv2.putText(display_image, "MOVING!!!", (int(IMAGE_SIZE[0]/3), int(IMAGE_SIZE[1]/6)), cv2.FONT_HERSHEY_COMPLEX, 2, 255)
            cv2.rectangle(display_image, bbox_coords[0], bbox_coords[1], (255, 0, 0), 2)
            cv2.imshow('motion', display_image)
        cv2.waitKey(0)
