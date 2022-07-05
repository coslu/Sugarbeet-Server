from PIL import Image, ImageOps


def run(image_file, label_file):

    index = 0
    outputs = []

    for line in open(label_file):
        if int(line.split(" ")[0]) != 0:
            print("Class is no sugar_beet!")
        else:
            coordinate_x = float(line.split(" ")[1])
            coordinate_y = float(line.split(" ")[2])
            relative_width = float(line.split(" ")[3])
            relative_height = float(line.split(" ")[4])

            im = Image.open(image_file)
            im = ImageOps.exif_transpose(im)

            real_width, real_height = im.size

            abs_x_center = coordinate_x * real_width
            abs_y_center = coordinate_y * real_height

            delta_x = (relative_width * real_width) / 2
            delta_y = (relative_height * real_height) / 2

            left = abs_x_center - delta_x
            top = abs_y_center - delta_y
            right = abs_x_center + delta_x
            bottom = abs_y_center + delta_y

            cropped_im = im.crop((left, top, right, bottom))

            output_file = image_file.removeprefix("input/")
            output_file_name = "preprocessing_output/" + output_file.split(".")[0] + str(index) + ".jpg"
            cropped_im.save(output_file_name)
            outputs.append(output_file_name)

            index = index + 1

    return outputs


def main(image_file, label_file):
    run(image_file, label_file)


if __name__ == "__main__":
    main("../test_images.jpg", "results/labels/test_images.txt")