import cv2


def resize_image(image_path, output_path, target_width=None, target_height=None, scale_factor=None):
    """
    将图像尺寸调整到指定值，并保存为新的文件。

    :param image_path: 输入图像文件路径
    :param output_path: 输出调整尺寸后图像文件路径
    :param target_width: 目标宽度，默认为None，表示不指定宽度
    :param target_height: 目标高度，默认为None，表示不指定高度
    :param scale_factor: 缩放比例，默认为None，表示不使用缩放比例
    """
    # 读取原始图像
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image.")
        return

    # 获取原始图像尺寸
    original_height, original_width = img.shape[:2]

    # 如果提供了缩放比例，则根据该比例计算新的尺寸
    if scale_factor is not None:
        target_width = int(original_width * scale_factor)
        target_height = int(original_height * scale_factor)

    # 确保至少指定了宽度或高度
    if target_width is None and target_height is None:
        print("Error: Either target width or height must be specified.")
        return

    # 如果只指定了宽度，则根据比例计算高度
    if target_width is not None and target_height is None:
        ratio = target_width / float(original_width)
        target_height = int(original_height * ratio)

    # 如果只指定了高度，则根据比例计算宽度
    if target_height is not None and target_width is None:
        ratio = target_height / float(original_height)
        target_width = int(original_width * ratio)

    # 调整图像尺寸
    resized_img = cv2.resize(img, (target_width, target_height))

    # 保存调整尺寸后的图像
    cv2.imwrite(output_path, resized_img)

# 示例调用函数
# 方法1：指定目标宽度和高度
resize_image("watermark.png", "watermark1.png", target_width=300, target_height=300)

# 方法2：使用缩放比例
# resize_image("input_image.jpg", "output_image.jpg", scale_factor=0.5)  # 缩小一半