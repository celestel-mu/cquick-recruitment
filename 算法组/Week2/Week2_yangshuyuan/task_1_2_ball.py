import cv2
import numpy as np

image_paths = ['ball1.jpg', 'ball2.jpg', 'ball3.jpg']

for i, path in enumerate(image_paths):
    print(f"\n===== 正在处理: {path} =====")

    # 1. 读取图片
    img = cv2.imread(path)

    # 检查图片是否读取成功
    if img is None:
        print(f"❌ 读取失败，找不到图片: {path}")
        continue

    # 2. 转换到 HSV 色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 设定红色/橙色目标的 HSV 阈值范围
    lower_color = np.array([0, 120, 70])
    upper_color = np.array([25, 255, 255])

    # 4. 生成二值掩膜（Mask）
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 5. 形态学处理
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 6. 寻找轮廓
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()
    target_found = False

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 500:
            x, y, w, h = cv2.boundingRect(c)

            # 计算中心点坐标 (u, v)
            center_u = x + w // 2
            center_v = y + h // 2

            print(f"✅ 成功锁定目标！中心坐标 (u, v): ({center_u}, {center_v}), 面积: {area}")

            # 画绿框和红点
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(result, (center_u, center_v), 5, (0, 0, 255), -1)
            text = f"u={center_u}, v={center_v}"
            # 打印坐标
            cv2.putText(result, text, (x, y - 10 if y - 10 > 20 else y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            cv2.putText(result, "Name: yangshuyuan | Date: 2026-10-31", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 3)
            # 把坐标文字打印在图片上
            text = f"u={center_u}, v={center_v}"
            text_position = (x, y - 10 if y - 10 > 20 else y + 30)
            cv2.putText(result, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            target_found = True

    if not target_found:
        print(f"⚠️ 未检测到目标！")

    # 7. 保存与拼接
    mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
    h_img, w_img = img.shape[:2]
    scale = 600 / w_img
    img_resized = cv2.resize(img, (600, int(h_img * scale)))
    mask_resized = cv2.resize(mask_bgr, (600, int(h_img * scale)))
    result_resized = cv2.resize(result, (600, int(h_img * scale)))

    combined = np.hstack((img_resized, mask_resized, result_resized))

    cv2.imwrite(f"result_visual_{i + 1}.jpg", result)
    cv2.imwrite(f"compare_image_{i + 1}.jpg", combined)

    cv2.imshow(f'Compare {i + 1}', combined)
    print("⚠️ 请按任意键继续下一张...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\n✅ 全部测试完毕！")

print("姓名: yangshuyuan, 时间:", __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))