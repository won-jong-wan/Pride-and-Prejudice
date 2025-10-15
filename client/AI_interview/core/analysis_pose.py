import xml.etree.ElementTree as ET

def parse_posture_summary(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    head_tilt_count = body_tilt_count = gesture_count = frame_total = 0

    for frame in root.findall("frame"):
        frame_total += 1
        analysis = frame.find("analysis")
        if analysis is None:
            continue
        for result in analysis.findall("result"):
            rtype = result.get("type", "")
            if rtype == "head_tilt":
                head_tilt_count += 1
            elif rtype == "body_tilt":
                body_tilt_count += 1
            elif rtype == "gesture":
                gesture_count += 1

    labels = []
    if frame_total > 0:
        if head_tilt_count > frame_total * 0.3:
            labels.append("머리 기울임 잦음")
        if body_tilt_count > frame_total * 0.3:
            labels.append("몸 기울어짐 잦음")

    if gesture_count == 0:
        gesture_label = "제스처 없음 (답변이 딱딱할 수 있음)"
    elif 1 <= gesture_count <= 3:
        gesture_label = "자연스러운 제스처 사용"
    else:
        gesture_label = "제스처 과다 (산만할 수 있음)"
    labels.append(gesture_label)

    return {
        "frames": frame_total,
        "head_tilt_count": head_tilt_count,
        "body_tilt_count": body_tilt_count,
        "gesture_count": gesture_count,
        "label": ", ".join(labels)
    }
