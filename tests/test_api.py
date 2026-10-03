from PIL import Image

from .conftest import encode


def post(client, data, content_type="image/png", **form):
    return client.post("/api/predict", files={"file": ("stone.png", data, content_type)}, data=form)


def test_health_lists_models(client):
    assert client.get("/api/health").json() == {"status": "ok", "models": ["yolo11s", "rtdetr-l"]}


def test_predict_single_model(client, png_bytes):
    response = post(client, png_bytes, model="yolo11s", conf="0.25")

    assert response.status_code == 200
    [prediction] = response.json()
    assert prediction["model"] == "yolo11s"
    assert prediction["image"] == {"width": 64, "height": 48}
    assert prediction["detections"] == [{"class": "Inclusion", "confidence": 0.9, "box": [10.0, 20.0, 30.0, 40.0]}]
    assert "Server-Timing" in response.headers


def test_predict_both_returns_one_result_per_model(client, png_bytes):
    response = post(client, png_bytes, model="both")
    assert [p["model"] for p in response.json()] == ["yolo11s", "rtdetr-l"]


def test_conf_filters_detections(client, png_bytes):
    [prediction] = post(client, png_bytes, model="yolo11s", conf="0.95").json()
    assert prediction["detections"] == []


def test_rejects_unsupported_content_type(client):
    assert post(client, b"hello", content_type="text/plain").status_code == 415


def test_rejects_bytes_that_are_not_an_image(client):
    assert post(client, b"not really a png").status_code == 415


def test_rejects_oversized_upload(client):
    assert post(client, b"0" * (1024 * 1024 + 1)).status_code == 413


def test_rejects_unknown_model_and_bad_conf(client, png_bytes):
    assert post(client, png_bytes, model="resnet").status_code == 422
    assert post(client, png_bytes, conf="1.5").status_code == 422


def test_exif_rotation_is_applied(client, detectors):
    image = Image.new("RGB", (80, 40), "white")
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90° clockwise when displayed
    response = post(client, encode(image, "JPEG", exif=exif), content_type="image/jpeg", model="yolo11s")

    assert response.json()[0]["image"] == {"width": 40, "height": 80}
    assert detectors["yolo11s"].seen_shapes[-1][:2] == (80, 40)


def test_large_jpeg_is_decoded_reduced_and_boxes_scaled_back(client, detectors):
    data = encode(Image.new("RGB", (4000, 3000), "white"), "JPEG")
    [prediction] = post(client, data, content_type="image/jpeg", model="yolo11s").json()

    decoded_width = detectors["yolo11s"].seen_shapes[-1][1]
    assert decoded_width < 4000
    assert prediction["image"] == {"width": 4000, "height": 3000}
    scale = 4000 / decoded_width
    assert prediction["detections"][0]["box"] == [round(v * scale, 1) for v in (10, 20, 30, 40)]


def test_missing_model_returns_503(detectors, tmp_path, png_bytes):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.settings import Settings

    settings = Settings(static_dir=tmp_path, metrics_csv=tmp_path / "none.csv")
    with TestClient(create_app(settings, {"yolo11s": detectors["yolo11s"]})) as client:
        assert post(client, png_bytes, model="rtdetr-l").status_code == 503
