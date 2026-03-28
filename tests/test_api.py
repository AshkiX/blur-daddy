"""Tests for the high-level Python API."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from blur_daddy.models import BlurResult, Detection, DetectionResult, Face


class TestModels:
    def test_detection_construction(self):
        d = Detection(id="det-0", target_type="face", box=(10, 20, 100, 200), confidence=0.95)
        assert d.id == "det-0"
        assert d.box_int == (10, 20, 100, 200)

    def test_face_inherits_detection(self):
        f = Face(id="face-0", target_type="face", box=(10, 20, 100, 200), confidence=0.9)
        assert isinstance(f, Detection)
        assert f.target_type == "face"

    def test_face_sets_target_type(self):
        f = Face(id="face-0", target_type="other", box=(0, 0, 1, 1), confidence=0.5)
        assert f.target_type == "face"

    def test_detection_result_faces_filters(self):
        faces = [Face(id="face-0", target_type="face", box=(0, 0, 1, 1), confidence=0.9)]
        others = [Detection(id="text-0", target_type="text", box=(0, 0, 1, 1), confidence=0.8)]
        result = DetectionResult(image=np.zeros((10, 10, 3), dtype=np.uint8), detections=faces + others)
        assert len(result.faces) == 1
        assert result.faces[0].id == "face-0"

    def test_detection_result_save(self, tmp_path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Face(id="face-0", target_type="face", box=(10, 10, 50, 50), confidence=0.9)
        result = DetectionResult(image=img, detections=[det])
        path = str(tmp_path / "preview.jpg")
        result.save(path)
        assert os.path.exists(path)

    def test_blur_result_save(self, tmp_path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = BlurResult(image=img, detections=[])
        path = str(tmp_path / "out.jpg")
        result.save(path)
        assert os.path.exists(path)


class TestBlurDaddyInit:
    def test_default_params(self):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        assert bd.model == "yolov8n-face"
        assert bd.method == "gaussian"

    def test_custom_params(self):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy(model="mtcnn", method="elliptical")
        assert bd.model == "mtcnn"
        assert bd.method == "elliptical"

    def test_invalid_model_raises(self):
        from blur_daddy import BlurDaddy

        with pytest.raises(ValueError, match="model"):
            BlurDaddy(model="invalid")

    def test_invalid_method_raises(self):
        from blur_daddy import BlurDaddy

        with pytest.raises(ValueError, match="method"):
            BlurDaddy(method="invalid")


class TestBlurDaddyDetect:
    def test_returns_detection_result(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        result = bd.detect(sample_image_path)
        assert isinstance(result, DetectionResult)

    def test_detects_faces(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        result = bd.detect(sample_image_path)
        assert len(result.faces) > 0

    def test_faces_have_valid_boxes(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        result = bd.detect(sample_image_path)
        for face in result.faces:
            x1, y1, x2, y2 = face.box
            assert x1 < x2
            assert y1 < y2
            assert face.confidence > 0

    def test_accepts_ndarray(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        img = cv2.imread(sample_image_path)
        result = bd.detect(img)
        assert len(result.faces) > 0

    def test_accepts_path_object(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        result = bd.detect(Path(sample_image_path))
        assert len(result.faces) > 0

    def test_mtcnn_returns_landmarks(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy(model="mtcnn")
        result = bd.detect(sample_image_path)
        assert len(result.faces) > 0
        assert result.faces[0].landmarks is not None

    def test_unsupported_target_raises(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        with pytest.raises(NotImplementedError, match="plates"):
            bd.detect(sample_image_path, targets=["plates"])

    def test_blank_image_returns_empty(self, blank_image):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        result = bd.detect(blank_image)
        assert len(result.detections) == 0


class TestBlurDaddyBlur:
    def test_returns_blur_result(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        result = bd.blur(sample_image_path)
        assert isinstance(result, BlurResult)

    def test_image_same_shape(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        original = cv2.imread(sample_image_path)
        result = bd.blur(sample_image_path)
        assert result.image.shape == original.shape

    def test_image_differs_from_original(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        original = cv2.cvtColor(cv2.imread(sample_image_path), cv2.COLOR_BGR2RGB)
        result = bd.blur(sample_image_path)
        assert not np.array_equal(result.image, original)

    def test_gaussian_method(self, sample_image_path):
        from blur_daddy import BlurDaddy

        result = BlurDaddy(method="gaussian").blur(sample_image_path)
        assert result.image is not None

    def test_pixelation_method(self, sample_image_path):
        from blur_daddy import BlurDaddy

        result = BlurDaddy(method="pixelation").blur(sample_image_path)
        assert result.image is not None

    def test_elliptical_method(self, sample_image_path):
        from blur_daddy import BlurDaddy

        result = BlurDaddy(method="elliptical").blur(sample_image_path)
        assert result.image is not None

    def test_keep_protects_detection(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        preview = bd.detect(sample_image_path)
        if len(preview.faces) < 2:
            pytest.skip("Need at least 2 faces for keep test")

        # Blur all, then blur keeping face 0
        result_all = bd.blur(sample_image_path)
        result_kept = bd.blur(sample_image_path, keep=[preview.faces[0]])

        # The kept result should differ from full blur (face 0 region not blurred)
        assert not np.array_equal(result_all.image, result_kept.image)

    def test_keep_all_returns_original(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        preview = bd.detect(sample_image_path)
        if not preview.faces:
            pytest.skip("No faces detected")

        original = cv2.cvtColor(cv2.imread(sample_image_path), cv2.COLOR_BGR2RGB)
        result = bd.blur(sample_image_path, keep=preview.faces)
        # If we keep all faces, nothing should be blurred
        assert np.array_equal(result.image, original)

    def test_track_raises_not_implemented(self, sample_image_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        with pytest.raises(NotImplementedError, match="tracking"):
            bd.blur(sample_image_path, track=True)

    def test_blur_result_save(self, sample_image_path, tmp_path):
        from blur_daddy import BlurDaddy

        bd = BlurDaddy()
        result = bd.blur(sample_image_path)
        path = str(tmp_path / "out.jpg")
        result.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestConvenienceFunction:
    def test_blur_to_file(self, sample_image_path, tmp_path):
        import blur_daddy

        out = str(tmp_path / "out.jpg")
        result = blur_daddy.blur(sample_image_path, out)
        assert isinstance(result, BlurResult)
        assert os.path.exists(out)

    def test_blur_without_output(self, sample_image_path):
        import blur_daddy

        result = blur_daddy.blur(sample_image_path)
        assert isinstance(result, BlurResult)
        assert result.image is not None

    def test_blur_with_method(self, sample_image_path):
        import blur_daddy

        result = blur_daddy.blur(sample_image_path, method="elliptical")
        assert result.image is not None

    def test_blur_ndarray(self, sample_image_path):
        import blur_daddy

        img = cv2.imread(sample_image_path)
        result = blur_daddy.blur(img)
        assert result.image is not None


class TestLazyLoading:
    def test_mtcnn_uses_lazy_singleton(self):
        """Verify MTCNN uses the lazy _get_mtcnn_model() pattern, not eager init."""
        from blur_daddy import detection

        # The lazy pattern: _mtcnn_model starts as None and is set by _get_mtcnn_model()
        assert hasattr(detection, "_mtcnn_model")
        assert hasattr(detection, "_get_mtcnn_model")
        # Calling it returns the same instance (singleton)
        m1 = detection._get_mtcnn_model()
        m2 = detection._get_mtcnn_model()
        assert m1 is m2
