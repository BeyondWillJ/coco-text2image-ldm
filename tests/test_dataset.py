"""
Tests for COCO dataset loading.

These tests use a synthetic in-memory COCO-format dataset to avoid requiring
the full COCO download.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers to build a tiny fake COCO dataset
# ---------------------------------------------------------------------------

def make_fake_coco(tmp_path, n_images=4, split="train"):
    """Create a minimal directory tree that mimics MS-COCO structure."""
    ann_dir = tmp_path / "annotations"
    img_dir = tmp_path / "images" / f"{split}2017"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    images = []
    annotations = []
    ann_id = 1
    for i in range(1, n_images + 1):
        fname = f"{i:012d}.jpg"
        # Save a random 64×64 RGB image
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / fname)
        images.append({"id": i, "file_name": fname, "width": 64, "height": 64})
        for _ in range(2):
            annotations.append({
                "id": ann_id,
                "image_id": i,
                "caption": f"a sample caption for image {i}",
            })
            ann_id += 1

    coco_ann = {"images": images, "annotations": annotations, "type": "captions"}
    ann_file = ann_dir / f"captions_{split}2017.json"
    with open(ann_file, "w") as f:
        json.dump(coco_ann, f)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dataset_len():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_coco(Path(tmp), n_images=4, split="train")
        from ldm.data.coco import COCOTrain

        ds = COCOTrain(data_root=str(root), size=32)
        assert len(ds) == 4


def test_dataset_item_keys():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_coco(Path(tmp), n_images=2, split="train")
        from ldm.data.coco import COCOTrain

        ds = COCOTrain(data_root=str(root), size=32)
        item = ds[0]
        assert "image" in item
        assert "caption" in item
        assert "img_id" in item


def test_dataset_image_shape_and_range():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_coco(Path(tmp), n_images=2, split="train")
        from ldm.data.coco import COCOTrain

        ds = COCOTrain(data_root=str(root), size=32)
        item = ds[0]
        img = item["image"]
        assert img.shape == torch.Size([32, 32, 3]), f"Bad shape: {img.shape}"
        assert img.min() >= -1.0 - 1e-5, "Image values below -1"
        assert img.max() <= 1.0 + 1e-5, "Image values above +1"


def test_dataset_caption_is_string():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_coco(Path(tmp), n_images=2, split="train")
        from ldm.data.coco import COCOTrain

        ds = COCOTrain(data_root=str(root), size=32)
        assert isinstance(ds[0]["caption"], str)


def test_val_split_no_flip():
    """Validation split should have flip_p=0 (no augmentation)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_coco(Path(tmp), n_images=2, split="val")
        from ldm.data.coco import COCOValidation

        ds = COCOValidation(data_root=str(root), size=32)
        assert ds.flip_p == 0.0
