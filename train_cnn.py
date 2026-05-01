from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE


class EpochMetricsLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        train_loss = float(logs.get("loss", 0.0))
        train_acc = float(logs.get("accuracy", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))
        val_acc = float(logs.get("val_accuracy", 0.0))
        print(
            f"[epoch {epoch + 1:03d}] "
            f"lr={lr:.2e} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight CNN for sign language recognition.")
    parser.add_argument("--data_dir", default="data", help="Folder with one subfolder per class.")
    parser.add_argument("--output_dir", default="artifacts", help="Directory where outputs are saved.")
    parser.add_argument("--image_size", type=int, default=224, help="Input size for the CNN.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--backbone", choices=["efficientnetb0", "resnet50", "mobilenetv2"], default="efficientnetb0", help="Pretrained backbone.")
    parser.add_argument("--head_epochs", type=int, default=18, help="Epochs for the classifier head.")
    parser.add_argument("--fine_tune_epochs", type=int, default=12, help="Epochs for fine-tuning.")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Initial epoch to resume head training from.")
    parser.add_argument("--fine_tune_initial_epoch", type=int, default=0, help="Initial epoch to resume fine-tune training from.")
    parser.add_argument("--resume", action="store_true", help="If set, load weights from existing checkpoint to resume training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_horizontal_flip", action="store_true", help="Apply horizontal flip augmentation when applicable.")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def configure_runtime() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("Runtime device: CPU (no GPU detected)")
        return

    print(f"Runtime device: GPU ({len(gpus)} detected)")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def list_samples(data_dir: str) -> tuple[list[str], list[int], list[str]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_names = sorted([item.name for item in root.iterdir() if item.is_dir()])
    if len(class_names) < 2:
        raise ValueError("Expected at least two class folders.")

    paths: list[str] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        for file_path in sorted((root / class_name).iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(file_path))
                labels.append(class_index)

    if not paths:
        raise ValueError(f"No image files found under {data_dir}.")

    return paths, labels, class_names


def build_splits(paths: list[str], labels: list[int], seed: int) -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]], tuple[list[str], list[int]]]:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def decode_image(path: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.int32)


def build_dataset(paths: list[str], labels: list[int], image_size: int, batch_size: int, training: bool, use_flip: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(min(len(paths), 4096), reshuffle_each_iteration=True)

    dataset = dataset.map(lambda path, label: decode_image(path, label, image_size), num_parallel_calls=AUTOTUNE)

    if training:
        def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            image = tf.image.random_brightness(image, max_delta=24.0)
            image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
            image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
            image = tf.image.random_hue(image, max_delta=0.02)
            image = tf.clip_by_value(image, 0.0, 255.0)
            if use_flip:
                image = tf.image.random_flip_left_right(image)
            return image, label

        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def build_model(num_classes: int, image_size: int, backbone: str) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="augment",
    )

    x = augment(inputs)

    if backbone == "efficientnetb0":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
    elif backbone == "resnet50":
        preprocess = tf.keras.applications.resnet50.preprocess_input
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
    else:
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )

    x = tf.keras.layers.Lambda(preprocess, name="preprocess")(x)
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(384, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs, name=f"sign_language_{backbone}")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def freeze_batch_norm_layers(model: tf.keras.Model) -> None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def evaluate_and_save_reports(model: tf.keras.Model, test_dataset: tf.data.Dataset, class_names: list[str], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_images, batch_labels in test_dataset:
        logits = model.predict(batch_images, verbose=0)
        predictions = np.argmax(logits, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(predictions.tolist())

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    print(f"\nTest accuracy: {accuracy * 100.0:.2f}%")
    print("\nClassification report:\n")
    print(report_text)

    weak_classes = sorted(
        ((name, report_dict[name]["recall"]) for name in class_names),
        key=lambda item: item[1],
    )[:5]

    print("\nWeak classes by recall:")
    for class_name, recall in weak_classes:
        print(f"- {class_name}: {recall:.3f}")

    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as file_handle:
        json.dump(report_dict, file_handle, indent=2)

    with (output_dir / "weak_classes.json").open("w", encoding="utf-8") as file_handle:
        json.dump([{ "class": name, "recall": recall } for name, recall in weak_classes], file_handle, indent=2)

    if plt is None:
        print("matplotlib not available; skipping confusion matrix image export.")
    else:
        fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.35), max(6, len(class_names) * 0.35)))
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        threshold = matrix.max() / 2.0 if matrix.size else 0.0
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                ax.text(
                    col_index,
                    row_index,
                    format(matrix[row_index, col_index], "d"),
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, col_index] > threshold else "black",
                    fontsize=7,
                )

        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
        plt.close(fig)


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        output_path.write_bytes(converter.convert())
        return
    except Exception as direct_error:
        print(f"Direct Keras->TFLite export failed: {direct_error}")
        print("Falling back to SavedModel->TFLite export...")

    saved_model_dir = output_path.parent / "saved_model_export"
    if saved_model_dir.exists():
        tf.io.gfile.rmtree(str(saved_model_dir))
    tf.saved_model.save(model, str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output_path.write_bytes(converter.convert())


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    configure_runtime()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, class_names = list_samples(args.data_dir)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = build_splits(paths, labels, args.seed)

    print(f"Samples: total={len(paths)} train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")
    print(f"Classes: {len(class_names)}")

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as file_handle:
        json.dump(class_names, file_handle, indent=2)

    train_dataset = build_dataset(train_paths, train_labels, args.image_size, args.batch_size, True, args.use_horizontal_flip)
    val_dataset = build_dataset(val_paths, val_labels, args.image_size, args.batch_size, False, False)
    test_dataset = build_dataset(test_paths, test_labels, args.image_size, args.batch_size, False, False)

    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.array(train_labels),
    )
    class_weights = {index: float(weight) for index, weight in enumerate(class_weight_values)}

    model, base_model = build_model(len(class_names), args.image_size, args.backbone)

    checkpoint_path = output_dir / "best_model.keras"
    csv_log_path = output_dir / f"training_history_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=False),
        EpochMetricsLogger(),
    ]

    compile_model(model, learning_rate=1e-3)
    # If user requested resume and a checkpoint exists, load weights
    if args.resume and checkpoint_path.exists():
        try:
            model.load_weights(str(checkpoint_path))
            print(f"Loaded weights from checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint weights: {e}")
    print("\nTraining classifier head...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.head_epochs,
        initial_epoch=args.initial_epoch,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    base_model.trainable = True
    if len(base_model.layers) > 50:
        for layer in base_model.layers[:-50]:
            layer.trainable = False

    freeze_batch_norm_layers(base_model)

    compile_model(model, learning_rate=1e-5)
    print("\nFine-tuning backbone...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.fine_tune_epochs,
        initial_epoch=args.fine_tune_initial_epoch,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(output_dir / "sign_language_cnn.keras")
    export_tflite(model, output_dir / "sign_language_cnn.tflite")
    evaluate_and_save_reports(model, test_dataset, class_names, output_dir)

    print(f"\nSaved model: {output_dir / 'sign_language_cnn.keras'}")
    print(f"Saved TFLite model: {output_dir / 'sign_language_cnn.tflite'}")
    print(f"Saved labels: {output_dir / 'label_map.json'}")
    print(f"Saved training log: {csv_log_path}")


if __name__ == "__main__":
    main()
