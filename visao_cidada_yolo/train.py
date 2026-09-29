"""Treina um modelo YOLO usando um dataset no formato Ultralytics.

Este script suporta execução via linha de comando ou integração com interface gráfica.
"""

import argparse
from typing import Optional

from config import (
    AMP,
    AUGMENTATION_PRESETS,
    BATCH_SIZE,
    CLOSE_MOSAIC,
    COPY_PASTE,
    COS_LR,
    DATASET_YAML,
    DEGREES,
    DEVICE,
    EPOCHS,
    FLIPLR,
    FLIPUD,
    HSV_H,
    HSV_S,
    HSV_V,
    IMAGE_SIZE,
    LABEL_SMOOTHING,
    LR0,
    LRF,
    MIXUP,
    MODEL_NAME,
    MODELS_DIR,
    MOSAIC,
    MULTI_SCALE,
    OPTIMIZER,
    PATIENCE,
    RECT,
    SCALE,
    TRANSLATE,
    WARMUP_EPOCHS,
    WARMUP_MOMENTUM,
    WEIGHT_DECAY,
    WORKERS,
    ensure_project_directories,
    require_dataset_yaml,
)
from ultralytics import YOLO
from utils.logger import get_logger

logger = get_logger("train")


def on_train_start(trainer):
    logger.info("Iniciando o treinamento do modelo.")


def on_train_end(trainer):
    logger.info("Treinamento finalizado.")


def train_from_ui(
    epochs: int = EPOCHS,
    model_name: str = MODEL_NAME,
    imgsz: int = IMAGE_SIZE,
    batch: int = BATCH_SIZE,
    device: str = DEVICE,
    workers: int = WORKERS,
    optimizer: str = OPTIMIZER,
    lr0: float = LR0,
    patience: int = PATIENCE,
    augment_preset: str = 'medium',
    resume: Optional[str] = None,
    freeze: int = 0,
    name: str = 'treino',
    cos_lr: bool = COS_LR,
    multi_scale: bool = MULTI_SCALE,
    amp: bool = AMP,
    label_smoothing: float = LABEL_SMOOTHING,
    rect: bool = RECT,
) -> None:
    ensure_project_directories()
    require_dataset_yaml()

    logger.info("Configuracoes de treinamento:")
    logger.info("Epocas: %d, Batch: %d, Image Size: %d", epochs, batch, imgsz)
    logger.info("Modelo Base: %s, Optimizer: %s, LR0: %f", model_name, optimizer, lr0)
    logger.info("Preset Augmentation: %s, Device: %s, Resume: %s", augment_preset, device, resume)
    logger.info("Patience: %d, Freeze: %d, AMP: %s, Label Smoothing: %f", patience, freeze, amp, label_smoothing)

    model_path = resume if resume else model_name
    model = YOLO(model_path)
    
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_end", on_train_end)

    preset = AUGMENTATION_PRESETS.get(augment_preset, AUGMENTATION_PRESETS.get('medium', {}))
    
    # Configurando os argumentos de treino
    train_args = dict(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(MODELS_DIR),
        name=name,
        exist_ok=True,
        optimizer=optimizer,
        lr0=lr0,
        lrf=LRF,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        patience=patience,
        close_mosaic=CLOSE_MOSAIC,
        amp=amp,
        label_smoothing=label_smoothing,
        multi_scale=multi_scale,
        cos_lr=cos_lr,
        rect=rect,
        resume=resume is not None,
        mosaic=preset.get("mosaic", MOSAIC),
        mixup=preset.get("mixup", MIXUP),
        copy_paste=preset.get("copy_paste", COPY_PASTE),
        degrees=preset.get("degrees", DEGREES),
        translate=preset.get("translate", TRANSLATE),
        scale=preset.get("scale", SCALE),
        flipud=preset.get("flipud", FLIPUD),
        fliplr=preset.get("fliplr", FLIPLR),
        hsv_h=preset.get("hsv_h", HSV_H),
        hsv_s=preset.get("hsv_s", HSV_S),
        hsv_v=preset.get("hsv_v", HSV_V),
    )
    
    if freeze > 0:
        train_args['freeze'] = freeze
        
    results = model.train(**train_args)
    logger.info("Treino concluido: %s", results.save_dir)
    
    logger.info("Iniciando validacao...")
    metrics = model.val()
    logger.info("Resultados de Validacao - mAP50: %.4f, mAP50-95: %.4f", metrics.box.map50, metrics.box.map)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um modelo YOLO.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER)
    parser.add_argument("--lr0", type=float, default=LR0)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--augment-preset", type=str, choices=['light','medium','aggressive'], default='medium')
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--name", type=str, default='treino')
    
    parser.add_argument("--cos-lr", action='store_true', default=COS_LR)
    parser.add_argument("--no-cos-lr", action='store_false', dest='cos_lr')
    
    parser.add_argument("--multi-scale", action='store_true', default=MULTI_SCALE)
    parser.add_argument("--no-multi-scale", action='store_false', dest='multi_scale')
    
    parser.add_argument("--amp", action='store_true', default=AMP)
    parser.add_argument("--no-amp", action='store_false', dest='amp')
    
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    
    parser.add_argument("--rect", action='store_true', default=RECT)
    parser.add_argument("--no-rect", action='store_false', dest='rect')

    args = parser.parse_args()

    train_from_ui(
        epochs=args.epochs,
        model_name=args.model,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        augment_preset=args.augment_preset,
        resume=args.resume,
        freeze=args.freeze,
        name=args.name,
        cos_lr=args.cos_lr,
        multi_scale=args.multi_scale,
        amp=args.amp,
        label_smoothing=args.label_smoothing,
        rect=args.rect,
    )


if __name__ == "__main__":
    main()
