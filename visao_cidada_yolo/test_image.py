"""Executa deteccao em uma imagem e salva o resultado anotado."""

import argparse
from pathlib import Path

from config import CONFIDENCE, MODEL_NAME
from inference import annotate_frame, count_people, detect_people, load_model, register_detection
from utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Caminho da imagem de entrada")
    parser.add_argument("--weights", default=MODEL_NAME, help="Pesos do modelo YOLO")
    parser.add_argument("--output", default="runs/predictions", help="Pasta de saida")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE, help="Confianca minima (0 a 1)")
    args = parser.parse_args()

    logger = get_logger("test_image")
    image = Path(args.image)
    if not image.exists():
        parser.error(f"Imagem nao encontrada: {image}")

    import cv2

    frame = cv2.imread(str(image))
    if frame is None:
        parser.error(f"Nao foi possivel ler a imagem: {image}")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence deve estar entre 0 e 1")
    result = detect_people(load_model(args.weights), frame, args.confidence)
    count = count_people(result)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / f"{image.stem}_resultado.jpg"), annotate_frame(result))
    register_detection(str(image), "imagem", count, args.confidence)
    logger.info("Pessoas detectadas: %s", count)
    logger.info("Resultado salvo em %s", output)


if __name__ == "__main__":
    main()
