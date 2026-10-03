from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    confidence: float
    box: list[float] = Field(description="[x1, y1, x2, y2] in original image pixels")


class ImageSize(BaseModel):
    width: int
    height: int


class Prediction(BaseModel):
    model: str
    inference_ms: float
    image: ImageSize
    detections: list[Detection]


class Health(BaseModel):
    status: str
    models: list[str]


class ModelSummary(BaseModel):
    name: str
    label: str
    architecture: str
    metrics: dict[str, float | str] | None = None
