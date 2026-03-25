from pydantic import BaseModel


class RequestData(BaseModel):
    service_type: str
    client_lat: float
    client_lon: float