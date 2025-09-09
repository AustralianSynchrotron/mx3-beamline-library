from os import environ, path

import yaml
from redis import StrictRedis
from redis.exceptions import ConnectionError

from .logger import setup_logger

logger = setup_logger()

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from opentelemetry.instrumentation.confluent_kafka import (
    ConfluentKafkaInstrumentor,
)

# Determine which mode the beamline library is running on, by default it is run
# in SIM mode
BL_ACTIVE = environ.get("BL_ACTIVE", "false").lower()

# Redis connection
REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
REDIS_USERNAME = environ.get("REDIS_USERNAME", None)
REDIS_PASSWORD = environ.get("REDIS_PASSWORD", None)
REDIS_DB = int(environ.get("REDIS_DB", "0"))

# Detector
SIMPLON_API = environ.get("SIMPLON_API", "http://0.0.0.0:8000")

# MD3
MD3_HOST = environ.get("MD3_REDIS_HOST", "localhost")
MD3_PORT = environ.get("MD3_REDIS_PORT", "8379")
MD3_DB = environ.get("MD3_REDIS_DB", "0")


try:
    redis_connection = StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
    )
except ConnectionError:
    logger.warning(
        "A redis connection is not available. Some functionalities may be limited."
    )

with open(
    path.join(path.dirname(__file__), "devices", "classes", "md3_config.yml")
) as config:
    MD3_CONFIG = yaml.safe_load(config)

OTEL_SDK_DISABLED = environ.get("OTEL_SDK_DISABLED", "true").lower() == "true" 
if not OTEL_SDK_DISABLED:
    # Opentelemetry-related
    # Automatically creates a Resource using environment variables
    resource = Resource.create()
    traceProvider = TracerProvider(resource=resource)
    # Automatically creates a OTLPSpanExporter using environment variables
    processor = BatchSpanProcessor(OTLPSpanExporter())
    traceProvider.add_span_processor(processor)
    trace.set_tracer_provider(traceProvider)

    # Instrument confluent_kafka producer
    ConfluentKafkaInstrumentor().instrument()