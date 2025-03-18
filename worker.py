# celery_app.py
import json
from datetime import datetime, timezone

import pymongo
from kombu import Exchange, Queue

from celery import Celery, signals

from . import OCR_agent
from .config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from .config import DATABASE_NAME as mongodb_logs_database
from .config import MONGO_URI as mongodb_connection_string
from .config import SENTRY_DSN, SENTRY_ENV, mongodb_logs_collection_name

app = Celery("celery", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

app.autodiscover_tasks([OCR_agent.__name__], related_name="tasks")

app.conf.update(
    result_expires=3600,
    task_track_started=True,
    timezone="Europe/Rome",
    broker_connection_retry_on_startup=True,
)

app.conf.task_queues = (
    Queue(
        "OCR_agent_queue",
        Exchange("OCR_agent_exchange"),
        routing_key="OCR_agent.#",
    ),
)
app.conf.task_default_queue = "aigot_queue"
app.conf.task_default_exchange = "aigot_exchange"
app.conf.task_default_routing_key = "aigot.default"
app.conf.task_routes = {
    "src.celery.OCR_agent.tasks.*": {"queue": "OCR_agent_queue"},
}


# MongoDB configuration for task logging
client = pymongo.MongoClient(mongodb_connection_string)
db = client[mongodb_logs_database]
task_log_collection = db[
    mongodb_logs_collection_name
]  # Collection for storing task logs


# Sentry integration
@signals.celeryd_init.connect
def init_sentry(**_kwargs):
    if SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.celery import CeleryIntegration

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            enable_tracing=True,
            integrations=[CeleryIntegration(propagate_traces=True)],
            environment=SENTRY_ENV,
        )


# Helper function to map routing key to queue name
def get_queue_name_from_routing_key(routing_key):
    """Returns the queue name based on the routing key"""
    if "OCR_agent" in routing_key:
        return "OCR_agent_queue"
    return "default"


def convert_keys_to_str(item):
    if isinstance(item, dict):
        return {str(k): convert_keys_to_str(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_keys_to_str(elem) for elem in item]
    else:
        return item


def convert_to_json(item):
    if isinstance(item, dict) or isinstance(item, list):
        return json.dumps(item)
    else:
        return item


# Define task logging function
def log_task(task_name, task_id, queue_name, inputs, output, timestamp):

    task_data = {
        "task_name": task_name,
        "task_id": task_id,
        "queue_name": queue_name,
        "inputs": convert_to_json(inputs),
        "output": convert_to_json(output),
        "timestamp": timestamp,
    }
    task_log_collection.insert_one(task_data)


@signals.task_received.connect
def task_received_handler(task_id=None, args=None, kwargs=None, task=None, **_):
    """Log when a task is received."""
    # Ensure the task is not None, then log task information
    if task:
        timestamp = datetime.now(
            timezone.utc
        )  # Get current UTC time as a timezone-aware datetime

        # Access the routing key from task's delivery info
        routing_key = task.request.delivery_info.get("routing_key", "default")

        # Get the queue name based on the routing key
        queue_name = get_queue_name_from_routing_key(routing_key)

        # Get task_id from the task object if not provided
        task_id = task.id if not task_id else task_id

        log_task(
            task_name=task.name,  # Get task name from the task object (not from sender)
            task_id=task.id,  # Ensure task_id is taken from task object
            queue_name=queue_name,
            inputs={"args": args, "kwargs": kwargs},
            output=None,  # Output is not available yet
            timestamp=timestamp,
        )
    else:
        # If task is None, log this condition to help with debugging
        print(
            f"Warning: received task with None task object, task_id={task_id}, args={args}, kwargs={kwargs}"
        )


@signals.task_success.connect
def task_success_handler(sender=None, task_id=None, result=None, **_):
    """Log when a task has successfully completed."""
    timestamp = datetime.now(
        timezone.utc
    )  # Get current UTC time as a timezone-aware datetime

    # Access the routing key from task's delivery info
    routing_key = sender.request.delivery_info.get("routing_key", "default")

    # Get the queue name based on the routing key
    queue_name = get_queue_name_from_routing_key(routing_key)

    # Get task_id from the sender object if not provided
    task_id = sender.request.id if not task_id else task_id

    log_task(
        task_name=sender.name,  # Use task's name
        task_id=task_id,  # Ensure task_id is taken from task object
        queue_name=queue_name,
        inputs=sender.request.args,  # Use the arguments from the task request
        output=result,
        timestamp=timestamp,
    )


@signals.task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **_):
    """Log when a task has failed."""
    timestamp = datetime.now(
        timezone.utc
    )  # Get current UTC time as a timezone-aware datetime

    # Access the routing key from task's delivery info
    routing_key = sender.request.delivery_info.get("routing_key", "default")

    # Get the queue name based on the routing key
    queue_name = get_queue_name_from_routing_key(routing_key)

    # Get task_id from the sender object if not provided
    task_id = sender.request.id if not task_id else task_id

    log_task(
        task_name=sender.name,  # Use task's name
        task_id=task_id,  # Ensure task_id is taken from task object
        queue_name=queue_name,
        inputs=sender.request.args,  # Use the arguments from the task request
        output=str(exception),
        timestamp=timestamp,
    )
