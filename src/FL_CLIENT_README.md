# Thoth FL Client

Federated Learning client for Thoth devices that integrates with Flower (flwr) framework.

## Overview

The FL client enables Thoth devices to participate in federated learning training sessions:
- Connects to a Flower server when online
- Trains on local data stored in `thoth/data/`
- Downloads assigned training data from the server (if configured)
- Reports training metrics back to the server

## Architecture

```
thoth/src/fl_client/
├── __init__.py      # Module exports
├── config.py        # Configuration dataclasses
├── client.py        # Main FLClient and ThothFlowerClient
├── data_loader.py   # FLDataLoader for local data
└── trainer.py       # LocalTrainer for model training
```

## Quick Start

```python
from thoth.src.fl_client import FLClient, FLClientConfig

# Create configuration
config = FLClientConfig(
    server_address="localhost:8080",
    device_id="thoth-001",
)

# Create and start client
client = FLClient(config)

# Check if server is reachable
status = client.check_server()
if status["reachable"]:
    # Check data availability
    data_status = client.check_data_availability()
    if data_status["available"]:
        # Start FL participation
        client.start(blocking=True)
```

## Configuration

### FLClientConfig

```python
FLClientConfig(
    # Server connection
    server_url="http://localhost:8080",
    server_address="localhost:8080",  # For Flower gRPC
    
    # Device identification
    device_id="thoth-001",
    
    # Data configuration
    data_dir="/path/to/data",
    sensor_types=["imu", "csi"],
    min_samples_for_training=100,
    
    # Model configuration
    model=ModelConfig(
        architecture="lstm_mini",
        num_classes=4,
        input_channels=6,
        seq_length=128,
    ),
    
    # Training configuration
    training=TrainingConfig(
        local_epochs=5,
        batch_size=32,
        learning_rate=0.001,
    ),
    
    # FL algorithm (set by server)
    algorithm="fedavg",
)
```

### Supported Algorithms

| Algorithm | Description | Proximal Term |
|-----------|-------------|---------------|
| `fedavg` | Federated Averaging | No |
| `fedprox` | FedProx with proximal term | Yes |
| `fedadam` | Adaptive FL with Adam | No |
| `fedyogi` | Adaptive FL with Yogi | No |
| `fedadagrad` | Adaptive FL with Adagrad | No |
| `fedavgm` | FedAvg with momentum | No |
| `fedmedian` | Byzantine-robust median | No |
| `krum` | Byzantine-robust Krum | No |
| `qfedavg` | Fair FL | No |
| `feddf` | Knowledge distillation | No |
| `fedmd` | Model distillation | No |

### Supported Models

Models are sourced from `Brain/server/dl_models/` for consistency:

| Model | Input Shape | Description |
|-------|-------------|-------------|
| `lstm_mini` | 3D (batch, seq, feat) | Bidirectional LSTM |
| `gru_mini` | 3D (batch, seq, feat) | Bidirectional GRU |
| `cnn1d_mini` | 3D (batch, seq, feat) | 1D CNN |
| `transformer_mini` | 3D (batch, seq, feat) | Transformer encoder |

## Data Requirements

The FL client uses data from `thoth/data/` that:
1. Has corresponding `.meta.json` metadata files
2. Has `fl_status.available_for_training = true`
3. Has valid labels for supervised learning

### Minimum Data Requirements
- Default: 100 samples minimum
- Configurable via `min_samples_for_training`

## Integration with Brain Server

The FL client connects to the Brain server's FL module:

```
Brain/server/fl/
├── core/           # Config, models, client
├── algorithms/     # FL strategies
├── datasets/       # Data loading
├── experiments/    # Experiment runner
└── session.py      # Session manager
```

### Server-Side Setup

```python
# On Brain server
from server.fl import FLSessionManager, create_experiment

# Create FL session
experiment = create_experiment(
    name="HAR-FedAvg",
    algorithm="fedavg",
    model="lstm_mini",
    num_rounds=100,
)

# Start server
fl_manager = FLSessionManager()
session = fl_manager.create_session(experiment)
fl_manager.start_session(session.session_id)
```

### Client-Side Connection

```python
# On Thoth device
from thoth.src.fl_client import FLClient, FLClientConfig

config = FLClientConfig(
    server_address="brain-server:8080",
)
client = FLClient(config)
client.start()
```

## Error Handling

The client handles common errors:

```python
# Check server connectivity
status = client.check_server()
if not status["reachable"]:
    print(f"Server unreachable: {status['error']}")

# Check data availability
data_status = client.check_data_availability()
if not data_status["available"]:
    print(f"Insufficient data: {data_status['total_samples']} samples")
    print(f"Required: {data_status['min_required']} samples")
```

## Offline Mode

When the server is unreachable, the client can:
1. Continue collecting data locally
2. Queue data for future FL participation
3. Train locally (centralized) on collected data

## Dependencies

```
flwr>=1.5.0
torch>=2.0.0
numpy>=1.24.0
```

Install with:
```bash
pip install flwr torch numpy
```

## Extending the Client

### Adding New Model Architectures

1. Add model to `Brain/server/dl_models/`
2. Register with `DLModelRegistry`
3. Model will be automatically available to FL client

### Adding New Sensors

1. Create sensor in `thoth/src/sensors/`
2. Register with `SensorRegistry`
3. Add sensor type to `FLClientConfig.sensor_types`
