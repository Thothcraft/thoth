# Thoth Research Portal API Plan

This document outlines the API endpoints and interface plan for the Thoth research device portal.

## Backend API Endpoints

### System Status
| Endpoint | Method | Description | Auth Required | Response |
|----------|--------|-------------|---------------|----------|
| `/health` | GET | System status and battery level | No | `{"status": "ok", "battery": 85, "wifi_connected": true, "collection_active": false}` |
| `/data/current` | GET | Latest IMU reading | No | `{"timestamp": "...", "imu": {...}, "accel": {...}}` |

### Data Management
| Endpoint | Method | Description | Auth Required | Response |
|----------|--------|-------------|---------------|----------|
| `/data/history` | GET | Paginated sensor logs | No | Array of sensor readings |
| `/data/stream` | WebSocket | Live IMU data feed | Optional | Real-time sensor events |
| `/upload` | POST | Upload batch data to server | Yes | `{"success": true, "uploaded": 1000}` |

### Device Control
| Endpoint | Method | Description | Auth Required | Response |
|----------|--------|-------------|---------------|----------|
| `/control/start` | POST | Start data collection | Yes | `{"status": "started"}` |
| `/control/stop` | POST | Stop data collection | Yes | `{"status": "stopped"}` |
| `/config/button` | GET/POST | Get/set button actions | Yes | `{"single": "toggle_collection", "double": "start_ap"}` |

### Network Configuration
| Endpoint | Method | Description | Auth Required | Response |
|----------|--------|-------------|---------------|----------|
| `/wifi-config` | POST | Configure WiFi from captive portal | No | `{"status": "connecting"}` |
| `/network/status` | GET | Network connection status | No | `{"connected": true, "ssid": "MyWiFi", "ip": "192.168.1.100"}` |
| `/network/scan` | GET | Scan for available networks | No | Array of network objects |

## Research Portal Interface (Frontend)

### Technology Stack
- **Framework**: React 18+ with TypeScript
- **Visualization**: Chart.js or D3.js for real-time plots
- **Communication**: Axios for REST API, Socket.IO for WebSocket
- **Styling**: Tailwind CSS or Material-UI
- **Deployment**: Vercel/Netlify with environment-based backend URL

### Pages and Components

#### 1. Dashboard (`/`)
**Purpose**: Real-time monitoring and quick controls

**Components**:
- **Status Card**: Battery level, WiFi status, collection state
- **Live Charts**: 
  - 3D orientation visualization (pitch/roll/yaw)
  - Real-time line charts for accelerometer data
  - Gyroscope and magnetometer readings
- **Quick Controls**: Start/Stop collection buttons
- **Data Summary**: Total readings, collection duration

**WebSocket Integration**:
```javascript
const socket = io(process.env.REACT_APP_BACKEND_URL);
socket.on('imu_data', (data) => {
  updateCharts(data);
  updateOrientationVisualization(data.imu);
});
```

#### 2. Data Explorer (`/data`)
**Purpose**: Browse and analyze historical data

**Components**:
- **Data Table**: Paginated sensor readings with filters
- **Export Tools**: Download CSV/JSON formats
- **Time Range Selector**: Filter by date/time
- **Statistical Summary**: Min/max/average values
- **Visualization Options**: 
  - Scatter plots for motion patterns
  - Frequency analysis charts
  - Activity detection visualization

**Features**:
- Search and filter by timestamp, sensor values
- Bulk data operations (delete, export)
- Data quality indicators

#### 3. Upload Center (`/upload`)
**Purpose**: Manage data uploads to research servers

**Components**:
- **Upload Configuration**: Server URL, API key settings
- **Batch Upload**: Select date ranges for upload
- **Upload History**: Track successful/failed uploads
- **Progress Indicators**: Real-time upload status
- **Retry Mechanisms**: Handle failed uploads

**Upload Flow**:
```javascript
const uploadData = async (dateRange) => {
  const response = await axios.post('/upload', {
    upload_url: serverUrl,
    date_range: dateRange
  });
  updateUploadStatus(response.data);
};
```

#### 4. Training Interface (`/training`) - Future Enhancement
**Purpose**: Machine learning model training and evaluation

**Components**:
- **Data Labeling**: Annotate activities/movements
- **Model Training**: Configure and train ML models
- **Evaluation Metrics**: Accuracy, confusion matrix
- **Model Deployment**: Deploy trained models to device

**ML Integration**:
- TensorFlow.js for client-side training
- Server-side training with scikit-learn/PyTorch
- Activity recognition models (walking, running, sitting)

#### 5. Configuration (`/config`)
**Purpose**: Device and system configuration

**Components**:
- **Button Mapping**: Configure PiSugar button actions
- **Sensor Settings**: Sample rate, calibration
- **Network Settings**: WiFi credentials, AP configuration
- **Data Retention**: Storage limits, cleanup policies
- **System Info**: Hardware status, software versions

### API Integration Examples

#### Real-time Data Visualization
```javascript
// Chart.js integration for live IMU data
const updateIMUChart = (data) => {
  chart.data.labels.push(new Date(data.timestamp));
  chart.data.datasets[0].data.push(data.imu.pitch);
  chart.data.datasets[1].data.push(data.imu.roll);
  chart.data.datasets[2].data.push(data.imu.yaw);
  
  // Keep only last 100 points
  if (chart.data.labels.length > 100) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(dataset => dataset.data.shift());
  }
  
  chart.update('none');
};
```

#### Data Export Functionality
```javascript
const exportData = async (format, dateRange) => {
  const response = await axios.get('/data/history', {
    params: {
      start_date: dateRange.start,
      end_date: dateRange.end,
      limit: 10000
    }
  });
  
  if (format === 'csv') {
    downloadCSV(response.data);
  } else {
    downloadJSON(response.data);
  }
};
```

### Deployment Architecture

#### Development Setup
```bash
# Frontend development
cd portal-frontend
npm install
npm start  # Runs on localhost:3000

# Backend connection
REACT_APP_BACKEND_URL=http://raspberrypi.local:5000
```

#### Production Deployment
1. **Frontend**: Deploy to Vercel/Netlify
2. **Backend Access**: 
   - Direct IP access for local network
   - VPN/Tailscale for remote access
   - ngrok tunnel for temporary external access

#### Security Considerations
- JWT authentication for sensitive endpoints
- CORS configuration for cross-origin requests
- Rate limiting for API endpoints
- HTTPS for production deployments

### Future Enhancements

#### Multi-Device Support
- Device discovery and management
- Synchronized data collection
- Distributed sensor networks

#### Advanced Analytics
- Real-time anomaly detection
- Predictive maintenance alerts
- Behavioral pattern analysis

#### Integration Options
- MQTT broker for IoT ecosystems
- InfluxDB for time-series data
- Grafana dashboards for monitoring
- Cloud storage integration (AWS S3, Google Cloud)

### Mobile Application
Consider developing a companion mobile app using:
- React Native for cross-platform development
- Flutter for native performance
- Progressive Web App (PWA) for web-based mobile experience

The mobile app would provide:
- Remote device monitoring
- Quick data collection controls
- Offline data viewing
- Push notifications for device status
