# 🧠 YOLOv11 Smart Detection Portal

A comprehensive AI-powered image analysis platform that combines **Azure Functions**, **YOLOv11 models**, and **Streamlit** to provide intelligent dress code compliance and dustbin detection services.

## 🚀 Features

- **Dual Detection Models**: Specialized YOLOv11 models for dress code analysis and dustbin detection
- **Cloud-Native Architecture**: Built on Azure Functions with blob storage integration
- **Interactive Web Interface**: User-friendly Streamlit application for image uploads
- **Real-time Processing**: Fast inference with comprehensive result visualization
- **MongoDB Integration**: Persistent logging and data management
- **Multi-format Support**: Handles JPG, JPEG, and PNG image formats

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Azure Function  │───▶│  YOLOv11 Models │
│     (app.py)    │    │  (Upload_image)  │    │   (Inference)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Azure Blob     │
                       │    Storage       │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │    MongoDB       │
                       │   (Logging)      │
                       └──────────────────┘
```

## 📁 Project Structure

```
Azure/
├── pazofunc/                    # Main Azure Functions project
│   ├── Upload_image/           # Primary function for image processing
│   ├── dresscode_analysis/     # Dress code specific function
│   ├── dustbin_detection/      # Dustbin detection function
│   ├── models/                 # YOLOv11 model files
│   │   ├── deepfashion2_yolov8s-seg.pt
│   │   ├── dustbin_yolo11_best.pt
│   │   ├── yolo11m-seg.pt
│   │   └── yolov11_fashipnpedia.pt
│   ├── scripts/               # Inference and utility scripts
│   ├── runs/                  # Model prediction outputs
│   ├── app.py                 # Streamlit web interface
│   ├── requirements.txt       # Python dependencies
│   └── host.json             # Azure Functions configuration
└── Hello/                     # Sample function
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Azure CLI
- Azure Functions Core Tools
- MongoDB instance
- Azure Storage Account

### 1. Clone Repository

```bash
git clone <repository-url>
cd Azure/pazofunc
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file or set environment variables:

```bash
# Azure Storage
AZURE_STORAGE_CONNECTION_STRING="your_storage_connection_string"
BLOB_CONTAINER_NAME="images"

# MongoDB
MONGO_URI="your_mongodb_connection_string"
MONGO_DB="yolov11db"
MONGO_COLLECTION="yolov11-collection"

# YOLO Inference Endpoint
YOLO_ENDPOINT="your_yolo_inference_endpoint"

# Azure Function
AZURE_FUNCTION_KEY="your_function_key"
```

### 4. Deploy Azure Function

```bash
func azure functionapp publish your-function-app-name
```

## 🚀 Usage

### Web Interface (Streamlit)

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the portal:** Open `http://localhost:8501`

3. **Upload images:**
   - **Dress Code Section**: Upload images for dress code compliance analysis
   - **Dustbin Section**: Upload images for dustbin detection

4. **Submit for analysis:** Click "🚀 Submit All" to process images

### API Endpoints

#### Upload & Analyze Image
```http
POST /api/Upload_image
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPG, JPEG, PNG)
```

**Response:**
```json
{
  "status": "success",
  "blob_url": "https://storage.blob.core.windows.net/images/filename.jpg",
  "prediction": {
    "detections": [...],
    "confidence": 0.95,
    "processing_time": 1.23
  }
}
```

## 🤖 Models

### Dress Code Detection
- **Model**: `deepfashion2_yolov8s-seg.pt`, `yolov11_fashipnpedia.pt`
- **Purpose**: Analyzes clothing compliance and dress code violations
- **Output**: Segmentation masks and classification results

### Dustbin Detection
- **Model**: `dustbin_yolo11_best.pt`
- **Purpose**: Identifies and locates dustbins in images
- **Output**: Bounding boxes with confidence scores

### General Object Detection
- **Model**: `yolo11m-seg.pt`, `yolo11n.pt`
- **Purpose**: Multi-class object detection and segmentation
- **Output**: Comprehensive object analysis

## 🔧 Configuration

### Azure Functions Settings

```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true,
        "excludedTypes": "Request"
      }
    }
  },
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.*, 5.0.0)"
  }
}
```

### Routing Logic

Images are automatically routed based on filename prefixes:
- `dresscode_*` → Dress code analysis endpoint
- `dustbin_*` → Dustbin detection endpoint
- Others → General inference endpoint

## 📊 Monitoring & Logging

- **Azure Application Insights**: Performance and error monitoring
- **MongoDB Logging**: Persistent storage of processing results
- **Streamlit Logs**: Real-time processing status in web interface

## 🔒 Security

- Environment-based configuration management
- Azure Function key authentication
- TLS encryption for MongoDB connections
- Content-type validation for uploaded files

## 🚀 Deployment

### Local Development
```bash
func start
streamlit run app.py
```

### Production Deployment
```bash
# Deploy Azure Function
func azure functionapp publish your-function-app

# Deploy Streamlit (example with Azure Container Instances)
docker build -t yolov11-portal .
docker run -p 8501:8501 yolov11-portal
```

## 📈 Performance

- **Processing Time**: ~1-3 seconds per image
- **Supported Formats**: JPG, JPEG, PNG
- **Concurrent Processing**: Multiple images in single request
- **Scalability**: Auto-scaling Azure Functions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the logs in Azure Application Insights
- Review MongoDB collections for processing history

## 🔄 Version History

- **v1.0.0**: Initial release with dress code and dustbin detection
- **v1.1.0**: Added Streamlit web interface
- **v1.2.0**: Enhanced model routing and error handling

---

**Powered by YOLOv11 • Azure Functions • Streamlit UI**